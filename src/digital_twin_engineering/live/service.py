"""The twin service: one Twin per case being replayed, from MQTT to InfluxDB.

`TwinService.handle` takes one message (see messages.py) and writes what follows
from it to a sink; `serve` feeds it from an MQTT broker. Several cases can run at
once (a Digital Twin Aggregate): each gets its own twin when its `meta` arrives.

Measurements written, all tagged with the case:
- vitals: the twin's grid values of the monitored channels, every second
- expert: each expert's state (tag expert), every second
- forecast: each forecast placed at the time it is for (tag horizon), so it can be
  drawn over what was later measured
- warning: hypotension probabilities per horizon and the alert, at decision points
- suggestion: per drug (tag drug) the change probabilities, proposed settings and
  their expected effects
- forecast_fidelity: rolling and whole-case RMSE of the forecasts that have come
  due, against persistence (tags signal, horizon)
- fidelity: Varvel metrics of each expert against the pump's Ce and the measured
  BIS and MAP, over the last window and the whole case (tags expert, signal, window)
- deviation: measured value minus the 1-min forecast made a minute earlier; flagged
  when larger than `deviation_k` test RMSEs with no target change in the last
  2 minutes (an unexplained deviation, such as the first incision). At k = 3 about
  1.5% of decision points are flagged per signal on test patients; in the 5 min
  after incision 5% for HR, but hardly more than elsewhere for BIS and MAP.
"""

import json
import math
import queue
import time
from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

from ..channels import STALENESS
from ..config import Config
from ..patients import Covariates
from ..targets import TARGETS
from ..verification import COMPARISONS, varvel
from . import messages
from .influx import Point
from .twin import Phases, Twin

VITALS = (
    "map",
    "sbp",
    "dbp",
    "hr",
    "bis",
    "sqi",
    "spo2",
    "etco2",
    "nibp_map",
    "propofol_target",
    "remifentanil_target",
    "propofol_ce_pump",
    "remifentanil_ce_pump",
)
EXPERT_SIGNALS = ("bis", "map", "hr", "propofol_ce", "remifentanil_ce")
REFERENCE_FRESH_S = 5  # as verification
CHANGE_LOOKBACK_S = 120
SETTINGS = ("propofol_target", "remifentanil_target")


@dataclass
class Case:
    twin: Twin
    start: float  # epoch s of case time 0
    issued: dict[float, dict] = field(default_factory=dict)  # decision t -> decision
    pending: deque = field(default_factory=deque)  # (due t, horizon, forecasts, persistence)
    errors: dict = field(default_factory=dict)  # (signal, horizon) -> [(model, persistence)]
    pairs: dict = field(default_factory=dict)  # (expert, signal) -> [(t, reference, expert)]


def from_run(run: str, sink, alert_threshold: float | None = None, **kwargs) -> "TwinService":
    """A service with a training run's models. The alert threshold defaults to the
    5-min hypotension probability that alerted at 10% of test decision points."""
    from ..pipeline import MODELS_DIR
    from .replayer import load_run

    config, models = load_run(run)
    clean = json.loads((MODELS_DIR / run / "metrics.json").read_text())["clean"]
    # regression targets only: classification targets (hypotension) have no RMSE
    rmse = {s: m["rmse"] for s in config.targets if "rmse" in (m := clean.get(f"y.{s}@60", {}).get("model", {}))}
    if alert_threshold is None:
        alert_threshold = clean.get("y.hypotension@300", {}).get("model", {}).get("threshold@10%_alerts")
        if alert_threshold is None:
            raise ValueError(f"run {run} predates alert thresholds in metrics.json; pass one")
    return TwinService(config, models, sink, rmse, alert_threshold, **kwargs)


class TwinService:
    def __init__(
        self,
        config: Config,
        models: dict,
        sink,
        rmse_1min: dict[str, float],
        alert_threshold: float,
        deviation_k: float = 3.0,
        window_s: float = 600,
    ):
        """`rmse_1min`: each value target's test RMSE at 1 min (sets the deviation
        threshold); `alert_threshold`: the 5-min hypotension probability that alerts."""
        self.config = config
        self.models = models
        self.sink = sink
        self.rmse_1min = rmse_1min
        self.values = [name for name in config.targets if TARGETS[name].kind == "value"]
        self.alert_threshold = alert_threshold
        self.deviation_k = deviation_k
        self.window_s = window_s
        self.cases: dict[int, Case] = {}

    def handle(self, topic: str, payload: dict) -> None:
        caseid, kind, track = messages.parse(topic)
        if kind == "meta":
            if payload and caseid not in self.cases:
                self.cases[caseid] = self._open(payload)
            return
        case = self.cases.get(caseid)
        if case is None:
            return
        if kind == "end":
            del self.cases[caseid]
            self.sink.flush()
            return
        t = payload["t"] - case.start
        match kind:
            case "track":
                case.twin.observe(track, t, payload["v"])
            case "event" if payload["name"] in messages.PHASES:
                case.twin.phases = replace(case.twin.phases, **{payload["name"]: t})
            case "clock":
                self.sink.write(self._tick(caseid, case, t))

    def _open(self, meta: dict) -> Case:
        # phases not yet announced: anaesthesia hasn't started, surgery neither
        phases = Phases(anestart=math.nan, opstart=math.inf, opend=math.inf)
        covariates = Covariates(**meta["covariates"])
        twin = Twin(self.config, self.models, covariates, meta["tracks"], phases)
        return Case(twin, meta["start"])

    def _tick(self, caseid: int, case: Case, t: float) -> list[Point]:
        decision = case.twin.tick(t)
        row = case.twin.buffer[-1]
        at = case.start + t
        tags = {"case": str(caseid)}
        points = [Point("vitals", tags, {k: row[k] for k in VITALS if k in row}, at)]
        for runner in case.twin.runners:
            name = runner.expert.name
            fields = {s: row[f"{name}.{s}"] for s in EXPERT_SIGNALS if f"{name}.{s}" in row}
            points.append(Point("expert", tags | {"expert": name}, fields, at))
        self._pair(case, row, t)
        if decision is not None:
            points += self._decide(case, decision, row, t, tags)
        return points

    def _pair(self, case: Case, row: dict, t: float) -> None:
        if not t >= case.twin.phases.anestart:
            return
        for runner in case.twin.runners:
            for signal, reference in COMPARISONS:
                column = f"{runner.expert.name}.{signal}"
                if column in row and row.get(reference + STALENESS, np.inf) <= REFERENCE_FRESH_S:
                    key = (runner.expert.name, signal)
                    case.pairs.setdefault(key, []).append((t, row[reference], row[column]))

    def _decide(self, case: Case, d: dict, row: dict, t: float, tags: dict) -> list[Point]:
        at = case.start + t
        horizons = self.config.horizons_s
        points = []
        for h in horizons:
            fields = {k.removesuffix(f"@{h}"): v for k, v in d.items() if k.endswith(f"@{h}") and "." not in k}
            points.append(Point("forecast", tags | {"horizon": str(h)}, fields, at + h))
        risk = {f"p@{h}": d[f"hypotension@{h}"] for h in horizons if f"hypotension@{h}" in d}
        if "p@300" in risk:
            risk["alert"] = bool(risk["p@300"] >= self.alert_threshold)
        risk["map_below_65"] = bool(row.get("map", np.nan) < 65)
        points.append(Point("warning", tags, risk, at))
        for drug in {k.split(".")[1] for k in d if k.startswith("suggest.")}:
            prefix = f"suggest.{drug}."
            fields = {k.removeprefix(prefix): v for k, v in d.items() if k.startswith(prefix)}
            points.append(Point("suggestion", tags | {"drug": drug}, fields, at))

        # forecasts come due: score them against what was measured, and against persistence
        now = {s: _measured(case.twin, s) for s in self.values}
        case.issued[t] = d
        for h in horizons:
            case.pending.append((t + h, h, {s: d.get(f"{s}@{h}", np.nan) for s in self.values}, now))
        while case.pending and case.pending[0][0] <= t:
            _, h, forecast, persistence = case.pending.popleft()
            for s in self.values:
                if np.isfinite(now[s] + forecast[s] + persistence[s]):
                    errors = case.errors.setdefault((s, h), [])
                    errors.append((t, now[s] - forecast[s], now[s] - persistence[s]))
        case.pending = deque(sorted(case.pending, key=lambda p: p[0]))
        for (s, h), errors in case.errors.items():
            e = np.array(errors)
            recent = e[e[:, 0] > t - self.window_s]
            fields = {
                "rmse_window": _rmse(recent[:, 1]),
                "rmse_window_persistence": _rmse(recent[:, 2]),
                "rmse_case": _rmse(e[:, 1]),
                "rmse_case_persistence": _rmse(e[:, 2]),
                "n_case": len(e),
            }
            points.append(Point("forecast_fidelity", tags | {"signal": s, "horizon": str(h)}, fields, at))

        for (expert, signal), pairs in case.pairs.items():
            p = np.array(pairs)
            for window, sel in (("window", p[p[:, 0] > t - self.window_s]), ("case", p)):
                fields = varvel(sel[:, 1], sel[:, 2]) if len(sel) else {}
                points.append(
                    Point("fidelity", tags | {"expert": expert, "signal": signal, "window": window}, fields, at)
                )

        earlier = case.issued.pop(t - 60, None)
        for old in [k for k in case.issued if k < t - 60]:
            del case.issued[old]
        if earlier is not None:
            changed = _changed(case.twin, CHANGE_LOOKBACK_S)
            for s in self.values:
                if f"{s}@60" not in earlier or not np.isfinite(now[s]):
                    continue
                residual = now[s] - earlier[f"{s}@60"]
                threshold = self.deviation_k * self.rmse_1min.get(s, np.inf)
                fields = {
                    "residual": residual,
                    "threshold": threshold,
                    "target_changed": changed,
                    "flagged": bool(abs(residual) > threshold and not changed),
                }
                points.append(Point("deviation", tags | {"signal": s}, fields, at))
        return points


def _measured(twin: Twin, name: str) -> float:
    """A value target as measured now: its channel's fresh reading, or for a
    smoothed target the median of the fresh readings over its window just past
    (the target is centred on its time, so this lags it by half the window)."""
    target = TARGETS[name]
    n = round((target.smooth_s or twin.config.step_s) / twin.config.step_s)
    recent = [
        r[target.channel]
        for r in list(twin.buffer)[-n:]
        if r.get(target.channel + STALENESS, np.inf) <= target.max_staleness_s
    ]
    if not recent or len(recent) < n // 2:
        return math.nan
    return float(np.median(recent))


def _rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2))) if len(x) else math.nan


def _changed(twin: Twin, lookback_s: float) -> bool:
    """Whether a TCI target changed within the last `lookback_s` grid steps."""
    recent = list(twin.buffer)[-round(lookback_s / twin.config.step_s) - 1 :]
    for s in SETTINGS:
        values = np.array([r.get(s, np.nan) for r in recent], float)
        values = values[np.isfinite(values)]
        if len(values) and values.min() != values.max():
            return True
    return False


def serve(service: TwinService, host: str, port: int = 1883, once: bool = False) -> None:
    """Run `service` on the messages of an MQTT broker; with `once`, return after
    the first case has ended and no case is left."""
    import paho.mqtt.client as mqtt

    inbox: queue.Queue = queue.Queue()
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="twin-service")
    client.on_connect = lambda c, userdata, flags, rc, props: c.subscribe(f"{messages.PREFIX}/#", qos=1)
    client.on_message = lambda c, userdata, m: inbox.put((m.topic, m.payload))
    client.connect(host, port)
    client.loop_start()
    seen = False
    flushed = time.monotonic()
    try:
        while True:
            try:
                topic, payload = inbox.get(timeout=0.5)
            except queue.Empty:
                service.sink.flush()
                continue
            service.handle(topic, messages.decode(payload))
            seen |= bool(service.cases)
            if time.monotonic() - flushed > 1:
                service.sink.flush()
                flushed = time.monotonic()
            if once and seen and not service.cases:
                return
    finally:
        service.sink.flush()
        client.loop_stop()
        client.disconnect()
