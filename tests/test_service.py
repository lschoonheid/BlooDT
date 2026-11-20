"""The live layer end to end: replayer -> MQTT -> twin service -> InfluxDB."""

import asyncio
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from digital_twin_engineering import stages, vitaldb_cache
from digital_twin_engineering.config import Config
from digital_twin_engineering.live import influx, replayer, service
from digital_twin_engineering.live.influx import Point

CASE = 45
START = 1_700_000_000
NO_MODELS = {"forecasters": [], "suggesters": []}

needs_case = pytest.mark.skipif(
    not (vitaldb_cache.CACHE_DIR / f"vital_files/{CASE:04d}.vital").exists(),
    reason="VitalDB case not downloaded",
)


def small_config() -> Config:
    return Config.from_dict(
        {
            "channels": ["map", "hr", "bis", "propofol_ce_pump"],
            "experts": [{"kind": "anesthesim", "name": "asim", "baseline": "pre_dose"}],
            "features": {"windows_s": [60, 300]},
        }
    )


def test_line_protocol():
    p = Point("vitals", {"case": "1", "expert": "a b"}, {"map": 70.0, "bis": np.nan, "alert": True}, 1.5)
    assert influx.line(p) == "vitals,case=1,expert=a\\ b map=70.0,alert=true 1500000000"
    assert influx.line(Point("vitals", {}, {"map": np.nan}, 0)) is None


@needs_case
def test_service_twin_matches_in_process_twin(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "DERIVED_DIR", tmp_path)
    config = small_config()
    msgs = replayer.stream(CASE, config, START, until=1200)
    svc = service.TwinService(config, NO_MODELS, influx.MemorySink(), {}, alert_threshold=0.5)
    for _, topic, payload in msgs[:-1]:  # all but `end`, which closes the twin
        svc.handle(topic, payload)
    live = svc.cases[CASE].twin

    tracks, recs, _ = replayer.records(CASE, list(config.all_channels))
    direct = replayer.make_twin(CASE, config, NO_MODELS, tracks)
    i = 0
    for t in np.arange(0, 1200, 1.0):
        while i < len(recs) and recs[i][0] <= t:
            direct.observe(recs[i][1], recs[i][0], recs[i][2])
            i += 1
        direct.tick(t)
    a, b = live.frame(), direct.frame()
    # times go through epoch seconds and back: equal to well under a millisecond
    np.testing.assert_allclose(a.to_numpy(), b[a.columns].to_numpy(), rtol=1e-6, atol=1e-5)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def broker():
    amqtt = pytest.importorskip("amqtt.broker")
    port = _free_port()
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()

    async def start():
        b = amqtt.Broker({"listeners": {"default": {"type": "tcp", "bind": f"127.0.0.1:{port}"}}})
        await b.start()
        return b

    b = asyncio.run_coroutine_threadsafe(start(), loop).result(10)
    yield port
    asyncio.run_coroutine_threadsafe(b.shutdown(), loop).result(10)
    loop.call_soon_threadsafe(loop.stop)


@pytest.fixture
def influxdb():
    """An HTTP server taking InfluxDB writes: yields (url, received bodies)."""
    bodies: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path.startswith("/api/v2/write?")
            assert self.headers["Authorization"] == "Token t"
            bodies.append(self.rfile.read(int(self.headers["Content-Length"])).decode())
            self.send_response(204)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", bodies
    server.shutdown()


@needs_case
def test_replay_over_mqtt_reaches_influx(tmp_path, monkeypatch, broker, influxdb):
    monkeypatch.setattr(stages, "DERIVED_DIR", tmp_path)
    url, bodies = influxdb
    config = small_config()
    sink = influx.InfluxSink(url, "t", "twin", "twin", batch=1000)
    svc = service.TwinService(config, NO_MODELS, sink, {}, alert_threshold=0.5)
    served = threading.Thread(target=service.serve, args=(svc, "127.0.0.1", broker, True))
    served.start()
    replayer.replay_mqtt(CASE, config, "127.0.0.1", broker, speed=np.inf, start=START, until=300)
    served.join(60)
    assert not served.is_alive()

    lines = [line for body in bodies for line in body.splitlines()]
    states = [line for line in lines if line.startswith(f"expert,case={CASE},expert=asim ")]
    assert len(states) == 300  # one per second: every clock tick got through, in order
    times = [int(line.rsplit(" ", 1)[1]) for line in states]
    assert times == sorted(times) and times[0] == START * 10**9
    assert any(line.startswith(f"vitals,case={CASE} ") for line in lines)
