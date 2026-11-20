"""Expert ported from AnestheSIM 1.0.0 (MATLAB/Simulink; github.com/izabirs/AnestheSIM).

Copyright (c) 2026 Isabela Birs. Ported under the MIT License; the full licence text
is the LICENSE file of the AnestheSIM repository, and this notice must be kept.

The port follows simulator/build_anesthesim_model.m: covariate PK models
(Schnider, Minto, Joachim norepinephrine, volatile effect sites) feed a BIS
interaction surface and a MAP/HR model. Deviations from the Simulink model:

- Linear compartments advance with an exact zero-order-hold step (as simulateLTI.m)
  instead of ode15s; the turnover MAP/HR model uses an exponential step with
  concentrations and HR held at their step midpoints.
- AnestheSIM takes MAP0/HR0 from the pre-induction recording (estimateBaselines.m),
  which is not causal. Here MAP0 defaults to that file's covariate fallback and HR0
  to the published 70.8 bpm; `baseline="pre_dose"` takes them from what was observed
  before the first dose instead (see baseline.py), and both can be overridden.
- AnestheSIM starts the noxious-stimulation term of the turnover model at t = 0.
  The incision time is not known causally, so stimulation is off unless
  `stim_onset_s` is given.
- `calibrate_bis` fits the patient's hypnotic sensitivity online (calibration.py):
  the BIS surface sees the hypnotic concentrations times a factor that the
  measured BIS keeps adjusting; `calibrate_map` likewise scales the unified MAP
  model's propofol effect. Not in AnestheSIM.
"""

import math
from collections import deque
from collections.abc import Mapping

import numpy as np
from scipy.linalg import block_diag, expm

from ..patients import Covariates
from . import baseline as baselines
from .calibration import LogScaleFilter

NE_MOLAR_MASS = 169.18  # g/mol, buildPatientModels.m
MAC40 = {"sevoflurane": 1.8, "desflurane": 6.6}  # vol%, Mapleson 1996 (VolatileModel.m)
MAPLESON_AGE_DECAY = -0.00269  # 1/year, VolatileModel.m

# BIS surfaces, bisParameters.m: (C50, gamma) per drug and the interaction E0, Emax, gamma.
# calibrate_bis defaults: the factor starts at 1 within about x/÷ 1.65, may drift
# ~20%/h; BIS noise is set to 25 (the monitor's ~8 over ~10 correlated samples)
BIS_CALIBRATION = dict(prior_sd=0.5, drift_sd_per_hour=0.2, noise_sd=25.0)
BIS_FRESH_RANGE = (10.0, 98.0)  # readings outside are artefacts or awake noise
# calibrate_map (unified MAP model only): the propofol Emax factor; MAP noise 30 mmHg
# for the same reason (the arterial line's beat-to-beat and artefact swings)
MAP_CALIBRATION = dict(prior_sd=0.7, drift_sd_per_hour=0.2, noise_sd=30.0)
MAP_FRESH_RANGE = (30.0, 160.0)

BIS_PARAMS = {
    "greco": dict(prop=(3.34, 1.2), remi=(2.41, 1.2), e0=95.0, emax=70.0, gamma=1.2),
    "bouillon": dict(prop=(4.47, 1.43), remi=(19.3, 1.43), e0=97.4, emax=97.4, gamma=1.43),
    "minto": dict(prop=(3.35, 1.58), remi=(7.96, 7.85), e0=93.0, emax=83.0, gamma=1.0),
}


def _calibration(defaults: dict, option: bool | dict) -> dict | None:
    if not option:
        return None
    return defaults | (option if isinstance(option, dict) else {})


def _lbm(c: Covariates) -> float:
    """James lean body mass [kg] (Patient.m)."""
    if c.is_male:
        return 1.10 * c.weight - 128 * (c.weight / c.height) ** 2
    return 1.07 * c.weight - 148 * (c.weight / c.height) ** 2


def _three_compartment(v1, v2, v3, cl1, cl2, cl3, ke0):
    """States [C1, C2, C3, Ce], input mass rate; clearances in L/min, ke0 in 1/s.

    threeCompartmentModel.m
    """
    k10, k12, k13 = cl1 / v1 / 60, cl2 / v1 / 60, cl3 / v1 / 60
    k21, k31 = cl2 / v2 / 60, cl3 / v3 / 60
    a = np.array(
        [
            [-(k10 + k12 + k13), k21, k31, 0],
            [k12, -k21, 0, 0],
            [k13, 0, -k31, 0],
            [ke0, 0, 0, -ke0],
        ]
    )
    return a, np.array([1 / v1, 0, 0, 0])


def _schnider(c: Covariates, lbm: float, ke0_per_min: float):
    # SchniderModel.m; ke0 1.21/min (Orchestra/Marsh) instead of Schnider's 0.456.
    age, w, h = c.age, c.weight, c.height
    return _three_compartment(
        4.27,
        18.9 - 0.391 * (age - 53),
        238,
        1.89 + 0.0456 * (w - 77) - 0.0681 * (lbm - 59) + 0.0264 * (h - 177),
        1.29 - 0.024 * (age - 53),
        0.836,
        ke0_per_min / 60,
    )


def _minto(c: Covariates, lbm: float):
    # MintoModel.m
    age = c.age
    return _three_compartment(
        5.1 - 0.0201 * (age - 40) + 0.072 * (lbm - 55),
        9.82 - 0.0811 * (age - 40) + 0.108 * (lbm - 55),
        5.42,
        2.6 - 0.0162 * (age - 40) + 0.0191 * (lbm - 55),
        2.05 - 0.0301 * (age - 40),
        0.076 - 0.00113 * (age - 40),
        (0.595 - 0.007 * (age - 40)) / 60,
    )


def _joachim(c: Covariates, lbm: float, bmi: float) -> dict:
    """Joachim 2024 norepinephrine PK/PD; states [depot, central, peripheral] in nmol.

    JoachimModel.m: smoking and hypertension set to 0, depot non-linearity linearised.
    """
    male = float(c.is_male)
    kel = math.exp(0.64 * math.log(lbm) - 5.52)  # 1/s
    ka = math.exp(-3.31)  # 1/s
    v1, k12, k21 = 0.49, 0.06, 0.04  # L, 1/s, 1/s
    a = np.array([[-ka, 0, 0], [ka, -(k12 + kel), k21], [0, k12, -k21]])
    return dict(
        a=a,
        b=np.array([1.0, 0, 0]),
        v1=v1,
        kel=kel,
        tlag=math.exp(0.26 * male + 3.21),  # s
        cendo=math.exp(0.57 * math.log(c.age) + 0.67 * math.log(bmi) - 4.56),  # nmol/L
        emax=math.exp(-0.46 * male + 4.09),  # mmHg
        ec50=15.7,  # nmol/L
        gamma=1.46,
    )


def covariate_map0(c: Covariates, bmi: float) -> float:
    """Heuristic awake MAP [mmHg] (estimateBaselines.m, covariate fallback)."""
    map0 = 75 + 0.4 * max(0.0, c.age - 30) + 5 * c.is_male + 0.5 * max(0.0, bmi - 25)
    return min(max(map0, 65.0), 115.0)


# What-if targets: the setting channel, dose input, B-matrix input and Ce state per drug.
# Orchestra pumps reach 1200 mL/h: 6.7 mg/s of propofol at 20 mg/mL, 6.7 ug/s of
# remifentanil at 20 ug/mL.
TCI_DRUGS = {
    "propofol": dict(setting="propofol_target", dose="propofol_dose", input=0, ce=3, max_rate=1200 / 3600 * 20),
    "remifentanil": dict(setting="remifentanil_target", dose="remifentanil_dose", input=1, ce=7, max_rate=1200 / 3600 * 20),
}
TCI_LEAD_S = 30  # how soon the TCI aims to reach its target: Ce gets there in ~90 s, overshooting ~1%


def _hill(u: float, gamma: float) -> float:
    ug = max(u, 0.0) ** gamma
    return ug / (1 + ug)


class AnestheSimExpert:
    """AnestheSIM's patient model driven by pump doses and end-tidal volatile levels.

    `bis_model`: "greco" (reduced Greco, the only surface with the volatile agents),
    "bouillon" or "minto" (intravenous only). `map_model`: "separable" (turnover model
    plus Joachim pressor effect) or "unified" (three-drug effect-site model).
    HR always comes from the turnover model.

    `lookahead` gives the signals after holding the current inputs for a while,
    without disturbing the stepped state: the model's own forecast of where the
    patient is heading if nothing changes. With `whatif` factors it also forecasts
    each TCI target scaled by each factor, steering its own PK model to the new
    target like a TCI pump (see `_tci_rate`): what a target change would do.
    """

    version = 1  # 1: what-ifs include the current target (x1)
    inputs = (
        "propofol_dose",
        "remifentanil_dose",
        "norepinephrine_dose",
        "sevoflurane_et",
        "desflurane_et",
    )
    signals = (
        "propofol_cp",  # ug/mL
        "propofol_ce",  # ug/mL
        "remifentanil_cp",  # ng/mL
        "remifentanil_ce",  # ng/mL
        "norepinephrine_cp",  # nmol/L, exogenous only
        "sevoflurane_ce",  # vol%
        "desflurane_ce",  # vol%
        "volatile_mac",  # summed effect-site MAC fraction, age-adjusted
        "bis",
        "map",  # mmHg
        "hr",  # /min
    )

    def __init__(
        self,
        name: str,
        bis_model: str = "greco",
        map_model: str = "separable",
        propofol_ke0: float = 1.21,  # 1/min
        map0: float | None = None,  # mmHg; default: covariate estimate
        hr0: float | None = None,  # /min; default: 70.8
        stim_onset_s: float | None = None,  # s from case start; None: no stimulation
        baseline: str = "covariates",  # or "pre_dose": MAP0/HR0 observed before the first dose
        whatif: tuple[float, ...] = (),  # TCI target factors to forecast, e.g. (0.8, 1.2)
        calibrate_bis: bool | dict = False,  # True, or BIS_CALIBRATION overrides
        calibrate_map: bool | dict = False,  # True, or MAP_CALIBRATION overrides
    ):
        if bis_model not in BIS_PARAMS:
            raise ValueError(f"unknown bis_model {bis_model!r}")
        if map_model not in ("separable", "unified"):
            raise ValueError(f"unknown map_model {map_model!r}")
        self.name = name
        self.bis_model = bis_model
        self.map_model = map_model
        self.propofol_ke0 = propofol_ke0
        self.map0_override = map0
        self.hr0_override = hr0
        self.stim_onset_s = stim_onset_s
        if baseline not in ("covariates", "pre_dose"):
            raise ValueError(f"unknown baseline {baseline!r}")
        self.baseline = baseline
        self.inputs = type(self).inputs
        if baseline == "pre_dose":
            self.inputs += baselines.INPUTS
        # the current target (x1) too: what-if effects are measured against keeping it
        self.whatif = tuple(sorted(set(whatif) | {1.0})) if whatif else ()
        if self.whatif:
            self.inputs += tuple(TCI_DRUGS[d]["setting"] for d in TCI_DRUGS)
        self.calibrate_bis = _calibration(BIS_CALIBRATION, calibrate_bis)
        self.calibrate_map = _calibration(MAP_CALIBRATION, calibrate_map)
        if self.calibrate_map and map_model != "unified":
            raise ValueError("calibrate_map needs map_model unified")
        self.signals = type(self).signals
        if self.calibrate_bis:
            self.inputs += ("bis", "bis__staleness")
            self.signals += ("bis_sensitivity",)
        if self.calibrate_map:
            self.inputs += ("map", "map__staleness")
            self.signals += ("map_sensitivity",)

    def reset(self, covariates: Covariates, step_s: float) -> None:
        c = covariates
        self.step_s = step_s
        lbm = _lbm(c)
        bmi = c.weight / (c.height / 100) ** 2

        # Linear part, states [propofol C1-3 Ce | remi C1-3 Ce | NE depot, central,
        # peripheral | Ce sevo | Ce des], inputs [mg/s, ug/s, nmol/s, vol%, vol%].
        prop_a, prop_b = _schnider(c, lbm, self.propofol_ke0)
        remi_a, remi_b = _minto(c, lbm)
        self.ne = ne = _joachim(c, lbm, bmi)
        vol_ke0 = 0.47 / 60  # 1/s, VolatileModel.m (both agents)
        a = block_diag(prop_a, remi_a, ne["a"], [[-vol_ke0]], [[-vol_ke0]])
        b = np.zeros((13, 5))
        b[0:4, 0], b[4:8, 1], b[8:11, 2] = prop_b, remi_b, ne["b"]
        b[11, 3] = b[12, 4] = vol_ke0
        m = expm(np.block([[a, b], [np.zeros((5, 18))]]) * step_s)
        self.ad, self.bd = m[:13, :13], m[:13, 13:]
        self.x = np.zeros(13)
        self.u = np.zeros(5)
        # Transport lag of the NE input, rounded to the grid (simulateLTI.m).
        self.ne_queue = deque([0.0] * round(ne["tlag"] / step_s))

        mac_age = 10 ** (MAPLESON_AGE_DECAY * (c.age - 40))
        self.mac = {k: v * mac_age for k, v in MAC40.items()}
        # bisParameters.m / VolatileModel.m: sevo C50 1.2 vol% at 40 y, des scaled by MAC ratio.
        self.c50_sevo = 1.2 * mac_age
        self.c50_des = 1.2 * MAC40["desflurane"] / MAC40["sevoflurane"] * mac_age
        self.bis_params = BIS_PARAMS[self.bis_model]

        self.map0 = self.map0_override if self.map0_override is not None else covariate_map0(c, bmi)
        self.hr0 = self.hr0_override if self.hr0_override is not None else 70.8
        # hemodynamicParameters.m (Garraza-Obaldia 2024, plasma concentrations)
        self.kout = 0.126 / 60  # 1/s
        self.decay = math.exp(-self.kout * step_s)
        self.map_turnover, self.hr = self.map0, self.hr0
        self.t = 0.0
        self.pre_dose = baselines.PreDoseBaseline() if self.baseline == "pre_dose" else None
        self.bis_filter = LogScaleFilter(**self.calibrate_bis, step_s=step_s) if self.calibrate_bis else None
        self.bis_scale = 1.0
        self.map_filter = LogScaleFilter(**self.calibrate_map, step_s=step_s) if self.calibrate_map else None
        self.map_scale = 1.0
        # TCI: Ce after TCI_LEAD_S of a constant rate is free + gain * rate
        ad_lead, reach = np.eye(13), np.zeros((13, 5))
        for _ in range(max(1, round(TCI_LEAD_S / step_s))):
            reach = self.ad @ reach + self.bd
            ad_lead = self.ad @ ad_lead
        self.tci = {
            d: (ad_lead[spec["ce"]], reach[spec["ce"], spec["input"]] * step_s)
            for d, spec in TCI_DRUGS.items()
        }

    def _set_baseline(self, map0: float | None, hr0: float | None) -> None:
        """Re-baseline before any drug is given (the model is still at rest)."""
        if map0 is not None and self.map0_override is None:
            self.map0 = min(max(map0, 50.0), 150.0)
        if hr0 is not None and self.hr0_override is None:
            self.hr0 = min(max(hr0, 35.0), 140.0)
        self.map_turnover, self.hr = self.map0, self.hr0

    def _bis(self, ce_p: float, ce_r: float, ce_sevo: float, ce_des: float, scale: float | None = None) -> float:
        # bisMinto.m, bisBouillon.m, bisGrecoVolatile.m; `scale`: hypnotic sensitivity
        p = self.bis_params
        (c50_p, g_p), (c50_r, g_r) = p["prop"], p["remi"]
        scale = self.bis_scale if scale is None else scale
        ce_p, ce_r = max(ce_p, 0.0) * scale, max(ce_r, 0.0)
        ce_sevo, ce_des = ce_sevo * scale, ce_des * scale
        if self.bis_model == "minto":
            s = (ce_p / c50_p) ** g_p + (ce_r / c50_r) ** g_r
            effect = s / (1 + s)
        elif self.bis_model == "bouillon":
            effect = _hill(ce_p / c50_p + ce_r / c50_r, p["gamma"])
        else:
            hypnotic = ce_p / c50_p + max(ce_sevo, 0.0) / self.c50_sevo + max(ce_des, 0.0) / self.c50_des
            effect = _hill(hypnotic * (1 + ce_r / c50_r), p["gamma"])
        return min(max(p["e0"] - p["emax"] * effect, 0.0), 100.0)

    def _ne_emax(self, cc: float) -> float:
        ne = self.ne
        return ne["emax"] * _hill(cc / ne["ec50"], ne["gamma"])

    def _turnover(self, cp_p: float, cp_r: float) -> None:
        """Advance MAP/HR one step (hemodynamicTurnover.m), inputs held at mid-step."""
        cp_p, cp_r = max(cp_p, 0.0), max(cp_r, 0.0)
        f_p = 0.77 * cp_p / (cp_p + 7.38)  # EmaxP, C50P [ug/mL]
        g_r = 0.65 * _hill(cp_r / 3.63, 1.37)  # EmaxR, C50R [ng/mL], gammaR
        hr_ss = self.hr0 * (1 - g_r)
        hr_new = hr_ss + (self.hr - hr_ss) * self.decay
        h = 1.0
        t_mid = self.t + self.step_s / 2
        if self.stim_onset_s is not None and t_mid >= self.stim_onset_s:
            h += 0.277 * math.exp(-0.018 / 60 * (t_mid - self.stim_onset_s))  # Stim, kStim
        hr_mid = max((self.hr + hr_new) / 2, 1e-6)
        production = self.map0 * (1 - f_p) * (hr_mid / self.hr0) ** 0.842 * h  # FBhr
        self.map_turnover = production + (self.map_turnover - production) * self.decay
        self.hr = hr_new

    def step(self, inputs: Mapping[str, float]) -> Mapping[str, float]:
        if self.pre_dose is not None and self.pre_dose.update(inputs):
            self._set_baseline(self.pre_dose.map0, self.pre_dose.hr0)
        y = self._advance(inputs)
        if self.bis_filter is not None:
            # y["bis"] was predicted with the factor so far; the reading then updates it
            bis = inputs["bis"]
            lo, hi = BIS_FRESH_RANGE
            fresh = inputs["bis__staleness"] < self.step_s and lo <= bis <= hi
            x = self.x
            ce = (float(x[3]), float(x[7]), float(x[11]), float(x[12]))
            self.bis_filter.step(bis if fresh else None, lambda theta: self._bis(*ce, scale=math.exp(theta)))
            self.bis_scale = self.bis_filter.scale
            y = dict(y) | {"bis_sensitivity": self.bis_scale}
        if self.map_filter is not None:
            map_ = inputs["map"]
            lo, hi = MAP_FRESH_RANGE
            fresh = inputs["map__staleness"] < self.step_s and lo <= map_ <= hi
            ce_p, ce_r, cc_ne = float(self.x[3]), float(self.x[7]), float(self.x[9]) / self.ne["v1"]
            self.map_filter.step(map_ if fresh else None, lambda theta: self._map_unified(ce_p, ce_r, cc_ne, math.exp(theta)))
            self.map_scale = self.map_filter.scale
            y = dict(y) | {"map_sensitivity": self.map_scale}
        return y

    def _map_unified(self, ce_p: float, ce_r: float, cc_ne: float, scale: float | None = None) -> float:
        # mapUnified.m / mapUnifiedParameters.m: the propofol/remifentanil values
        # are AnestheSIM's illustrative ones (no published set); NE on exogenous Cc.
        # `scale`: the patient's propofol Emax factor
        scale = self.map_scale if scale is None else scale
        ce_p, ce_r = max(ce_p, 0.0), max(ce_r, 0.0)
        c50 = max(2.0 * (1 - 0.5 * ce_r / (ce_r + 2.0)), 1e-6)  # C50P, kR, C50R
        return self.map0 - 30.0 * scale * ce_p / (ce_p + c50) + self._ne_emax(max(cc_ne, 0.0))  # EmaxP

    lookahead_signals = ("propofol_ce", "remifentanil_ce", "bis", "map", "hr")

    def lookahead_columns(self, horizons_s: tuple[int, ...]) -> list[str]:
        """`<signal>@<h>` holding everything, `<signal>@<h>.<drug>_x<factor>` per what-if."""
        names = []
        for suffix in [""] + [f".{d}_x{f:g}" for d in TCI_DRUGS for f in self.whatif]:
            names += [f"{s}@{h}{suffix}" for h in sorted(horizons_s) for s in self.lookahead_signals]
        return names

    def lookahead(self, held: Mapping[str, float], horizons_s: tuple[int, ...]) -> dict[str, float]:
        """Signals after holding `held` (one step's inputs) for each horizon, and per
        what-if with that drug steered to its scaled target instead; starting from
        the current state, which is left as it was. What-ifs for a drug whose pump
        isn't running are NaN."""
        out = self.scenario(held, horizons_s)
        for drug, spec in TCI_DRUGS.items():
            for f in self.whatif:
                target = held[spec["setting"]]
                suffix = f".{drug}_x{f:g}"
                if target == target and target > 0:
                    ahead = self.scenario(held, horizons_s, drug, target * f)
                    out |= {k + suffix: v for k, v in ahead.items()}
                else:
                    out |= {f"{s}@{h}{suffix}": np.nan for h in horizons_s for s in self.lookahead_signals}
        return out

    def scenario(
        self,
        held: Mapping[str, float],
        horizons_s: tuple[int, ...],
        drug: str | None = None,
        target: float | None = None,
    ) -> dict[str, float]:
        """`<signal>@<h>` after holding `held` for each horizon, with `drug` (if
        given) steered to TCI `target` instead; the state is left as it was."""
        state = (self.x.copy(), self.u.copy(), deque(self.ne_queue), self.map_turnover, self.hr, self.t)
        inputs = dict(held)
        out, done = {}, 0
        for h in sorted(horizons_s):
            for _ in range(round(h / self.step_s) - done):
                if drug is not None:
                    inputs[TCI_DRUGS[drug]["dose"]] = self._tci_rate(drug, target) * self.step_s
                y = self._advance(inputs)
            done = round(h / self.step_s)
            out |= {f"{s}@{h}": y[s] for s in self.lookahead_signals}
        self.x, self.u, self.ne_queue, self.map_turnover, self.hr, self.t = state
        return out

    def _tci_rate(self, drug: str, target: float) -> float:
        """The infusion rate a TCI pump would run now: the constant rate that brings
        the model's effect-site concentration to `target` in TCI_LEAD_S, within the
        pump's range (so it boluses at its maximum when far below, and pauses when
        above)."""
        free, gain = self.tci[drug]
        rate = (target - free @ self.x) / gain
        return min(max(rate, 0.0), TCI_DRUGS[drug]["max_rate"])

    def _advance(self, inputs: Mapping[str, float]) -> Mapping[str, float]:
        dt = self.step_s
        u = self.u
        u[0] = inputs["propofol_dose"] / dt  # mg/s
        u[1] = inputs["remifentanil_dose"] / dt  # ug/s
        ne_rate = inputs["norepinephrine_dose"] / dt * 1e3 / NE_MOLAR_MASS  # nmol/s
        if self.ne_queue:
            self.ne_queue.append(ne_rate)
            ne_rate = self.ne_queue.popleft()
        u[2] = ne_rate
        sevo, des = inputs["sevoflurane_et"], inputs["desflurane_et"]
        u[3] = sevo if sevo == sevo else 0.0  # NaN: no volatile
        u[4] = des if des == des else 0.0

        x0 = self.x
        x = self.x = self.ad @ x0 + self.bd @ u
        cp_p, ce_p, cp_r, ce_r = float(x[0]), float(x[3]), float(x[4]), float(x[7])
        cc_ne = float(x[9]) / self.ne["v1"]
        ce_sevo, ce_des = float(x[11]), float(x[12])

        self._turnover((float(x0[0]) + cp_p) / 2, (float(x0[4]) + cp_r) / 2)
        self.t += dt

        if self.map_model == "separable":
            # norepinephrineEffect.m: pressor effect above the endogenous level
            cendo = self.ne["cendo"]
            map_ = self.map_turnover + self._ne_emax(max(cc_ne, 0.0) + cendo) - self._ne_emax(cendo)
        else:
            map_ = self._map_unified(ce_p, ce_r, cc_ne)

        return {
            "propofol_cp": cp_p,
            "propofol_ce": ce_p,
            "remifentanil_cp": cp_r,
            "remifentanil_ce": ce_r,
            "norepinephrine_cp": cc_ne,
            "sevoflurane_ce": ce_sevo,
            "desflurane_ce": ce_des,
            "volatile_mac": max(ce_sevo, 0.0) / self.mac["sevoflurane"]
            + max(ce_des, 0.0) / self.mac["desflurane"],
            "bis": self._bis(ce_p, ce_r, ce_sevo, ce_des),
            "map": map_,
            "hr": self.hr,
        }
