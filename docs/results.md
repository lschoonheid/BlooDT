# Results

What the pipeline achieves on the TIVA arterial-line cohort (`cohorts/tiva-artline.yaml`): 1,585 cases of 1,543 adult patients under propofol and remifentanil TCI with an arterial line, BIS and at least an hour of recording. Patients are split 80/20 into training and test sets, so no test patient was seen in training. Decision points are every 30 s during anaesthesia: 580,938 training rows.

Every model is compared with a baseline that needs no learning. The "sensor faults" columns score the same test patients seen through simulated faults: up to 10 s extra delay per channel and 4 dropouts per hour of 30 s on average.

Numbers below are from run `fourth` (`configs/default.yaml` as of 2026-11-20, `models/fourth/metrics.json`).

## Forecasts

RMSE (mmHg for MAP, bpm for HR) against persistence (the current value stays the same):

| Target | Horizon | Model | Persistence | Model, sensor faults | Persistence, sensor faults |
|---|---|---|---|---|---|
| MAP | 1 min | 9.46 | 12.41 | 9.55 | 12.57 |
| MAP | 3 min | 11.22 | 14.92 | 11.26 | 14.95 |
| MAP | 5 min | 12.09 | 16.19 | 12.12 | 16.20 |
| MAP | 10 min | 12.96 | 17.65 | 12.97 | 17.63 |
| BIS | 1 min | 5.30 | 6.35 | 5.42 | 6.52 |
| BIS | 3 min | 6.61 | 8.53 | 6.65 | 8.59 |
| BIS | 5 min | 7.33 | 9.80 | 7.34 | 9.84 |
| BIS | 10 min | 8.15 | 11.60 | 8.15 | 11.61 |
| HR | 1 min | 6.03 | 7.49 | 6.09 | 7.65 |
| HR | 3 min | 6.85 | 8.63 | 6.88 | 8.73 |
| HR | 5 min | 7.29 | 9.16 | 7.30 | 9.20 |
| HR | 10 min | 7.98 | 9.97 | 7.99 | 10.03 |

Sensor faults cost at most 0.12 in RMSE, mostly at 1 min where a delayed reading matters most.

## Hypotension warnings

A hypotension event is MAP below 65 mmHg for at least a minute; the target is whether one is under way at any time within the horizon. Over all decision points the forecaster reaches AUROC 0.990 / 0.968 / 0.949 / 0.913 at 1 / 3 / 5 / 10 min, but most positives there are already hypotensive, which "MAP already low" (score = −current MAP) also catches (0.968 / 0.928 / 0.898 / 0.845). The useful question is the onset: decision points where MAP is not already below 65.

| Horizon | Onsets followed by hypotension | AUROC | AUROC "MAP already low" | AUPRC | AUPRC "MAP already low" | Sensitivity at 10% alert rate |
|---|---|---|---|---|---|---|
| 1 min | 1.5% | 0.951 | 0.809 | 0.273 | 0.148 | 0.85 |
| 3 min | 4.0% | 0.912 | 0.777 | 0.338 | 0.216 | 0.65 |
| 5 min | 6.2% | 0.888 | 0.754 | 0.368 | 0.254 | 0.54 |
| 10 min | 10.9% | 0.851 | 0.716 | 0.419 | 0.302 | 0.41 |

Onset AUROC with sensor faults is within 0.002 of these. The probabilities are calibrated: expected calibration error 0.003 to 0.008 over ten equal-count bins (reliability tables in `metrics.json`).

## Intervention suggestions

Per drug, a calibrated classifier gives the probability that the clinician lowers, keeps or raises the TCI target within the next minute, and a regressor per direction gives the new setting a decrease or an increase would go to. These imitate observed practice. The new-setting error is for the direction the clinician actually took.

| Drug | Decrease / hold / increase in the data | Log loss (class prior) | AUROC decrease / increase | ECE decrease / increase | New-setting MAE (keep current) |
|---|---|---|---|---|---|
| Propofol (µg/mL) | 2.9% / 95.1% / 2.0% | 0.189 (0.228) | 0.797 / 0.840 | 0.0016 / 0.0007 | 0.46 (1.06) |
| Remifentanil (ng/mL) | 3.3% / 93.6% / 3.1% | 0.236 (0.282) | 0.810 / 0.810 | 0.0015 / 0.0007 | 0.52 (1.24) |

Changes are rare (about 1 in 20 decision points), so even well-ranked probabilities stay low: in the top tenth of propofol-increase predictions about 1 in 10 is followed by an increase within a minute.

### Example: a test patient's case through the twin

`uv run dte replay 17 --run fourth` follows case 17 (vascular surgery, about 5.7 h of anaesthesia; the patient is in the test set) the way the live twin would. Its 5-min MAP forecasts over the case have an RMSE of 9.6 mmHg. At four of the moments a clinician changed the propofol target:

| Time | Target (µg/mL) | BIS | MAP | Clinician did | p(decrease) / p(increase) | Twin's setting if lowered / raised | Expected effect in 5 min if lowered | if raised |
|---|---|---|---|---|---|---|---|---|
| 6.5 min | 4.0 | 49 | 106 | lowered to 3.0 | 0.17 / 0.01 | 3.05 / 5.05 | BIS +4.0, MAP +1.6 | BIS −2.9, MAP −1.2 |
| 14 min | 2.5 | 64 | 88 | raised to 3.5 | 0.00 / 0.15 | 1.94 / 3.14 | BIS +5.3, MAP +1.9 | BIS −4.8, MAP −1.6 |
| 28 min | 3.5 | 24 | 103 | lowered to 3.0 | 0.05 / 0.01 | 2.90 / 3.93 | BIS +3.9, MAP +1.3 | BIS −2.3, MAP −0.8 |
| 3.4 h | 2.8 | 25 | 68 | lowered to 2.6 | 0.04 / 0.00 | 1.94 / 3.00 | BIS +7.1, MAP +2.6 | BIS −1.3, MAP −0.5 |

The probability of the change the clinician made is above its base rate (about 0.03 for a decrease and 0.02 for an increase within a minute) at each of these moments: 6 to 7 times at the first two, only slightly at the last two. The proposed setting for the direction taken is close to what the clinician chose. The expected effects come from AnestheSIM steering its own model to each setting like a TCI pump, compared with keeping the current target.

## What each input group adds

`dte ablate` retrains the 5-min forecasters without one group of inputs (run `ablation2`, clean test patients; the variants without experts or covariates differ from "all" by about as much as reruns would):

| Variant | Features | MAP RMSE | BIS RMSE | Hypotension AUROC | Onset AUROC |
|---|---|---|---|---|---|
| All | 733 | 12.125 | 7.371 | 0.948 | 0.886 |
| No experts | 353 | 12.133 | 7.391 | 0.949 | 0.887 |
| Only AnestheSIM (unified MAP) | 462 | 12.137 | 7.365 | 0.948 | 0.885 |
| Only PAS Marsh | 407 | 12.137 | 7.363 | 0.948 | 0.885 |
| No staleness columns | 715 | 12.128 | 7.364 | 0.947 | 0.884 |
| No patient covariates | 727 | 12.127 | 7.372 | 0.948 | 0.885 |
| No phase (time in anaesthesia, surgery) | 731 | 12.148 | 7.384 | 0.948 | 0.885 |

The forecasts come from the recent course of the vitals and drug settings. The expert models improve BIS by about 0.02 and nothing else measurably, with or without pre-dose baselines and lookaheads (run `ablation` before them showed the same). Their value to the twin is elsewhere: effect-site concentrations no monitor shows, and what-ifs for a target change.

## Expert fidelity

`dte verify` on all 1,585 cases (median over cases; MDPE/MDAPE in %, RMSE in the signal's unit). Concentrations are compared with the pump's own TCI model, BIS and MAP with the monitor.

| Signal | Expert | MDPE | MDAPE | RMSE |
|---|---|---|---|---|
| Propofol Ce | PAS Marsh | −1.0 | 2.6 | 0.10 µg/mL |
| Propofol Ce | PAS Schnider | 0.9 | 10.5 | 0.43 µg/mL |
| Propofol Ce | PAS Eleveld | −18.7 | 19.2 | 0.80 µg/mL |
| Propofol Ce | AnestheSIM (Schnider) | 1.3 | 11.1 | 0.67 µg/mL |
| Remifentanil Ce | Minto (PAS, AnestheSIM) | −0.2 | 0.6 | 0.03 ng/mL |
| BIS | AnestheSIM Greco | −8.5 | 16.9 | 12.4 |
| BIS | PAS Eleveld | 13.3 | 21.1 | 12.0 |
| BIS | PAS Bouillon (Marsh) | −22.4 | 23.7 | 15.4 |
| MAP | AnestheSIM unified | 6.4 | 15.7 | 18.7 mmHg |
| MAP | AnestheSIM separable | 44.2 | 44.4 | 29.3 mmHg |
| MAP | PAS (any) | 49 to 53 | 49 to 53 | 31 to 32 mmHg |

PAS Marsh with Minto is what the Orchestra pumps run, so its near-exact concentrations confirm the dose reconstruction. Every MAP model except AnestheSIM's unified one predicts far too strong a propofol effect: measured MAP is about 1.5 times theirs.

## Causality

The twin (`live/twin.py`) takes raw records one at a time and keeps its own grid. Replaying a case through it reproduces the offline frame exactly (`tests/test_live.py`, a case with a pump-volume jump), so no offline feature uses a value from after its decision point.

## Known limits

- Boluses are not in VitalDB's tracks (only per-case totals), so a vasopressor bolus is invisible: it shows up only as MAP rising.
- The suggesters learn when clinicians at one hospital changed settings, not when they should have.
- Single readings 5 min ahead are noisy; part of the remaining MAP error is beat-to-beat and artefact variation no model can forecast.
- The what-if effects use AnestheSIM's PK/PD with its own TCI; they are directionally plausible but not validated against observed responses to target changes.
