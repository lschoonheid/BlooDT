# Results

What the pipeline achieves on the TIVA arterial-line cohort (`cohorts/tiva-artline.yaml`): 1,585 cases of 1,543 adult patients under propofol and remifentanil TCI with an arterial line, BIS and at least an hour of recording. Patients are split 80/20 into training and test sets, so no test patient was seen in training. Decision points are every 30 s during anaesthesia: 580,938 training rows.

Every model is compared with a baseline that needs no learning. The "sensor faults" columns score the same test patients seen through simulated faults: up to 10 s extra delay per channel and 4 dropouts per hour of 30 s on average.

Numbers below are from run `sixth` (`models/sixth/metrics.json`, `configs/default.yaml` at commit 9e7ad72) unless a section names another run. Run `sixth` added the 1-min MAP target and online BIS calibration to run `fourth`; every metric `fourth` also had is within 0.02 of it (0.006 for AUROC and AUPRC). Today's simpler config (run `lean`) gives the same results for what it still forecasts: see "The simplified pipeline".

## The simplified pipeline

Ablation showed most of the pipeline's inputs add nothing measurable, so run `lean` (today's `configs/default.yaml`) keeps what the twin uses:

- 2 expert models instead of 5: PAS Marsh (the pumps' own model, which verifies the doses) and AnestheSIM with online BIS calibration (the what-ifs). PAS Schnider, PAS Eleveld and AnestheSIM's separable MAP model went.
- No expert lookaheads or what-ifs computed at every decision point as features (they added nothing); the twin and `dte effects` simulate a target change when they need its effect.
- 8 forecasters instead of 20: 1-min MAP, BIS, HR and hypotension at 1 and 5 min. The 1-min forecasts serve the deviation flags.
- No staleness columns among the features (a stale value is already missing).

Test metrics of what both runs forecast:

| Target | `sixth` | `lean` | `sixth`, sensor faults | `lean`, sensor faults |
|---|---|---|---|---|
| 1-min MAP RMSE, 1 min | 6.68 | 6.68 | 6.80 | 6.80 |
| 1-min MAP RMSE, 5 min | 10.04 | 10.05 | 10.07 | 10.08 |
| BIS RMSE, 1 min | 5.30 | 5.30 | 5.42 | 5.42 |
| BIS RMSE, 5 min | 7.31 | 7.32 | 7.33 | 7.34 |
| HR RMSE, 1 min | 6.03 | 6.04 | 6.09 | 6.10 |
| HR RMSE, 5 min | 7.28 | 7.28 | 7.30 | 7.30 |
| Hypotension onset AUROC, 1 min | 0.952 | 0.951 | 0.952 | 0.951 |
| Hypotension onset AUROC, 5 min | 0.888 | 0.888 | 0.888 | 0.888 |
| Propofol suggester log loss | 0.189 | 0.189 | 0.190 | 0.190 |
| Remifentanil suggester log loss | 0.236 | 0.236 | 0.237 | 0.237 |

Every metric is within 0.01 of `sixth`, about what a rerun changes. The alert threshold (10% alert rate) is 0.55, catching 59% of hypotension within 5 min as before; the what-if validation and expert fidelity are unchanged (the same AnestheSIM and PAS Marsh). Rows have 512 columns instead of 836; from cached channels the dataset builds in 18 min instead of 21 (including recomputing PAS Marsh), training takes 27 min instead of 75, and the twin service runs a case 1.8 times faster.

`dte ablate` and `dte transfer` produced the two sections below that name them; they were removed with the simplification (commit 9e7ad72 has them).

## Forecasts

RMSE (mmHg for MAP, bpm for HR) against persistence (the current value stays the same; for 1-min MAP, the mean over the last minute). "MAP" is the reading at the horizon, "1-min MAP" the median over the minute around it: single readings carry beat-to-beat and artefact swings no model can forecast, which the hypotension definition leaves out too.

| Target | Horizon | Model | Persistence | Model, sensor faults | Persistence, sensor faults |
|---|---|---|---|---|---|
| MAP | 1 min | 9.46 | 12.41 | 9.55 | 12.57 |
| MAP | 3 min | 11.22 | 14.92 | 11.26 | 14.95 |
| MAP | 5 min | 12.09 | 16.19 | 12.12 | 16.20 |
| MAP | 10 min | 12.96 | 17.65 | 12.97 | 17.63 |
| 1-min MAP | 1 min | 6.68 | 9.65 | 6.80 | 9.85 |
| 1-min MAP | 3 min | 9.00 | 12.26 | 9.04 | 12.36 |
| 1-min MAP | 5 min | 10.04 | 13.63 | 10.07 | 13.70 |
| 1-min MAP | 10 min | 11.16 | 15.34 | 11.17 | 15.38 |
| BIS | 1 min | 5.30 | 6.35 | 5.42 | 6.52 |
| BIS | 3 min | 6.61 | 8.53 | 6.65 | 8.59 |
| BIS | 5 min | 7.31 | 9.80 | 7.33 | 9.84 |
| BIS | 10 min | 8.14 | 11.60 | 8.14 | 11.61 |
| HR | 1 min | 6.03 | 7.49 | 6.09 | 7.65 |
| HR | 3 min | 6.85 | 8.63 | 6.88 | 8.73 |
| HR | 5 min | 7.28 | 9.16 | 7.29 | 9.20 |
| HR | 10 min | 7.97 | 9.97 | 7.97 | 10.03 |

Sensor faults cost at most 0.12 in RMSE, mostly at 1 min where a delayed reading matters most.

## Hypotension warnings

A hypotension event is MAP below 65 mmHg for at least a minute; the target is whether one is under way at any time within the horizon. Over all decision points the forecaster reaches AUROC 0.990 / 0.968 / 0.949 / 0.913 at 1 / 3 / 5 / 10 min, but most positives there are already hypotensive, which "MAP already low" (score = −current MAP) also catches (0.968 / 0.928 / 0.898 / 0.845). The useful question is the onset: decision points where MAP is not already below 65.

| Horizon | Onsets followed by hypotension | AUROC | AUROC "MAP already low" | AUPRC | AUPRC "MAP already low" | Sensitivity at 10% alert rate |
|---|---|---|---|---|---|---|
| 1 min | 1.5% | 0.952 | 0.809 | 0.272 | 0.148 | 0.85 |
| 3 min | 4.0% | 0.912 | 0.777 | 0.341 | 0.216 | 0.65 |
| 5 min | 6.2% | 0.888 | 0.754 | 0.374 | 0.254 | 0.54 |
| 10 min | 10.9% | 0.850 | 0.716 | 0.419 | 0.302 | 0.41 |

Onset AUROC with sensor faults is within 0.002 of these. The probabilities are calibrated: expected calibration error 0.003 to 0.008 over ten equal-count bins (reliability tables in `metrics.json`). The twin alerts when the 5-min probability reaches 0.54, which alerted at 10% of test decision points and caught 59% of the hypotension within 5 min.

## Intervention suggestions

Per drug, a calibrated classifier gives the probability that the clinician lowers, keeps or raises the TCI target within the next minute, and a regressor per direction gives the new setting a decrease or an increase would go to. These imitate observed practice. The new-setting error is for the direction the clinician actually took.

| Drug | Decrease / hold / increase in the data | Log loss (class prior) | AUROC decrease / increase | ECE decrease / increase | New-setting MAE (keep current) |
|---|---|---|---|---|---|
| Propofol (µg/mL) | 2.9% / 95.1% / 2.0% | 0.189 (0.228) | 0.797 / 0.841 | 0.0016 / 0.0008 | 0.46 (1.06) |
| Remifentanil (ng/mL) | 3.3% / 93.6% / 3.1% | 0.236 (0.282) | 0.809 / 0.810 | 0.0019 / 0.0008 | 0.52 (1.24) |

Changes are rare (about 1 in 20 decision points), so even well-ranked probabilities stay low: in the top tenth of propofol-increase predictions about 1 in 10 is followed by an increase within a minute.

### Example: a test patient's case through the twin

`uv run dte replay 17 --run sixth` follows case 17 (vascular surgery, about 5.7 h of anaesthesia; the patient is in the test set) the way the live twin would. Its 5-min forecasts over the case have an RMSE of 9.5 mmHg for MAP and 7.5 for 1-min MAP. At four of the moments a clinician changed the propofol target:

| Time | Target (µg/mL) | BIS | MAP | Clinician did | p(decrease) / p(increase) | Twin's setting if lowered / raised | Expected effect in 5 min if lowered | if raised |
|---|---|---|---|---|---|---|---|---|
| 6.5 min | 4.0 | 49 | 106 | lowered to 3.0 | 0.16 / 0.01 | 3.07 / 5.12 | BIS +5.4, MAP +1.6 | BIS −5.2, MAP −1.3 |
| 14 min | 2.5 | 64 | 88 | raised to 3.5 | 0.00 / 0.14 | 1.92 / 3.14 | BIS +5.0, MAP +1.9 | BIS −4.6, MAP −1.6 |
| 28 min | 3.5 | 24 | 103 | lowered to 3.0 | 0.06 / 0.01 | 2.99 / 3.96 | BIS +2.3, MAP +1.1 | BIS −1.7, MAP −0.8 |
| 3.4 h | 2.8 | 25 | 68 | lowered to 2.6 | 0.03 / 0.00 | 2.11 / 2.99 | BIS +1.8, MAP +2.0 | BIS −0.4, MAP −0.4 |

The probability of the change the clinician made is above its base rate (about 0.03 for a decrease and 0.02 for an increase within a minute) at each of these moments: 6 to 7 times at the first two, only slightly at the last two. The proposed setting for the direction taken is close to what the clinician chose. The expected effects come from AnestheSIM steering its own model to each setting like a TCI pump, compared with keeping the current target, with the patient's hypnotic sensitivity calibrated online. At the last two moments BIS is near the model's floor, so a change moves it little.

## Do the what-ifs hold?

The expected effects of a target change come from AnestheSIM alone, so `dte effects` checks them against what followed the clinicians' own changes in the test patients (7,014 changes in 314 cases, `models/sixth/effects.parquet`). For each change, AnestheSIM simulates it from the decision point before it: the drug steered to the new target minus kept at the current one. What followed is the forecaster's residual at that point, the measured value at the horizon minus its forecast. The forecaster doesn't know the change is coming, so the residual holds the change's effect plus noise. It may partly anticipate changes from what prompts them, which shrinks residuals towards zero. At points without a change the mean residual is within 0.36 of zero for every signal and horizon.

Mean expected effect and mean residual 5 min after a change, with a 95% interval over cases (bootstrap) for the residual:

| Change | Signal | Expected | Residual |
|---|---|---|---|
| Propofol lowered | BIS | +4.0 | +1.6 (1.1 to 2.1) |
| Propofol raised | BIS | −3.0 | −2.7 (−3.3 to −2.1) |
| Remifentanil lowered | BIS | +2.6 | +0.9 (0.5 to 1.3) |
| Remifentanil raised | BIS | −1.9 | −1.1 (−1.5 to −0.7) |
| Propofol lowered | MAP | +1.5 | +0.7 (0.0 to 1.5) |
| Propofol raised | MAP | −1.2 | −0.7 (−1.6 to 0.1) |
| Remifentanil lowered | MAP | +0.3 | +1.1 (0.5 to 1.8) |
| Remifentanil raised | MAP | −0.2 | −0.2 (−0.9 to 0.6) |
| Remifentanil lowered | HR | +2.4 | +0.1 (−0.3 to 0.4) |
| Remifentanil raised | HR | −1.9 | +0.1 (−0.3 to 0.5) |

- **BIS**: the expected effects have the right sign in all four directions, and the residual is 40 to 90% of the expected effect. Across propofol changes, expected and residual correlate (r = 0.31 at 5 min and 0.36 at 10 min, slope 0.57 and 0.74); remifentanil changes less so (0.18 and 0.15). Online calibration raised these from 0.28 and 0.35.
- **MAP**: a propofol change moves MAP the way expected but by about half as much, and the size of single changes isn't predicted (r = 0.03). AnestheSIM underestimates remifentanil's effect on MAP (lowering it raises MAP by about 1 mmHg, not 0.3).
- **HR**: AnestheSIM expects remifentanil changes to move HR by about 2 bpm; nothing of the kind follows. Its HR what-ifs shouldn't be shown.

So the twin's expected BIS effects can be trusted in direction and roughly in size, its MAP effects in direction only, and its HR effects not at all. Fitting the unified MAP model's propofol effect online as well (`calibrate_map`) brings its MAP error from 15% to 6% MDAPE, ending at about 0.6 of the population effect, in line with the half-size effects above; but its what-ifs agree no better with what follows a change (r = 0.02 to 0.04), so it was removed (commit 9e7ad72 has it).

## What each input group adds

`dte ablate` (removed since; commit 9e7ad72) retrained the 5-min forecasters without one group of inputs (run `ablation2`, clean test patients; the variants without experts or covariates differ from "all" by about as much as reruns would):

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
| BIS | AnestheSIM Greco, calibrated online | −1.5 | 9.0 | 7.5 |
| BIS | AnestheSIM Greco | −8.5 | 16.9 | 12.4 |
| BIS | PAS Eleveld | 13.3 | 21.1 | 12.0 |
| BIS | PAS Bouillon (Marsh) | −22.4 | 23.7 | 15.4 |
| MAP | AnestheSIM unified | 6.4 | 15.7 | 18.7 mmHg |
| MAP | AnestheSIM separable | 44.2 | 44.4 | 29.3 mmHg |
| MAP | PAS (any) | 49 to 53 | 49 to 53 | 31 to 32 mmHg |

Numbers from run `sixth`'s config. PAS Marsh with Minto is what the Orchestra pumps run, so its near-exact concentrations confirm the dose reconstruction. Online calibration (below) halves AnestheSIM's BIS error; He et al. (2023) report an RMSE of 15.6 for a standard drug model. Every MAP model except AnestheSIM's unified one predicts far too strong a propofol effect: measured MAP is about 1.5 times theirs.

## Other patient groups

`dte transfer` (removed since; commit 9e7ad72) trained the 5-min forecasters on one group's training patients and tested them on both groups' test patients (run `transfer`, `configs/default.yaml` as in run `fifth`). The larger group's training data is cut to the size of the smaller one, so the two models learn from equally much. Each cell is the test metric on the "tested on" group; MAP and BIS are RMSE, hypotension is onset AUROC.

| Split | Tested on | MAP, own group's model | MAP, other group's model | BIS, own | BIS, other | Onset AUROC, own | Onset AUROC, other |
|---|---|---|---|---|---|---|---|
| Sex | female | 12.71 | 12.80 | 8.04 | 8.06 | 0.877 | 0.876 |
| Sex | male | 11.87 | 11.91 | 7.06 | 7.11 | 0.879 | 0.882 |
| Age | 65 and over | 11.94 | 12.05 | 7.31 | 7.40 | 0.871 | 0.863 |
| Age | under 65 | 12.39 | 12.41 | 7.49 | 7.61 | 0.888 | 0.882 |
| Department | general and other | 11.63 | 11.96 | 7.12 | 7.20 | 0.882 | 0.880 |
| Department | thoracic | 13.12 | 13.46 | 8.07 | 8.11 | 0.879 | 0.864 |
| Approach | open | 11.66 | 11.93 | 6.96 | 7.15 | 0.866 | 0.861 |
| Approach | videoscopic or robotic | 12.69 | 12.88 | 7.90 | 7.85 | 0.899 | 0.902 |
| ASA | 1-2 | 12.24 | 12.33 | 7.58 | 7.57 | 0.875 | 0.872 |
| ASA | 3+ | 12.57 | 12.70 | 7.36 | 7.43 | 0.846 | 0.835 |

A model from the other group costs at most 0.34 mmHg (between general and thoracic surgery), 0.19 BIS and 0.015 onset AUROC. Within this hospital's TIVA cohort the forecasters carry over between these groups; other hospitals, with other practice and monitors, remain untested.

## Causality

The twin (`live/twin.py`) takes raw records one at a time and keeps its own grid. Replaying a case through it reproduces the offline frame exactly (`tests/test_live.py`, a case with a pump-volume jump), so no offline feature uses a value from after its decision point.

## Live stack

Replayer → Mosquitto → twin service → InfluxDB → Grafana, as in the project plan (README, "Live stack"). The twin service runs one twin per case, so several operating rooms can be replayed at once; with all experts and models it runs a case about 190 times faster than real time on one core. Feeding a case to the service message by message gives the same forecasts as the in-process replay (`tests/test_service.py`, also through an MQTT broker). The service writes, per case:

- the vitals and every expert's state, each second;
- each forecast at the time it is for, so the dashboard draws it over what was later measured;
- the hypotension risk, with an alert at the 5-min probability that alerted at 10% of test decision points;
- the suggestions with their expected effects;
- fidelity so far: the forecasts' RMSE against persistence, and each expert's Varvel metrics against the pump's Ce and the measured BIS and MAP;
- deviations: the measured value minus the forecast made a minute earlier, flagged when it exceeds 3 test RMSEs without a target change in the last 2 minutes. On test patients about 1.5% of decision points are flagged per signal. In the 5 minutes after incision the rate is 5% for HR, but hardly higher than elsewhere for BIS and MAP: the first incision shows mainly in the heart rate.

The InfluxDB and Grafana side has been checked only by its line protocol (in the tests), not yet against a running InfluxDB or Grafana.

## Known limits

- Boluses are not in VitalDB's tracks (only per-case totals), so a vasopressor bolus is invisible. Nor can it be recovered from the vitals: per case, the rate of sudden MAP rises the forecaster didn't expect does not tell cases given ephedrine or phenylephrine from the rest (AUROC 0.38 to 0.43). The rate of unexpected drops does, weakly (0.61): the need for a vasopressor shows, its effect doesn't.
- The suggesters learn when clinicians at one hospital changed settings, not when they should have.
- Single readings 5 min ahead are noisy; part of the remaining MAP error is beat-to-beat and artefact variation no model can forecast.
- The what-if effects use AnestheSIM's PK/PD with its own TCI. Against observed responses (above), BIS effects hold in direction and roughly in size, MAP effects only in direction, HR effects not at all.
