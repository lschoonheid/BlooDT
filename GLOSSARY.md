# Anaesthesia Digital Twin

A digital shadow of a patient under anaesthesia, built from replayed VitalDB cases: it estimates unmeasurable drug levels, forecasts vitals and suggests infusion changes to a clinician.

## Data

**Patient**:
One person in VitalDB, identified by `subjectid`. A patient can have several cases.
_Avoid_: Subject (in prose), case

**Case**:
One recorded operation of one patient, identified by `caseid`.
_Avoid_: Patient, recording, file

**Track**:
One named time series recorded by one device in a case, such as `Solar8000/ART_MBP`.
_Avoid_: Sensor, signal, channel

**Channel**:
A logical signal, such as mean arterial pressure or cumulative propofol dose, in canonical units, satisfied in a case by the first of its candidate tracks that the case has.
_Avoid_: Sensor, feature, track (when the logical signal is meant)

**Staleness**:
How many seconds ago a channel's current value was actually observed. A value is never invented between observations; it only grows stale.
_Avoid_: Age, lag, delay

**Patient covariates**:
The static facts about a patient at the time of a case: age, sex, height, weight, BMI, ASA class.
_Avoid_: Demographics, metadata

**Cohort**:
A named, reproducible set of cases chosen by clinical criteria and required tracks.
_Avoid_: Dataset, selection, query

## Twin

**Expert model**:
A mechanistic pharmacological or physiological model (e.g. Schnider PK with a Bouillon BIS model) run over a case's drug inputs. Its outputs inform the forecaster; it is not a member of a gated mixture of experts.
_Avoid_: Mixture of experts, MoE, simulator

**Expert signal**:
One time series produced by an expert model for a case, such as propofol effect-site concentration or model-predicted BIS.
_Avoid_: Expert feature, prediction

**Forecaster**:
A learned model that predicts a target at a horizon from tracks, expert signals and patient covariates.
_Avoid_: Predictor, ML model

**Decision point**:
A moment in a case at which the twin makes its forecasts and suggestions, using only what was observed up to then.
_Avoid_: Sample, timestep, row

**Target**:
A quantity a forecaster predicts, either a future value of a vital or the occurrence of an event.

**Horizon**:
How far ahead of the current moment a target is predicted.
_Avoid_: Lookahead, lead time (lead time is the evaluation metric for warnings)

**Hypotension event**:
Mean arterial pressure below 65 mmHg sustained for at least one minute.
_Avoid_: Low BP, IOH (in code)

## Interventions

**Target concentration**:
The concentration a clinician sets on a TCI pump; the pump then chooses the infusion rate itself to reach it.
_Avoid_: Setpoint, CT (in prose)

**Intervention**:
A deliberate, sustained change of a pump setting observed in a case: the target concentration of a TCI pump, or the rate of a rate-mode pump. Rate changes a TCI pump makes on its own are not interventions.
_Avoid_: Action, dose event, bolus, rate change

**Bolus**:
A one-off injected drug dose. VitalDB records only per-case totals, so boluses are not interventions in this project.

**Intervention suggestion**:
The twin's estimate of the infusion change a clinician would typically make now, with its probability. It imitates observed practice; it does not claim the change is optimal.
_Avoid_: Recommendation, prescription, advice
