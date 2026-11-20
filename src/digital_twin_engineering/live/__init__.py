"""The live layer: Replayer -> MQTT -> twin service -> InfluxDB -> Grafana.

Services talk only through MQTT and InfluxDB (see docker-compose.yml): the
replayer publishes a case's raw records (messages.py), the twin service runs a
Twin per case and writes its outputs (service.py, influx.py). The twin reuses the
offline pipeline's pieces unchanged: channels' causal grid, experts' `step`,
`features.window_features` on a rolling buffer, and the trained forecasters and
suggesters from models/<run>/models.joblib.
"""
