"""The live layer: Replayer -> MQTT -> Twin -> InfluxDB -> Grafana (services outlined).

Services talk only through MQTT and InfluxDB (see docker-compose.yml). The twin
reuses the offline pipeline's pieces unchanged: channels' causal grid, experts'
`step`, `features.window_features` on a rolling buffer, and the trained
forecasters and suggesters from models/<run>/models.joblib.
"""
