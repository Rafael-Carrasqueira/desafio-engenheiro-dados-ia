"""Gera grafana/dashboards/ai_service.json (uso: python tools/gerar_dashboard.py grafana/dashboards/ai_service.json)."""
import json
import sys

PROM = {"type": "prometheus", "uid": "prometheus"}
PG = {"type": "grafana-postgresql-datasource", "uid": "postgres"}
_id = 0


def nid():
    global _id
    _id += 1
    return _id


def stat(title, expr, x, y, w=6, h=4, unit="short", color="blue", thresholds=None, desc=""):
    steps = thresholds or [{"color": color, "value": None}]
    return {
        "id": nid(), "type": "stat", "title": title, "description": desc, "datasource": PROM,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [{"refId": "A", "datasource": PROM, "expr": expr, "instant": True}],
        "fieldConfig": {"defaults": {"unit": unit, "thresholds": {"mode": "absolute", "steps": steps},
                                     "color": {"mode": "thresholds"}}, "overrides": []},
        "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                    "colorMode": "background", "graphMode": "none", "textMode": "value", "justifyMode": "center"},
    }


def ts(title, targets, x, y, w=12, h=8, unit="short", stack=False, desc="", overrides=None, draw="line"):
    return {
        "id": nid(), "type": "timeseries", "title": title, "description": desc, "datasource": PROM,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [{"refId": chr(65 + i), "datasource": PROM, "expr": e, "legendFormat": l, "range": True}
                    for i, (e, l) in enumerate(targets)],
        "fieldConfig": {
            "defaults": {"unit": unit, "color": {"mode": "palette-classic"},
                         "custom": {"drawStyle": draw, "lineWidth": 2, "fillOpacity": 25 if stack else 8,
                                    "stacking": {"mode": "normal" if stack else "none", "group": "A"},
                                    "showPoints": "never", "spanNulls": True}},
            "overrides": overrides or []},
        "options": {"legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                    "tooltip": {"mode": "multi", "sort": "desc"}},
    }


def fixed(name, color, dash=False):
    props = [{"id": "color", "value": {"mode": "fixed", "fixedColor": color}}]
    if dash:
        props.append({"id": "custom.lineStyle", "value": {"fill": "dash", "dash": [10, 10]}})
        props.append({"id": "custom.fillOpacity", "value": 0})
    return {"matcher": {"id": "byName", "options": name}, "properties": props}


panels = [
    stat("Incidentes confirmados", "sum(ai_log_incidents_total) or vector(0)", 0, 0, color="purple",
         desc="Incidentes encerrados e explicados pelo ai_service desde a subida"),
    stat("Incidente em andamento", "max(ai_log_incident_open) or vector(0)", 6, 0,
         thresholds=[{"color": "green", "value": None}, {"color": "red", "value": 1}],
         desc="1 enquanto janelas consecutivas seguem anômalas"),
    stat("Linhas de log / min", "sum(rate(ai_log_lines_total[2m])) * 60", 12, 0, unit="short", color="blue"),
    stat("Score da última janela", "max(ai_log_window_anomaly_score)", 18, 0, unit="none",
         thresholds=[{"color": "green", "value": None}, {"color": "orange", "value": 0.55},
                     {"color": "red", "value": 0.62}],
         desc="Isolation Forest por janela de 1 min (maior = mais anômalo)"),
    ts("Score de anomalia (Isolation Forest por janela) x limiar",
       [("max(ai_log_window_anomaly_score)", "score"), ("max(ai_log_window_anomaly_threshold)", "limiar do modelo")],
       0, 4, w=16, unit="none",
       overrides=[fixed("score", "orange"), fixed("limiar do modelo", "red", dash=True)],
       desc="Acima do limiar = minuto anômalo; minutos seguidos e explicados viram incidente"),
    {
        "id": nid(), "type": "bargauge", "title": "Incidentes por causa", "datasource": PROM,
        "gridPos": {"x": 16, "y": 4, "w": 8, "h": 8},
        "targets": [{"refId": "A", "datasource": PROM, "expr": "sum by (tipo) (ai_log_incidents_total)",
                     "legendFormat": "{{tipo}}", "instant": True}],
        "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}, "min": 0, "decimals": 0,
                                     "displayName": "${__field.labels.tipo}"}, "overrides": []},
        "options": {"orientation": "horizontal", "displayMode": "gradient", "showUnfilled": True,
                    "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}},
    },
    ts("Linhas por nível (por minuto)", [("sum by (level) (rate(ai_log_lines_total[2m])) * 60", "{{level}}")],
       0, 12, stack=True,
       overrides=[fixed("ERROR", "red"), fixed("WARN", "orange"), fixed("INFO", "green"), fixed("DEBUG", "blue")]),
    ts("Categorias de evento (NLP: TF-IDF + K-Means) por minuto",
       [("sum by (category) (rate(ai_log_lines_by_category_total[2m])) * 60", "{{category}}")], 12, 12, stack=True),
    ts("Falhas de login por minuto", [("sum by (origem) (rate(ai_log_failed_logins_total[2m])) * 60", "{{origem}}")],
       0, 20, w=8, overrides=[fixed("externa", "red"), fixed("interna", "blue")]),
    ts("Latência extraída dos logs (p50 / p95)",
       [("histogram_quantile(0.50, sum by (le) (rate(ai_log_latency_ms_bucket[2m])))", "p50"),
        ("histogram_quantile(0.95, sum by (le) (rate(ai_log_latency_ms_bucket[2m])))", "p95")],
       8, 20, w=8, unit="ms"),
    ts("Eventos individuais anômalos / min (Isolation Forest por evento)",
       [("sum(rate(ai_log_event_anomalies_total[2m])) * 60", "eventos anômalos"),
        ("sum(rate(ai_log_isolated_alerts_total[10m])) * 600", "alertas isolados descartados (/10 min)")],
       16, 20, w=8),
    {
        "id": nid(), "type": "table", "title": "Incidentes registrados no Postgres (tabela ai_incidentes)",
        "datasource": PG, "gridPos": {"x": 0, "y": 28, "w": 24, "h": 9},
        "targets": [{"refId": "A", "datasource": PG, "format": "table", "rawQuery": True, "editorMode": "code",
                     "rawSql": "SELECT origem, to_char(inicio, 'YYYY-MM-DD HH24:MI') AS inicio, to_char(fim, 'HH24:MI') AS fim, "
                               "tipo, hipotese, round(score::numeric, 3) AS score, linhas, erros "
                               "FROM ai_incidentes ORDER BY registrado_em DESC, ai_incidentes.inicio DESC LIMIT 25"}],
        "fieldConfig": {"defaults": {"custom": {"align": "auto"}}, "overrides": [
            {"matcher": {"id": "byName", "options": "hipotese"}, "properties": [{"id": "custom.width", "value": 430}]}]},
        "options": {"showHeader": True, "cellHeight": "sm"},
    },
]

dash = {
    "uid": "ai-logs", "title": "ai_service - Análise de Logs com IA", "tags": ["ia", "logs", "observabilidade"],
    "timezone": "browser", "schemaVersion": 41, "version": 1, "editable": False, "graphTooltip": 1,
    "refresh": "10s", "time": {"from": "now-30m", "to": "now"},
    "panels": panels,
    "annotations": {"list": [{
        "builtIn": 0, "datasource": PROM, "enable": True, "iconColor": "red", "name": "Incidentes",
        "expr": "changes(ai_log_incidents_total[1m]) > 0", "titleFormat": "Incidente: {{tipo}}", "step": "30s"}]},
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(dash, f, indent=2, ensure_ascii=False)
print("panels:", len(panels))
