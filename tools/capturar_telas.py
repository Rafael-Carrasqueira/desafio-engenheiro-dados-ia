"""
Captura as telas da stack de observabilidade (Grafana, Prometheus, /metrics) para a pasta evidencias/.

Roda dentro do container oficial do Playwright, na mesma rede do docker-compose:

    docker run --rm --network observabilidade-ia_backend -v ${PWD}/evidencias:/out -v ${PWD}/tools:/tools \
        mcr.microsoft.com/playwright/python:v1.55.0-noble \
        sh -c "pip install -q playwright==1.55.0 && python /tools/capturar_telas.py"
"""
import time

from playwright.sync_api import sync_playwright

GRAFANA = "http://grafana:3000"
PROM = "http://prometheus:9090"
AI = "http://ai_service:8000"

with sync_playwright() as p:
    b = p.chromium.launch()

    pg = b.new_page(viewport={"width": 1600, "height": 1500}, device_scale_factor=1, locale="pt-BR", timezone_id="America/Sao_Paulo")
    pg.goto(f"{GRAFANA}/d/ai-logs/?orgId=1&kiosk&from=now-35m&to=now&refresh=", wait_until="load")
    time.sleep(15)                                  # painéis terminam de consultar
    pg.screenshot(path="/out/05_grafana_dashboard.png", full_page=True)
    print("grafana ok")

    pg = b.new_page(viewport={"width": 1500, "height": 900}, locale="pt-BR", timezone_id="America/Sao_Paulo")
    pg.goto(f"{PROM}/targets", wait_until="load")
    time.sleep(6)
    pg.screenshot(path="/out/06_prometheus_targets.png", full_page=True)
    print("targets ok")

    pg.goto(f"{PROM}/alerts", wait_until="load")
    time.sleep(6)
    pg.screenshot(path="/out/07_prometheus_alertas.png", full_page=True)
    print("alerts ok")

    q = "sum%20by%20(tipo)%20(ai_log_incidents_total)"
    pg.goto(f"{PROM}/query?g0.expr={q}&g0.tab=table", wait_until="load")
    time.sleep(6)
    pg.screenshot(path="/out/08_prometheus_promql.png", full_page=False)
    print("promql ok")

    pg = b.new_page(viewport={"width": 1300, "height": 1100}, locale="pt-BR", timezone_id="America/Sao_Paulo")
    pg.goto(f"{AI}/metrics", wait_until="load")
    pg.screenshot(path="/out/09_ai_service_metrics.png", full_page=False)
    print("metrics ok")
    b.close()
