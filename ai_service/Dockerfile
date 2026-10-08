# ai_service - análise de logs com IA (Isolation Forest + NLP) e exportação de métricas Prometheus
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    LOG_FILE=/data/app.log \
    MODEL_FILE=/model/anomaly_model.pkl \
    OUTPUT_DIR=/output

WORKDIR /app

# fuso horário (TZ vem do docker-compose) para os logs ao vivo saírem no horário local
RUN apt-get update && apt-get install -y --no-install-recommends tzdata && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código. Logs (/data), modelo (/model) e relatórios (/output) chegam por volume, não pela imagem:
# no material original o COPY ia para /app/model enquanto o script lia /model.
COPY gerar_logs.py log_analyzer.py ./

EXPOSE 8000
CMD ["python", "log_analyzer.py", "--serve"]
