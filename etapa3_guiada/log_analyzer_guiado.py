import re
from collections import defaultdict
from datetime import datetime
import numpy as np
from sklearn.ensemble import IsolationForest
import pandas as pd

# Etapa 3 - código guiado do enunciado, transcrito sem alterações de lógica.
# Única mudança: LOG_FILE aponta para o app.log deste repositório (no container: /data/app.log).

# Caminho do arquivo de log

# Para ler o LOG-FILE no container
LOG_FILE = '/data/app.log'

# Para ler o LOG-FILE no ambiente local, aqui no caso o Windows
#LOG_FILE = 'D:/DockerCompose/Observabilidade/ai_service/data/app.log'

# Função para pré-processar logs e extrair features
def preprocess_logs():
    data = []
    log_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\w+) (.+)'

    with open(LOG_FILE, 'r') as file:
        for line in file:
            match = re.match(log_pattern, line)
            if match:
                timestamp_str, level, message = match.groups()
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

                # Extração de features básicas
                hour = timestamp.hour
                level_num = {"INFO": 0, "DEBUG": 1, "WARN": 2, "ERROR": 3}.get(level, -1)
                message_length = len(message)
                slow_query = int('Query executed in' in message and int(message.split()[-1][:-2]) > 100)

                # Adiciona as features à lista
                data.append([hour, level_num, message_length, slow_query])

    # Converte a lista para um DataFrame
    df = pd.DataFrame(data, columns=['hour', 'level_num', 'message_length', 'slow_query'])
    return df

# Função para detectar anomalias usando Isolation Forest
def detect_anomalies(df):
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(df)
    anomalies = model.predict(df)

    # Seleciona eventos anômalos
    anomaly_indices = np.where(anomalies == -1)[0]
    anomaly_logs = df.iloc[anomaly_indices]
    return anomaly_logs

# Função principal de análise de logs
def analyze_logs():
    df = preprocess_logs()
    anomaly_logs = detect_anomalies(df)

    print("\n=== Detected Anomalies ===")
    if not anomaly_logs.empty:
        for index, row in anomaly_logs.iterrows():
            print(f"Anomaly at hour {row['hour']}: Level {row['level_num']}, Message Length {row['message_length']}, Slow Query {row['slow_query']}")
    else:
        print("No anomalies detected.")

    # Exibe um resumo dos tipos de logs
    print("\n=== Log Summary ===")
    print(df.describe())

if __name__ == '__main__':
    analyze_logs()
