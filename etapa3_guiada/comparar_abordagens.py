"""
Compara a abordagem guiada (Etapa 3) com o log_analyzer.py (Etapa 2) usando o gabarito do gerador.

Reproduz exatamente as features e o modelo do código guiado, mas guarda o timestamp de cada linha
para saber quantas das linhas marcadas como anômalas pertencem a algum incidente real.
"""
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

LOG_FILE = "/data/app.log"
GABARITO = "/data/incidentes.json"
RELATORIO = "/output/relatorio.json"

linhas, ts = [], []
padrao = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\w+) (.+)"
with open(LOG_FILE) as f:
    for line in f:
        m = re.match(padrao, line)
        if m:
            t, level, message = m.groups()
            dt = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
            slow = int("Query executed in" in message and int(message.split()[-1][:-2]) > 100)
            linhas.append([dt.hour, {"INFO": 0, "DEBUG": 1, "WARN": 2, "ERROR": 3}.get(level, -1), len(message), slow])
            ts.append(dt)

df = pd.DataFrame(linhas, columns=["hour", "level_num", "message_length", "slow_query"])
pred = IsolationForest(contamination=0.05, random_state=42).fit(df).predict(df)
marcadas = pred == -1
ts = pd.Series(pd.to_datetime(ts))

gt = json.load(open(GABARITO))["incidentes"]
no_incidente = pd.Series(False, index=ts.index)
cobertura = []
for g in gt:
    dentro = (ts >= g["inicio"]) & (ts < g["fim"])
    no_incidente |= dentro
    cobertura.append((g["id"], g["tipo"], int((dentro & marcadas).sum()), int(dentro.sum())))

print("ABORDAGEM GUIADA (Etapa 3): Isolation Forest por linha, features hora/nível/tamanho/slow_query")
print(f"  linhas marcadas como anômalas: {marcadas.sum():,} de {len(df):,} ({marcadas.mean():.1%})")
tp = int((marcadas & no_incidente).sum())
print(f"  marcadas que estão dentro de incidentes reais: {tp:,} ({tp / marcadas.sum():.1%} das marcadas)")
print(f"  linhas de incidentes que foram marcadas: {tp:,} de {int(no_incidente.sum()):,} "
      f"({tp / no_incidente.sum():.1%})")
print("  cobertura por incidente (linhas marcadas / linhas no intervalo):")
for i, tipo, m, n in cobertura:
    print(f"    #{i:02d} {tipo:<22} {m:>5}/{n:<5} ({m / max(n, 1):.0%})")
print("  saída: uma linha por anomalia, sem horário completo, sem agrupamento nem causa\n")

rel = json.load(open(RELATORIO))
av = rel["avaliacao"]
print("LOG_ANALYZER.PY (Etapa 2): NLP + Isolation Forest por evento e por janela de 1 min")
print(f"  incidentes reportados: {len(rel['incidentes'])} | recall {av['recall']:.0%} | precisão {av['precisao']:.0%} "
      f"| causa correta {av['tipo_correto']}/{len(av['linhas'])}")
print("  saída: incidentes com início/fim, templates em excesso, hipótese de causa e ação sugerida")
