"""
log_analyzer.py - Análise de logs com Inteligência Artificial (Desafio Final - Engenheiro(a) de Dados IA Expert)

Lê o data/app.log (formato "YYYY-MM-DD HH:MM:SS LEVEL message") e combina três técnicas:

  1. NLP  - mascaramento de variáveis (números, IPs, usuários) transforma 1 milhão de mensagens em
            poucas dezenas de *templates*; TF-IDF + K-Means agrupa os templates em categorias de evento.
  2. Isolation Forest por EVENTO  - aponta linhas individualmente raras (ex.: admin exportando uma
            tabela inteira de madrugada), usando hora, severidade, latência, raridade do template etc.
  3. Isolation Forest por JANELA de 1 minuto - aponta minutos cujo comportamento agregado foge do
            padrão (taxa de erro, latência p95, falhas de login por IP, volume exportado...). Minutos
            anômalos consecutivos viram INCIDENTES.

Cada incidente é explicado comparando os templates observados com o esperado para aquele volume
(templates "em excesso") e recebe uma hipótese de causa por regras determinísticas - o modelo
probabilístico detecta, as regras explicam. Se existir data/incidentes.json (gabarito do gerador),
o programa mede precisão e recall da detecção.

Modos:
    python log_analyzer.py                 # analisa o log (treina o modelo na 1a vez e o salva)
    python log_analyzer.py --retrain       # força novo treino
    python log_analyzer.py --serve         # ai_service: acompanha um log ao vivo e expõe métricas
                                           # Prometheus em :8000 (Etapa 4)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as SKLEARN_VERSION
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import poisson
from sklearn.metrics import silhouette_score

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = Path(os.getenv("LOG_FILE", BASE_DIR / "data" / "app.log"))
MODEL_FILE = Path(os.getenv("MODEL_FILE", BASE_DIR / "model" / "anomaly_model.pkl"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "output"))
DATABASE_URL = os.getenv("DATABASE_URL")          # opcional: grava incidentes no Postgres

MODEL_VERSION = "2.4"
SURPRESA_MAX = 300.0              # teto da surpresa (-log10 p); evita infinito em rajadas enormes
CONTAMINACAO_EVENTO = 0.002       # ~0,2% das linhas
CONTAMINACAO_JANELA = float(os.getenv("CONTAMINACAO_JANELA", "0.015"))   # ~1,5% dos minutos (~150/semana)
LEVELS = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3}
PRIVILEGIADOS = ("admin", "root", "administrator")
LARGURA = 100

LOG_REGEX = r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (DEBUG|INFO|WARN|ERROR) (.+)$"

EVENT_FEATURES = ["hour", "fora_horario", "level_num", "log_latency", "raridade_template",
                  "privilegiado_fora_horario", "ip_externo", "log_rows"]
WINDOW_FEATURES = ["razao_volume", "z_erros", "z_warns", "surpresa_template", "log_queries_lentas",
                   "log_respostas_lentas", "log_falhas_mesmo_ip", "log_rows_exportadas", "eventos_raros",
                   "privilegiadas_fora_horario"]


# ======================================================================================
# 1. Leitura e extração de campos
# ======================================================================================
def carregar_logs(caminho: Path) -> tuple[pd.DataFrame, int]:
    """Lê o arquivo inteiro de forma vetorizada e separa timestamp, nível e mensagem."""
    with open(caminho, encoding="utf-8", errors="replace") as f:
        bruto = pd.Series(f.read().splitlines(), dtype="string")
    partes = bruto.str.extract(LOG_REGEX)
    validas = partes[0].notna()
    df = pd.DataFrame({
        "ts": pd.to_datetime(partes.loc[validas, 0], format="%Y-%m-%d %H:%M:%S"),
        "level": partes.loc[validas, 1].astype("category"),
        "message": partes.loc[validas, 2],
    }).reset_index(drop=True)
    return df, int((~validas).sum())


def linhas_para_df(linhas: list[str]) -> pd.DataFrame:
    """Versão para pequenos lotes (modo serviço)."""
    partes = pd.Series(linhas, dtype="string").str.extract(LOG_REGEX).dropna()
    return pd.DataFrame({
        "ts": pd.to_datetime(partes[0], format="%Y-%m-%d %H:%M:%S"),
        "level": partes[1], "message": partes[2],
    }).reset_index(drop=True)


def mascarar(msg: pd.Series) -> pd.Series:
    """Troca partes variáveis por marcadores -> template do evento (log parsing por NLP)."""
    s = msg.str.replace(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<IP>", regex=True)
    s = s.str.replace(r"(?i)\b(user )(?!(?:admin|root|administrator)\b)[\w.]+", r"\1<USER>", regex=True)
    s = s.str.replace(r"\d+", "<NUM>", regex=True)
    return s


def extrair_campos(df: pd.DataFrame) -> pd.DataFrame:
    msg = df["message"]
    df["template"] = mascarar(msg).astype("category")
    df["latency_ms"] = pd.to_numeric(msg.str.extract(r"(\d+)ms", expand=False), errors="coerce")
    df["rows"] = pd.to_numeric(msg.str.extract(r"\((\d+) rows\)", expand=False), errors="coerce")
    df["ip"] = msg.str.extract(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b", expand=False)
    df["user"] = msg.str.extract(r"(?i)\buser ([\w.]+)", expand=False)
    df["falha_login"] = msg.str.startswith("Failed login attempt").fillna(False)
    df["minuto"] = df["ts"].dt.floor("min")
    return df


def features_evento(df: pd.DataFrame, freq_template: dict[str, float]) -> pd.DataFrame:
    hora = df["ts"].dt.hour
    fim_de_semana = df["ts"].dt.weekday >= 5
    piso = min(freq_template.values()) / 10 if freq_template else 1e-7
    freq = df["template"].astype(str).map(freq_template).fillna(piso).astype(float)
    texto = df["message"].str.lower()
    fora = ((hora < 7) | (hora >= 22) | fim_de_semana).astype(int)
    privilegiado = texto.str.contains(r"\b(?:admin|root|administrator)\b", regex=True).fillna(False).astype(int)
    return pd.DataFrame({
        "hour": hora,
        "fora_horario": fora,
        "level_num": df["level"].astype(str).map(LEVELS).fillna(1).astype(int),
        "log_latency": np.log1p(df["latency_ms"].fillna(0)),
        "raridade_template": -np.log10(freq),
        "privilegiado": privilegiado,
        "privilegiado_fora_horario": privilegiado * fora,
        "ip_externo": (df["ip"].notna() & ~df["ip"].fillna("").str.startswith("10.")).astype(int),
        "log_rows": np.log1p(df["rows"].fillna(0)),
    })


# ======================================================================================
# 2. NLP - categorias de evento a partir dos templates
# ======================================================================================
def _sem_marcadores(t: str) -> str:
    return re.sub(r"<[A-Z]+>", " ", t).lower()


def treinar_categorias(templates: pd.Series, seed: int = 42):
    """TF-IDF nos templates únicos + K-Means; k escolhido pelo coeficiente de silhueta."""
    unicos = templates.astype(str).unique().tolist()
    vetorizador = TfidfVectorizer(token_pattern=r"[a-zA-Z]{3,}", stop_words="english", sublinear_tf=True)
    X = vetorizador.fit_transform([_sem_marcadores(t) for t in unicos])
    melhor = (None, -1.0, 0)
    for k in range(6, min(14, len(unicos) - 1)):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        s = silhouette_score(X, km.labels_)
        if s > melhor[1]:
            melhor = (km, s, k)
    km, sil, k = melhor
    termos = np.array(vetorizador.get_feature_names_out())
    rotulos = {}
    for c in range(k):
        centro = km.cluster_centers_[c]
        rotulos[c] = " / ".join(termos[np.argsort(centro)[::-1][:3]])
    return vetorizador, km, rotulos, sil


def categorizar(templates: pd.Series, vetorizador, km, rotulos) -> pd.Series:
    unicos = templates.astype(str).unique()
    ids = km.predict(vetorizador.transform([_sem_marcadores(t) for t in unicos]))
    mapa = {t: rotulos[i] for t, i in zip(unicos, ids)}
    return templates.astype(str).map(mapa)


# ======================================================================================
# 3. Features por janela de 1 minuto
# ======================================================================================
def features_janela(df: pd.DataFrame, ev: pd.DataFrame, ref: dict,
                    minutos: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Agrega cada minuto. As contagens de ERROR/WARN viram z-scores de Poisson em relação ao esperado
    para o volume daquele minuto: 1 erro em 10 linhas de madrugada não é incidente, 40 erros/min é."""
    idx = minutos if minutos is not None else pd.date_range(df["minuto"].min(), df["minuto"].max(), freq="min")

    def conta(mask) -> pd.Series:
        return df.loc[mask].groupby("minuto").size().reindex(idx, fill_value=0)

    lv = df["level"].astype(str)
    msg = df["message"]
    w = pd.DataFrame(index=idx)
    w["total"] = df.groupby("minuto").size().reindex(idx, fill_value=0)
    w["erros"] = conta(lv == "ERROR")
    w["warns"] = conta(lv == "WARN")
    w["p95_latencia"] = df.groupby("minuto")["latency_ms"].quantile(0.95).reindex(idx).fillna(0)
    w["queries_lentas"] = conta(msg.str.startswith("Query executed") & (df["latency_ms"] > 300))
    w["respostas_lentas"] = conta(msg.str.startswith("Slow response"))
    falhas = df[df["falha_login"]]
    w["falhas_login"] = conta(df["falha_login"])
    por_ip = falhas.groupby(["minuto", "ip"]).size()
    w["falhas_mesmo_ip"] = (por_ip.groupby(level=0).max() if len(por_ip) else pd.Series(dtype=float)).reindex(idx, fill_value=0)
    w["rows_exportadas"] = df.groupby("minuto")["rows"].sum().reindex(idx, fill_value=0)
    w["eventos_raros"] = (ev["raridade_template"] >= ref["limiar_raro"]).groupby(df["minuto"]).sum().reindex(idx, fill_value=0)
    w["acoes_privilegiadas"] = ev["privilegiado"].groupby(df["minuto"]).sum().reindex(idx, fill_value=0)

    fora = np.array([(t.hour < 7 or t.hour >= 22 or t.weekday() >= 5) for t in idx])
    w["privilegiadas_fora_horario"] = np.where(fora, w["acoes_privilegiadas"], 0)
    chave = [(t.hour, int(t.weekday() >= 5)) for t in idx]
    esperado = np.array([ref["baseline_volume"].get(k, 1.0) for k in chave], dtype=float)
    w["razao_volume"] = w["total"] / np.maximum(esperado, 1.0)
    for col, taxa in (("erros", "taxa_erro"), ("warns", "taxa_warn")):
        lam = w["total"] * ref[taxa]
        w["z_" + col] = (w[col] - lam) / np.sqrt(lam + 1)
    for col in ["queries_lentas", "respostas_lentas", "falhas_mesmo_ip", "rows_exportadas"]:
        w["log_" + col] = np.log1p(w[col])
    w["surpresa_template"] = surpresa_por_minuto(df, w["total"], ref["freq_template"]).reindex(idx, fill_value=0)
    return w


def surpresa_por_minuto(df: pd.DataFrame, total: pd.Series, freq: dict) -> pd.Series:
    """Maior surpresa estatística entre os templates do minuto: -log10 P(X >= n), X ~ Poisson(freq x volume).
    Liga o NLP à detecção: qualquer rajada de um tipo de mensagem (falha de login, exportação, pool...)
    aparece aqui, mesmo que não mude a severidade nem o volume total."""
    cnt = df.groupby(["minuto", "template"], observed=True).size()
    cnt = cnt[cnt > 0]
    if cnt.empty:
        return pd.Series(dtype=float)
    minutos = cnt.index.get_level_values(0)
    piso = min(freq.values()) / 10
    f = cnt.index.get_level_values(1).astype(str).map(lambda t: freq.get(t, piso)).to_numpy(dtype=float)
    lam = f * total.reindex(minutos).to_numpy(dtype=float)
    n = cnt.to_numpy(dtype=float)
    sup = np.where(n >= 2 * lam, np.minimum(-poisson.logsf(n - 1, lam) / math.log(10), SURPRESA_MAX), 0.0)
    return pd.Series(sup, index=minutos).groupby(level=0).max()


def referencia(df: pd.DataFrame) -> dict:
    """Estatísticas de referência do período de treino (volume típico por hora, taxas de base)."""
    por_min = df.groupby("minuto").size()
    chave = list(zip(por_min.index.hour, (por_min.index.weekday >= 5).astype(int)))
    lv = df["level"].astype(str)
    return {
        "baseline_volume": pd.Series(por_min.values, index=pd.MultiIndex.from_tuples(chave))
                             .groupby(level=[0, 1]).median().to_dict(),
        "taxa_erro": float((lv == "ERROR").mean()),
        "taxa_warn": float((lv == "WARN").mean()),
        "limiar_raro": float(-np.log10(1e-4)),       # template com frequência < 0,01%
        "p95_query": float(df.loc[df["message"].str.startswith("Query executed"), "latency_ms"].quantile(0.95)),
    }


# ======================================================================================
# 4. Treino / persistência do modelo (Etapa 4: treina uma vez, reutiliza)
# ======================================================================================
def treinar(df: pd.DataFrame, seed: int = 42) -> dict:
    t0 = time.time()
    freq = df["template"].astype(str).value_counts(normalize=True).to_dict()
    vetorizador, km, rotulos, sil = treinar_categorias(df["template"], seed)

    ev = features_evento(df, freq)
    amostra = ev.sample(n=min(200_000, len(ev)), random_state=seed)
    modelo_evento = IsolationForest(n_estimators=200, contamination=CONTAMINACAO_EVENTO, random_state=seed, n_jobs=-1)
    modelo_evento.fit(amostra[EVENT_FEATURES])

    ref = referencia(df)
    ref["freq_template"] = freq
    jan = features_janela(df, ev, ref)
    modelo_janela = IsolationForest(n_estimators=300, contamination=CONTAMINACAO_JANELA, random_state=seed, n_jobs=-1)
    modelo_janela.fit(jan[WINDOW_FEATURES])

    return {
        "versao": MODEL_VERSION,
        "treinado_em": datetime.now().isoformat(timespec="seconds"),
        "sklearn": SKLEARN_VERSION,
        "linhas_treino": int(len(df)),
        "periodo_treino": [str(df["ts"].min()), str(df["ts"].max())],
        "freq_template": freq,
        "vetorizador": vetorizador, "kmeans": km, "rotulos": rotulos, "silhueta": float(sil),
        "modelo_evento": modelo_evento, "modelo_janela": modelo_janela,
        "ref": ref,
        "mediana_janela": jan[WINDOW_FEATURES + ["total", "erros", "p95_latencia", "falhas_login",
                                                 "falhas_mesmo_ip", "rows_exportadas"]].median().to_dict(),
        "tempo_treino_s": round(time.time() - t0, 1),
    }


def carregar_ou_treinar(df: pd.DataFrame, retrain: bool) -> tuple[dict, bool]:
    if MODEL_FILE.exists() and not retrain:
        try:
            bundle = joblib.load(MODEL_FILE)
            if bundle.get("versao") == MODEL_VERSION and bundle.get("sklearn") == SKLEARN_VERSION:
                return bundle, False
            print(f"Modelo salvo incompatível (versão {bundle.get('versao')}/{bundle.get('sklearn')}); retreinando.")
        except Exception as e:                       # arquivo corrompido -> retreina
            print(f"Não foi possível carregar o modelo ({e}); retreinando.")
    bundle = treinar(df)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_FILE, compress=3)
    return bundle, True


# ======================================================================================
# 5. Incidentes: agrupamento, explicação e hipótese de causa
# ======================================================================================
def agrupar_incidentes(jan: pd.DataFrame, gap_min: int = 3) -> list[dict]:
    anom = jan[jan["anomalo"]].sort_index()
    grupos: list[list[pd.Timestamp]] = []
    for t in anom.index:
        if grupos and (t - grupos[-1][-1]) <= pd.Timedelta(minutes=gap_min):
            grupos[-1].append(t)
        else:
            grupos.append([t])
    out = []
    for g in grupos:
        bloco = jan.loc[g[0]:g[-1]]
        out.append({"inicio": g[0], "fim": g[-1] + pd.Timedelta(minutes=1),
                    "minutos_anomalos": len(g), "score": float(bloco["score"].max())})
    return out


def templates_em_excesso(df_inc: pd.DataFrame, freq: dict, top: int = 3) -> list[dict]:
    """Templates que aparecem muito mais do que o esperado para o volume da janela.
    Surpresa = -log10 P(X >= observado) numa Poisson com média = frequência de treino x volume."""
    obs = df_inc["template"].astype(str).value_counts()
    total = len(df_inc)
    piso = min(freq.values()) / 10
    linhas = []
    for tpl, n in obs.items():
        esperado = freq.get(tpl, piso) * total
        if n < esperado * 2:
            continue
        surpresa = min(float(-poisson.logsf(n - 1, esperado) / math.log(10)), SURPRESA_MAX)
        if surpresa >= 3:
            linhas.append({"template": tpl, "observado": int(n), "esperado": round(esperado, 2),
                           "lift": round(n / max(esperado, 1e-9), 1), "surpresa": round(surpresa, 1)})
    linhas.sort(key=lambda x: x["surpresa"], reverse=True)
    return linhas[:top]


# Hipótese de causa: o template mais "surpreendente" da janela decide (regras determinísticas
# sobre a saída do modelo probabilístico, como recomenda o Cap. 7 de Governança Inteligente).
REGRAS_CAUSA = [
    # (tipo, termos no template, hipótese, recomendação)
    ("forca_bruta", ("failed login attempt",),
     "Força bruta de login a partir de um único IP",
     "Bloquear o IP no WAF, exigir MFA e revisar a sessão aberta após as tentativas"),
    ("esgotamento_pool", ("connection pool", "from pool"),
     "Esgotamento do pool de conexões com o banco",
     "Aumentar o pool / investigar conexões presas e queries longas; aplicar backpressure"),
    ("queda_banco", ("connect to database", "connection reset"),
     "Indisponibilidade do banco de dados",
     "Verificar saúde do banco/rede, failover para réplica e circuit breaker nas APIs"),
    ("admin_fora_horario", ("exported table", "permission changed", "audit log", "user admin"),
     "Ação administrativa sensível fora do horário comercial",
     "Confirmar com o dono da conta admin, revogar permissões concedidas e auditar a exportação"),
    ("exfiltracao", ("exported by user",),
     "Exportação em massa de dados por um único usuário",
     "Suspender o usuário, avaliar vazamento (LGPD) e impor limite de linhas por exportação"),
    ("degradacao_latencia", ("slow response", "query executed"),
     "Degradação de performance (latência muito acima do normal)",
     "Checar plano de execução das queries, CPU/IO do banco e deploys recentes"),
]
SEM_CAUSA = ("desconhecido", "Padrão atípico sem regra conhecida", "Investigar manualmente os templates em excesso")


def classificar(inc: dict) -> tuple[str, str, str]:
    for t in inc["excesso"]:
        texto = t["template"].lower()
        for tipo, termos, hip, rec in REGRAS_CAUSA:
            if tipo == "admin_fora_horario" and not inc["fora_horario"]:
                continue                          # admin em horário comercial é rotina
            if any(x in texto for x in termos):
                return tipo, hip, rec
    if inc["p95_latencia"] >= 3 * max(inc["p95_ref"], 1):
        return REGRAS_CAUSA[-1][0], REGRAS_CAUSA[-1][2], REGRAS_CAUSA[-1][3]
    return SEM_CAUSA


def _p95_queries(d: pd.DataFrame) -> float:
    q = d.loc[d["message"].str.startswith("Query executed"), "latency_ms"]
    return float(q.quantile(0.95)) if len(q) else 0.0


def mesclar_incidentes(incs: list[dict], gap_min: int = 5) -> list[dict]:
    """Une incidentes vizinhos da mesma causa (ex.: exportação em massa com pausas de poucos minutos)."""
    out: list[dict] = []
    for i in sorted(incs, key=lambda x: x["inicio"]):
        ant = out[-1] if out else None
        if ant and ant["tipo"] == i["tipo"] and i["inicio"] - ant["fim"] <= pd.Timedelta(minutes=gap_min):
            ant["fim"] = max(ant["fim"], i["fim"])
            ant["score"] = max(ant["score"], i["score"])
            ant["minutos_anomalos"] += i["minutos_anomalos"]
            ant["_mesclar"] = True
        else:
            out.append(i)
    return out


def explicar_incidente(inc: dict, df: pd.DataFrame, ev: pd.DataFrame, jan: pd.DataFrame, bundle: dict,
                       contexto_min: int = 2) -> dict:
    """Descreve o incidente. A explicação olha 2 min antes/depois: o modelo costuma disparar um pouco
    depois do início real (a primeira fase de um esgotamento de pool, por exemplo, só gera WARN)."""
    ctx = pd.Timedelta(minutes=contexto_min)
    mask = (df["ts"] >= inc["inicio"] - ctx) & (df["ts"] < inc["fim"] + ctx)
    d = df[mask]
    bloco = jan.loc[inc["inicio"] - ctx:inc["fim"] + ctx]
    falhas = d[d["falha_login"]]
    ip_top = falhas["ip"].value_counts()
    exp = d[d["rows"].notna()].groupby("user")["rows"].sum().sort_values(ascending=False)
    nucleo = (df["ts"] >= inc["inicio"]) & (df["ts"] < inc["fim"])
    inc.update({
        "linhas": int(nucleo.sum()),
        "erros": int((df.loc[nucleo, "level"].astype(str) == "ERROR").sum()),
        "p95_latencia": _p95_queries(d),
        "p95_ref": float(bundle["ref"]["p95_query"]),
        "falhas_mesmo_ip": int(bloco["falhas_mesmo_ip"].max()) if len(bloco) else 0,
        "ip_suspeito": ip_top.index[0] if len(ip_top) else None,
        "rows_exportadas": int(d["rows"].sum()),
        "usuario_exportador": exp.index[0] if len(exp) else None,
        "acoes_privilegiadas": int(ev.loc[mask, "privilegiado"].sum()),
        "fora_horario": bool(inc["inicio"].hour < 7 or inc["inicio"].hour >= 22 or inc["inicio"].weekday() >= 5),
        "excesso": templates_em_excesso(d, bundle["freq_template"]),
    })
    inc["tipo"], inc["hipotese"], inc["recomendacao"] = classificar(inc)
    return inc


def evidencias(i: dict) -> list[str]:
    """Números que sustentam a hipótese, específicos de cada causa."""
    t = i["tipo"]
    out = []
    if t == "forca_bruta" and i["ip_suspeito"]:
        out.append(f"IP {i['ip_suspeito']} com até {i['falhas_mesmo_ip']} falhas de login/min")
    if t in ("exfiltracao", "admin_fora_horario") and i["rows_exportadas"] >= 1_000_000:
        out.append(f"{i['rows_exportadas']:,} linhas exportadas (top: {i['usuario_exportador']})")
    if t == "admin_fora_horario":
        out.append(f"{i['acoes_privilegiadas']} ações da conta admin às {i['inicio']:%H:%M}")
    if t == "degradacao_latencia":
        out.append(f"p95 das queries {i['p95_latencia']:.0f} ms (normal ~{i['p95_ref']:.0f} ms)")
    if t in ("queda_banco", "esgotamento_pool"):
        out.append(f"{i['erros']:,} ERROR em {i['linhas']:,} linhas ({i['erros'] / max(i['linhas'], 1):.0%})")
    return out


def confirmado(inc: dict) -> bool:
    """O modelo detecta; a confirmação exige explicação: causa identificada ou anomalia sustentada (>= 2 min)."""
    return inc["tipo"] != "desconhecido" or inc["minutos_anomalos"] >= 2


def avaliar(incidentes: list[dict], gabarito_path: Path, tolerancia_min: int = 1) -> dict | None:
    if not gabarito_path.exists():
        return None
    gt = json.loads(gabarito_path.read_text(encoding="utf-8"))["incidentes"]
    tol = pd.Timedelta(minutes=tolerancia_min)
    usados_det = set()
    linhas = []
    for g in gt:
        gi, gf = pd.Timestamp(g["inicio"]), pd.Timestamp(g["fim"])
        achou = None
        for j, d in enumerate(incidentes):
            if d["inicio"] < gf + tol and d["fim"] > gi - tol:
                achou = j if achou is None else achou
                usados_det.add(j)
        det = incidentes[achou] if achou is not None else None
        linhas.append({
            "id": g["id"], "tipo_real": g["tipo"], "inicio": g["inicio"][5:16],
            "detectado": det is not None,
            "atraso_min": round((det["inicio"] - gi).total_seconds() / 60, 1) if det else None,
            "tipo_previsto": det["tipo"] if det else None,
            "tipo_correto": bool(det and det["tipo"] == g["tipo"]),
        })
    tp = sum(l["detectado"] for l in linhas)
    recall = tp / len(gt) if gt else 0
    precisao = len(usados_det) / len(incidentes) if incidentes else 0
    return {"linhas": linhas, "recall": recall, "precisao": precisao,
            "falsos_positivos": len(incidentes) - len(usados_det),
            "tipo_correto": sum(l["tipo_correto"] for l in linhas)}


# ======================================================================================
# 6. Análise completa (modo lote)
# ======================================================================================
def secao(titulo: str) -> None:
    print("\n" + "=" * LARGURA)
    print(f" {titulo}")
    print("=" * LARGURA)


def analisar(caminho: Path, retrain: bool, gabarito: Path, top: int) -> dict:
    tempos = {}
    t = time.time()
    print(f"log_analyzer.py | modelo v{MODEL_VERSION} | scikit-learn {SKLEARN_VERSION} | "
          f"{datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Arquivo: {caminho}")
    df, malformadas = carregar_logs(caminho)
    tempos["leitura"] = time.time() - t

    t = time.time()
    df = extrair_campos(df)
    tempos["extracao_nlp"] = time.time() - t

    t = time.time()
    bundle, treinou = carregar_ou_treinar(df, retrain)
    tempos["treino" if treinou else "carga_modelo"] = time.time() - t

    t = time.time()
    df["categoria"] = categorizar(df["template"], bundle["vetorizador"], bundle["kmeans"], bundle["rotulos"])
    ev = features_evento(df, bundle["freq_template"])
    df["score_evento"] = -bundle["modelo_evento"].score_samples(ev[EVENT_FEATURES])
    df["anomalo_evento"] = bundle["modelo_evento"].predict(ev[EVENT_FEATURES]) == -1
    jan = features_janela(df, ev, bundle["ref"])
    jan["score"] = -bundle["modelo_janela"].score_samples(jan[WINDOW_FEATURES])
    jan["anomalo"] = bundle["modelo_janela"].predict(jan[WINDOW_FEATURES]) == -1
    candidatos = [explicar_incidente(i, df, ev, jan, bundle) for i in agrupar_incidentes(jan)]
    candidatos = [explicar_incidente(i, df, ev, jan, bundle) if i.pop("_mesclar", False) else i
                  for i in mesclar_incidentes(candidatos)]
    incidentes = sorted((i for i in candidatos if confirmado(i)), key=lambda i: i["inicio"])
    isolados = [i for i in candidatos if not confirmado(i)]
    tempos["deteccao"] = time.time() - t

    # ---------------------------------------------------------------- resumo do arquivo
    secao("1. RESUMO DO ARQUIVO")
    niveis = df["level"].astype(str).value_counts()
    print(f"Linhas válidas: {len(df):,} | malformadas: {malformadas:,} | "
          f"período: {df['ts'].min()} -> {df['ts'].max()}")
    print("Níveis: " + " | ".join(f"{n} {niveis.get(n, 0):,} ({niveis.get(n, 0) / len(df):.1%})"
                                   for n in ["INFO", "DEBUG", "WARN", "ERROR"]))
    q = df.loc[df["message"].str.startswith("Query executed"), "latency_ms"]
    print(f"Queries: {len(q):,} | mediana {q.median():.0f} ms | p95 {q.quantile(.95):.0f} ms | "
          f"lentas (>100 ms) {(q > 100).mean():.1%}")
    por_hora = df.groupby(df["ts"].dt.hour).size()
    erro_hora = (df["level"].astype(str) == "ERROR").groupby(df["ts"].dt.hour).mean()
    print(f"Hora de pico: {por_hora.idxmax():02d}h ({por_hora.max():,} linhas) | "
          f"maior taxa de erro: {erro_hora.idxmax():02d}h ({erro_hora.max():.2%})")
    print(f"Templates distintos (NLP): {df['template'].nunique()} a partir de "
          f"{df['message'].nunique():,} mensagens distintas")

    # ---------------------------------------------------------------- modelo
    secao("2. MODELO")
    estado = "TREINADO e salvo" if treinou else "CARREGADO de disco"
    print(f"{estado}: {MODEL_FILE} (v{bundle['versao']}, treinado em {bundle['treinado_em']}, "
          f"{bundle['linhas_treino']:,} linhas)")
    print(f"NLP: TF-IDF + K-Means -> {len(bundle['rotulos'])} categorias (silhueta {bundle['silhueta']:.2f})")
    print(f"Isolation Forest por evento: {len(EVENT_FEATURES)} features, contamination={CONTAMINACAO_EVENTO:.1%}")
    print(f"Isolation Forest por janela de 1 min: {len(WINDOW_FEATURES)} features, contamination={CONTAMINACAO_JANELA:.1%}, "
          f"{len(jan):,} janelas")

    # ---------------------------------------------------------------- categorias
    secao("3. CATEGORIAS DE EVENTO (NLP: TF-IDF + K-MEANS)")
    cat = df.groupby("categoria", observed=True).agg(
        linhas=("message", "size"),
        templates=("template", "nunique"),
        pct_error=("level", lambda s: (s.astype(str) == "ERROR").mean()),
    ).sort_values("linhas", ascending=False)
    print(f"{'categoria (termos mais fortes)':<44}{'linhas':>11}{'%':>8}{'templates':>11}{'% ERROR':>10}")
    for nome, r in cat.iterrows():
        print(f"{nome[:43]:<44}{int(r.linhas):>11,}{r.linhas / len(df):>8.1%}{int(r.templates):>11}{r.pct_error:>10.1%}")

    # ---------------------------------------------------------------- incidentes
    secao(f"4. INCIDENTES DETECTADOS (Isolation Forest por janela) - {len(incidentes)} incidentes")
    for k, i in enumerate(incidentes, 1):
        dur = int((i["fim"] - i["inicio"]).total_seconds() // 60)
        print(f"\n#{k:02d} {i['inicio']:%a %d/%m %H:%M} -> {i['fim']:%H:%M} ({dur} min) | score {i['score']:.3f} | "
              f"{i['linhas']:,} linhas, {i['erros']:,} ERROR")
        print(f"    Hipótese: {i['hipotese']}  [{i['tipo']}]")
        detalhes = evidencias(i)
        if detalhes:
            print("    Evidências: " + "; ".join(detalhes))
        for e in i["excesso"]:
            print(f"    + {e['template'][:62]:<62} {e['observado']:>5}x (esperado {e['esperado']:.0f}, lift {e['lift']:.0f}x)")
        print(f"    Ação sugerida: {i['recomendacao']}")

    if isolados:
        print(f"\n{len(isolados)} alerta(s) isolado(s) de 1 min sem template em excesso foram descartados como ruído "
              f"(scores {min(i['score'] for i in isolados):.3f}-{max(i['score'] for i in isolados):.3f})")

    # ---------------------------------------------------------------- eventos raros
    secao("5. EVENTOS INDIVIDUAIS MAIS ANÔMALOS (Isolation Forest por evento)")
    an = df[df["anomalo_evento"]]
    dentro = pd.Series(False, index=df.index)
    for i in incidentes:
        dentro |= (df["ts"] >= i["inicio"]) & (df["ts"] < i["fim"])
    fora = an[~dentro[an.index]]
    print(f"{len(an):,} eventos anômalos ({len(an) / len(df):.2%} das linhas); "
          f"{len(an) - len(fora):,} dentro dos incidentes acima e {len(fora):,} fora deles.")
    print("Templates mais frequentes entre os anômalos FORA de incidentes (candidatos a investigação):")
    for tpl, n in fora["template"].astype(str).value_counts().head(5).items():
        print(f"  {n:>6,}x  {tpl[:85]}")
    print("\nMaiores scores (um exemplo por template):")
    exemplos = df[df["anomalo_evento"]].sort_values("score_evento", ascending=False).drop_duplicates("template")
    for _, r in exemplos.head(top).iterrows():
        print(f"  {r.score_evento:.3f}  {r.ts}  {r.level:<5} {r.message[:70]}")

    # ---------------------------------------------------------------- avaliação
    resultado_aval = avaliar(incidentes, gabarito)
    if resultado_aval:
        secao("6. AVALIAÇÃO CONTRA O GABARITO (data/incidentes.json)")
        print(f"{'id':>3} {'tipo real':<22}{'início':<14}{'detectado':<11}{'atraso':>8}  {'tipo previsto':<22}")
        for l in resultado_aval["linhas"]:
            atraso = f"{l['atraso_min']:+.0f} min" if l["detectado"] else "-"
            prev = (l["tipo_previsto"] or "-") + (" OK" if l["tipo_correto"] else "")
            print(f"{l['id']:>3} {l['tipo_real']:<22}{l['inicio']:<14}{'SIM' if l['detectado'] else 'NÃO':<11}"
                  f"{atraso:>8}  {prev:<22}")
        n = len(resultado_aval["linhas"])
        print(f"\nRecall: {resultado_aval['recall']:.0%} ({sum(l['detectado'] for l in resultado_aval['linhas'])}/{n}) | "
              f"Precisão: {resultado_aval['precisao']:.0%} | falsos positivos: {resultado_aval['falsos_positivos']} | "
              f"causa correta: {resultado_aval['tipo_correto']}/{n}")

    # ---------------------------------------------------------------- insights
    secao("7. INSIGHTS")
    for linha in gerar_insights(df, incidentes, q):
        print(f"- {linha}")

    total = sum(tempos.values())
    print("\nTempo: " + " | ".join(f"{k} {v:.1f}s" for k, v in tempos.items()) + f" | total {total:.1f}s")

    relatorio = montar_relatorio(df, malformadas, bundle, treinou, incidentes, resultado_aval, tempos, cat)
    salvar_relatorio(relatorio)
    gravar_incidentes_db(incidentes, origem="lote")
    return relatorio


def gerar_insights(df: pd.DataFrame, incidentes: list[dict], q: pd.Series) -> list[str]:
    out = []
    tipos = pd.Series([i["tipo"] for i in incidentes]).value_counts()
    if len(tipos):
        out.append("Incidentes por causa: " + ", ".join(f"{t} ({n})" for t, n in tipos.items()))
    inc_mask = pd.Series(False, index=df.index)
    for i in incidentes:
        inc_mask |= (df["ts"] >= i["inicio"]) & (df["ts"] < i["fim"])
    erros = df["level"].astype(str) == "ERROR"
    if erros.sum():
        out.append(f"{(erros & inc_mask).sum() / erros.sum():.0%} dos ERROR do período ocorreram dentro de "
                   f"incidentes que somam {inc_mask.mean():.2%} das linhas: priorizar alertas por janela, "
                   f"não por linha isolada")
    erros_fora = df[erros & ~inc_mask]["template"].astype(str).value_counts().head(2)
    if len(erros_fora):
        out.append("Erros recorrentes fora de incidentes (ruído de fundo): " +
                   "; ".join(f"'{t}' ({n:,})" for t, n in erros_fora.items()))
    seg = [i for i in incidentes if i["tipo"] in ("forca_bruta", "admin_fora_horario", "exfiltracao")]
    if seg:
        out.append(f"{len(seg)} incidentes são de SEGURANÇA e quase não geram ERROR (força bruta é WARN, "
                   f"exportação é INFO): monitorar só a severidade não os encontraria")
    ips = df[df["falha_login"] & ~df["ip"].fillna("").str.startswith("10.")]["ip"].value_counts()
    if len(ips):
        out.append("IPs externos com falhas de login: " + ", ".join(f"{ip} ({n:,})" for ip, n in ips.head(3).items()))
    lentas = (q > 100).mean()
    out.append(f"{lentas:.1%} das queries passam de 100 ms - o critério fixo 'slow_query > 100 ms' do código guiado "
               f"marca milhares de eventos normais; o modelo por janela compara com o p95 típico")
    return out


def montar_relatorio(df, malformadas, bundle, treinou, incidentes, aval, tempos, cat) -> dict:
    def ser(i):
        return {k: (v.isoformat() if isinstance(v, pd.Timestamp) else v) for k, v in i.items()}
    return {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "arquivo": str(LOG_FILE), "linhas": int(len(df)), "malformadas": malformadas,
        "periodo": [str(df["ts"].min()), str(df["ts"].max())],
        "niveis": df["level"].astype(str).value_counts().to_dict(),
        "modelo": {"arquivo": str(MODEL_FILE), "versao": bundle["versao"], "treinado_agora": treinou,
                   "treinado_em": bundle["treinado_em"], "categorias": list(bundle["rotulos"].values())},
        "categorias": {k: int(v) for k, v in cat["linhas"].items()},
        "incidentes": [ser(i) for i in incidentes],
        "avaliacao": aval,
        "tempos_s": {k: round(v, 2) for k, v in tempos.items()},
    }


def salvar_relatorio(rel: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "relatorio.json").write_text(json.dumps(rel, indent=2, ensure_ascii=False, default=str),
                                               encoding="utf-8")
    print(f"Relatório salvo em {OUTPUT_DIR / 'relatorio.json'}")


# ======================================================================================
# 7. Persistência opcional dos incidentes no Postgres (serviço db do docker-compose)
# ======================================================================================
def gravar_incidentes_db(incidentes: list[dict], origem: str) -> None:
    if not DATABASE_URL or not incidentes:
        return
    try:
        import psycopg
        with psycopg.connect(DATABASE_URL, connect_timeout=5) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS ai_incidentes (
                    id SERIAL PRIMARY KEY, origem TEXT, inicio TIMESTAMP, fim TIMESTAMP,
                    tipo TEXT, hipotese TEXT, score DOUBLE PRECISION, linhas INT, erros INT,
                    detalhes JSONB, registrado_em TIMESTAMP DEFAULT now(),
                    UNIQUE (origem, inicio))""")
            for i in incidentes:
                con.execute("""
                    INSERT INTO ai_incidentes (origem, inicio, fim, tipo, hipotese, score, linhas, erros, detalhes)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (origem, inicio) DO UPDATE SET fim=EXCLUDED.fim, tipo=EXCLUDED.tipo,
                        hipotese=EXCLUDED.hipotese, score=EXCLUDED.score, linhas=EXCLUDED.linhas,
                        erros=EXCLUDED.erros, detalhes=EXCLUDED.detalhes""",
                            (origem, i["inicio"].to_pydatetime(), i["fim"].to_pydatetime(), i["tipo"],
                             i["hipotese"], i["score"], i.get("linhas", 0), i.get("erros", 0),
                             json.dumps(i.get("excesso", []), ensure_ascii=False)))
        print(f"{len(incidentes)} incidente(s) gravado(s) no Postgres (ai_incidentes, origem={origem})")
    except Exception as e:
        print(f"[aviso] não foi possível gravar no Postgres: {e}")


# ======================================================================================
# 8. Modo serviço (Etapa 4): acompanha o log ao vivo e expõe métricas Prometheus
# ======================================================================================
def servir(live_log: Path, porta: int, retrain: bool) -> None:
    from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server

    if MODEL_FILE.exists() and not retrain:
        bundle = joblib.load(MODEL_FILE)
        print(f"[serve] modelo carregado de {MODEL_FILE} (treinado em {bundle['treinado_em']})", flush=True)
    else:
        if not LOG_FILE.exists():                   # clone limpo: gera o app.log de treino primeiro
            from gerar_logs import gerar_arquivo
            print(f"[serve] {LOG_FILE} não existe: gerando 1.000.000 de linhas de treino ...", flush=True)
            gerar_arquivo(1_000_000, LOG_FILE, LOG_FILE.with_name("incidentes.json"))
        print(f"[serve] modelo inexistente: treinando a partir de {LOG_FILE} ...", flush=True)
        df, _ = carregar_logs(LOG_FILE)
        bundle, _ = carregar_ou_treinar(extrair_campos(df), retrain=True)
        print(f"[serve] modelo treinado e salvo em {MODEL_FILE}", flush=True)

    m_linhas = Counter("ai_log_lines_total", "Linhas de log processadas", ["level"])
    m_cat = Counter("ai_log_lines_by_category_total", "Linhas por categoria NLP", ["category"])
    m_lat = Histogram("ai_log_latency_ms", "Latência extraída das mensagens (ms)",
                      buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000])
    m_falhas = Counter("ai_log_failed_logins_total", "Tentativas de login com falha", ["origem"])
    m_ev = Counter("ai_log_event_anomalies_total", "Eventos individuais anômalos (Isolation Forest por evento)")
    m_score = Gauge("ai_log_window_anomaly_score", "Score de anomalia da última janela de 1 min (maior = mais anômalo)")
    m_limiar = Gauge("ai_log_window_anomaly_threshold", "Limiar de decisão do modelo por janela")
    m_flag = Gauge("ai_log_window_anomalous", "1 se a última janela foi classificada como anômala")
    m_inc = Counter("ai_log_incidents_total", "Incidentes detectados e encerrados", ["tipo"])
    m_isolado = Counter("ai_log_isolated_alerts_total", "Alertas de 1 min descartados por falta de explicação")
    m_inc_aberto = Gauge("ai_log_incident_open", "1 enquanto há incidente em andamento")
    m_ts = Gauge("ai_log_last_window_timestamp_seconds", "Timestamp da última janela avaliada")
    m_info = Info("ai_model", "Metadados do modelo carregado")
    m_info.info({"versao": bundle["versao"], "treinado_em": bundle["treinado_em"],
                 "linhas_treino": str(bundle["linhas_treino"]), "sklearn": bundle["sklearn"]})
    m_limiar.set(-bundle["modelo_janela"].offset_)

    start_http_server(porta)
    print(f"[serve] métricas em http://0.0.0.0:{porta}/metrics | acompanhando {live_log}", flush=True)

    while not live_log.exists():
        time.sleep(1)
    buffer: dict[pd.Timestamp, list[str]] = {}
    aberto: dict | None = None
    historico_jan: list[pd.DataFrame] = []

    def avaliar_minuto(minuto: pd.Timestamp, linhas: list[str]) -> None:
        nonlocal aberto
        df = linhas_para_df(linhas)
        if df.empty:
            return
        df = extrair_campos(df)
        df["categoria"] = categorizar(df["template"], bundle["vetorizador"], bundle["kmeans"], bundle["rotulos"])
        ev = features_evento(df, bundle["freq_template"])
        for lv, n in df["level"].astype(str).value_counts().items():
            m_linhas.labels(lv).inc(n)
        for c, n in df["categoria"].value_counts().items():
            m_cat.labels(c).inc(n)
        for v in df["latency_ms"].dropna():
            m_lat.observe(v)
        f = df[df["falha_login"]]
        externas = (~f["ip"].fillna("").str.startswith("10.")).sum()
        m_falhas.labels("externa").inc(int(externas))
        m_falhas.labels("interna").inc(int(len(f) - externas))
        m_ev.inc(int((bundle["modelo_evento"].predict(ev[EVENT_FEATURES]) == -1).sum()))

        jan = features_janela(df, ev, bundle["ref"],
                              minutos=pd.DatetimeIndex([minuto]))
        score = float(-bundle["modelo_janela"].score_samples(jan[WINDOW_FEATURES])[0])
        anomalo = bool(bundle["modelo_janela"].predict(jan[WINDOW_FEATURES])[0] == -1)
        jan["score"], jan["anomalo"] = score, anomalo
        m_score.set(score)
        m_flag.set(int(anomalo))
        m_ts.set(minuto.timestamp())
        historico_jan.append((minuto, df, ev, jan))
        del historico_jan[:-30]
        print(f"[serve] {minuto:%H:%M} linhas={len(df):>4} score={score:.3f} "
              f"{'ANÔMALO' if anomalo else 'normal'}", flush=True)

        if anomalo and aberto is None:
            aberto = {"inicio": minuto, "fim": minuto + pd.Timedelta(minutes=1),
                      "minutos_anomalos": 1, "score": score, "ultimo": minuto}
            m_inc_aberto.set(1)
        elif anomalo and aberto:
            aberto.update(fim=minuto + pd.Timedelta(minutes=1), score=max(aberto["score"], score),
                          ultimo=minuto, minutos_anomalos=aberto["minutos_anomalos"] + 1)
        elif aberto and minuto - aberto["ultimo"] > pd.Timedelta(minutes=2):
            janelas = [h for h in historico_jan if aberto["inicio"] <= h[0] < aberto["fim"]]
            dfi = pd.concat([h[1] for h in janelas], ignore_index=True)
            evi = pd.concat([h[2] for h in janelas], ignore_index=True)
            ji = pd.concat([h[3] for h in janelas])
            inc = explicar_incidente(dict(aberto), dfi, evi, ji, bundle)
            m_inc_aberto.set(0)
            aberto = None
            if not confirmado(inc):
                m_isolado.inc()
                print(f"[serve] alerta isolado {inc['inicio']:%H:%M} descartado (sem explicação)", flush=True)
                return
            m_inc.labels(inc["tipo"]).inc()
            print(f"[serve] INCIDENTE {inc['inicio']:%H:%M}-{inc['fim']:%H:%M} {inc['tipo']}: {inc['hipotese']}",
                  flush=True)
            gravar_incidentes_db([inc], origem="live")

    with open(live_log, encoding="utf-8", errors="replace") as fh:
        fh.seek(0, os.SEEK_END)                     # só o que chegar a partir de agora
        parcial = ""
        while True:
            pedaco = fh.readline()
            if pedaco:
                parcial += pedaco
                if not parcial.endswith("\n"):
                    continue
                linha, parcial = parcial.rstrip("\n"), ""
                try:
                    minuto = pd.Timestamp(linha[:16])
                except ValueError:
                    continue
                buffer.setdefault(minuto, []).append(linha)
                continue
            # sem linhas novas: fecha janelas cujo minuto já passou (com 5 s de folga)
            limite = pd.Timestamp(datetime.now() - timedelta(seconds=65)).floor("min")
            for minuto in sorted(k for k in buffer if k <= limite):
                avaliar_minuto(minuto, buffer.pop(minuto))
            time.sleep(0.5)


# ======================================================================================
def main() -> None:
    p = argparse.ArgumentParser(description="Análise de logs com IA (Isolation Forest + NLP)")
    p.add_argument("--log", type=Path, default=LOG_FILE, help="arquivo de log (padrão: data/app.log)")
    p.add_argument("--retrain", action="store_true", help="ignora o modelo salvo e treina de novo")
    p.add_argument("--gabarito", type=Path, default=None, help="incidentes.json para avaliação")
    p.add_argument("--top", type=int, default=8, help="itens nas listas de top")
    p.add_argument("--serve", action="store_true", help="modo serviço com métricas Prometheus")
    p.add_argument("--live-log", type=Path, default=Path(os.getenv("LIVE_LOG", LOG_FILE.with_name("live.log"))))
    p.add_argument("--port", type=int, default=int(os.getenv("METRICS_PORT", "8000")))
    a = p.parse_args()

    if a.serve:
        servir(a.live_log, a.port, a.retrain)
    else:
        if not a.log.exists():
            sys.exit(f"Arquivo de log não encontrado: {a.log} (gere com: python gerar_logs.py)")
        analisar(a.log, a.retrain, a.gabarito or a.log.with_name("incidentes.json"), a.top)


if __name__ == "__main__":
    main()
