"""
Etapa 1 - Gerador do arquivo de log sintético (app.log).

O enunciado pede que uma IA generativa produza um app.log com 1 milhão de linhas a partir
de 5 linhas de exemplo. Nenhum chat devolve 1 milhão de linhas; o caminho prático é pedir
à IA o *gerador* (ver docs/prompt_ia_generativa.md). Este script é o resultado desse prompt.

Características dos dados (dados sintéticos que preservam padrões de produção):
- formato:  "YYYY-MM-DD HH:MM:SS LEVEL message"  (LEVEL em INFO, DEBUG, WARN, ERROR)
- 7 dias (2024-11-04 a 2024-11-10), sazonalidade diária e fim de semana mais calmo
- ruído realista: erros e falhas de login isolados, ~4% de queries acima de 100 ms
- 12 incidentes plantados (6 tipos), gravados em data/incidentes.json como gabarito
  para avaliar se a IA encontra o que deveria encontrar

Uso:
    python gerar_logs.py                       # gera data/app.log + data/incidentes.json
    python gerar_logs.py --linhas 200000       # arquivo menor para testes
    python gerar_logs.py --live --saida /data/live.log   # gera logs em tempo real (Etapa 4)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
INICIO = datetime(2024, 11, 4)          # segunda-feira
DIAS = 7
MINUTOS = DIAS * 24 * 60

PRIMEIROS = ["ana", "bruno", "carla", "diego", "elisa", "felipe", "gabriela", "heitor",
             "isabela", "joao", "karina", "lucas", "marina", "nicolas", "olivia", "pedro",
             "rafaela", "samuel", "tatiana", "vitor"]
SOBRENOMES = ["silva", "souza", "oliveira", "santos", "lima", "costa", "mendes", "rocha",
              "almeida", "barbosa", "ferreira", "gomes", "ribeiro", "carvalho", "martins"]
PAGINAS = ["dashboard", "reports", "settings", "profile", "billing", "orders", "inventory"]
RECURSOS = ["orders", "customers", "products", "invoices", "payments", "users", "search"]
JOBS = ["backup-db", "sync-crm", "rebuild-index", "send-newsletter", "cleanup-tmp", "etl-sales"]
IPS_ATACANTES = ["185.220.101.47", "45.155.205.233"]


def _usuarios(rng: random.Random) -> list[str]:
    nomes = {f"{rng.choice(PRIMEIROS)}.{rng.choice(SOBRENOMES)}" for _ in range(400)}
    return sorted(nomes)


# --------------------------------------------------------------------------------------
# Tráfego normal
# --------------------------------------------------------------------------------------
def fator_horario(dt: datetime) -> float:
    """Intensidade relativa do tráfego (1.0 = pico de horário comercial)."""
    h = dt.hour + dt.minute / 60
    if h < 6:
        f = 0.18
    elif h < 9:
        f = 0.18 + (h - 6) / 3 * 0.82
    elif h < 12:
        f = 1.0
    elif h < 14:
        f = 0.8
    elif h < 18:
        f = 1.0
    elif h < 23:
        f = 1.0 - (h - 18) / 5 * 0.75
    else:
        f = 0.22
    if dt.weekday() >= 5:
        f *= 0.45
    return f


class GeradorNormal:
    """Gera uma mensagem de tráfego normal; os pesos definem o mix de níveis."""

    def __init__(self, seed: int = 42):
        self.r = random.Random(seed)
        self.usuarios = _usuarios(self.r)
        self.comercial = True
        r = self.r
        self.modelos = [
            # (peso, nível, função que gera a mensagem)
            (14, "INFO", lambda: f"User {self._u()} logged in"),
            (10, "INFO", lambda: f"User {self._u()} logged out"),
            (20, "INFO", lambda: f"User {self._u()} accessed {r.choice(PAGINAS)}"),
            (14, "INFO", lambda: f"Request GET /api/{r.choice(RECURSOS)} completed with status 200 in {self._ms(120, .5)}ms"),
            (1.2, "INFO", lambda: f"Scheduled job {r.choice(JOBS)} completed in {r.randint(2, 240)}s"),
            (1.5, "INFO", lambda: f"Report rpt-{r.randint(1000, 9999)} exported by user {self._u()} ({r.randint(50, 5000)} rows)"),
            (1.0, "INFO", lambda: f"Cache hit ratio at {r.randint(82, 99)}%"),
            (22, "DEBUG", lambda: f"Query executed in {self._ms(35, .6)}ms"),
            (6, "DEBUG", lambda: f"Cache miss for key {r.choice(RECURSOS)}:{r.randint(1, 99999)}"),
            (3, "DEBUG", lambda: f"Opening connection to database (pool {r.randint(5, 30)}/50 in use)"),
            (2.2, "WARN", lambda: f"Slow response on /api/{r.choice(RECURSOS)}: {r.randint(800, 2000)}ms"),
            (1.6, "WARN", lambda: f"Failed login attempt for user {self._u()} from 10.0.{r.randint(0, 20)}.{r.randint(2, 254)}"),
            (0.9, "WARN", lambda: "Connection pool running low"),
            (0.5, "WARN", lambda: f"Disk usage at {r.randint(75, 88)}%"),
            (0.8, "WARN", lambda: f"Retrying request to payment-gateway (attempt {r.randint(1, 3)})"),
            (0.35, "ERROR", lambda: "Failed to connect to database"),
            (0.45, "ERROR", lambda: f"Timeout while calling payment-gateway after {r.randint(5000, 15000)}ms"),
            (0.4, "ERROR", lambda: f"Request GET /api/{r.choice(RECURSOS)} failed with status 500"),
            (0.25, "ERROR", lambda: f"Unhandled exception in /api/{r.choice(RECURSOS)}: KeyError"),
        ]
        pesos = np.array([m[0] for m in self.modelos], dtype=float)
        self.cum = np.cumsum(pesos / pesos.sum())

    def _u(self) -> str:
        # a conta admin só é usada em horário comercial de dias úteis, e pouco
        if self.comercial and self.r.random() < 0.01:
            return "admin"
        return self.r.choice(self.usuarios)

    def _ms(self, mediana: float, sigma: float) -> int:
        return max(1, int(self.r.lognormvariate(np.log(mediana), sigma)))

    def mensagem(self, dt: datetime) -> tuple[str, str]:
        self.comercial = dt.weekday() < 5 and 8 <= dt.hour < 19
        i = int(np.searchsorted(self.cum, self.r.random()))
        _, nivel, f = self.modelos[min(i, len(self.modelos) - 1)]
        return nivel, f()


# --------------------------------------------------------------------------------------
# Incidentes plantados
# --------------------------------------------------------------------------------------
@dataclass
class Incidente:
    tipo: str
    inicio: datetime
    duracao_min: int

    @property
    def fim(self) -> datetime:
        return self.inicio + timedelta(minutes=self.duracao_min)


DESCRICOES = {
    "queda_banco": "Rajada de falhas de conexão com o banco de dados",
    "degradacao_latencia": "Degradação de latência: queries e APIs até 10x mais lentas",
    "forca_bruta": "Força bruta de login a partir de IP externo, terminando em login bem-sucedido",
    "admin_fora_horario": "Acesso administrativo de madrugada com alteração de permissões e exportação",
    "esgotamento_pool": "Esgotamento do pool de conexões com timeouts em cascata",
    "exfiltracao": "Exportação em massa de relatórios por um único usuário",
}

AGENDA = [
    Incidente("queda_banco", datetime(2024, 11, 4, 10, 12), 11),
    Incidente("degradacao_latencia", datetime(2024, 11, 4, 15, 40), 30),
    Incidente("forca_bruta", datetime(2024, 11, 5, 2, 47), 8),
    Incidente("esgotamento_pool", datetime(2024, 11, 5, 14, 5), 17),
    Incidente("admin_fora_horario", datetime(2024, 11, 6, 3, 14), 8),
    Incidente("exfiltracao", datetime(2024, 11, 6, 11, 30), 14),
    Incidente("forca_bruta", datetime(2024, 11, 7, 9, 15), 8),
    Incidente("degradacao_latencia", datetime(2024, 11, 7, 20, 35), 15),
    Incidente("queda_banco", datetime(2024, 11, 8, 10, 1), 8),
    Incidente("admin_fora_horario", datetime(2024, 11, 9, 1, 40), 7),
    Incidente("esgotamento_pool", datetime(2024, 11, 9, 16, 10), 18),
    Incidente("exfiltracao", datetime(2024, 11, 10, 22, 5), 15),
]


def linhas_incidente(inc: Incidente, r: random.Random, usuarios: list[str]) -> list[tuple[datetime, str, str]]:
    """Linhas extras injetadas durante o incidente (somam-se ao tráfego normal)."""
    out: list[tuple[datetime, str, str]] = []

    def em(minuto: int, n: int, gerar):
        for _ in range(n):
            ts = inc.inicio + timedelta(minutes=minuto, seconds=r.randint(0, 59))
            nivel, msg = gerar()
            out.append((ts, nivel, msg))

    for m in range(inc.duracao_min):
        if inc.tipo == "queda_banco":
            em(m, r.randint(30, 45), lambda: ("ERROR", "Failed to connect to database"))
            em(m, r.randint(8, 14), lambda: ("ERROR", "Query failed: connection reset by peer"))
            em(m, r.randint(5, 10), lambda: ("ERROR", f"Request GET /api/{r.choice(RECURSOS)} failed with status 500"))
        elif inc.tipo == "degradacao_latencia":
            em(m, r.randint(25, 40), lambda: ("DEBUG", f"Query executed in {int(r.lognormvariate(np.log(450), .4))}ms"))
            em(m, r.randint(12, 20), lambda: ("WARN", f"Slow response on /api/{r.choice(RECURSOS)}: {r.randint(2500, 9000)}ms"))
        elif inc.tipo == "forca_bruta":
            ip = IPS_ATACANTES[0] if inc.inicio.hour < 6 else IPS_ATACANTES[1]
            alvos = ["admin", "root", "administrator", "test", "suporte"]
            em(m, r.randint(50, 70), lambda: ("WARN", f"Failed login attempt for user {r.choice(alvos)} from {ip}"))
            if m == inc.duracao_min - 1:
                out.append((inc.inicio + timedelta(minutes=m, seconds=59), "INFO",
                            f"User admin logged in from {ip}"))
        elif inc.tipo == "admin_fora_horario":
            acoes = [
                ("INFO", "User admin logged in"),
                ("INFO", "User admin accessed user-management"),
                ("WARN", f"Permission changed: user {r.choice(usuarios)} granted role admin by admin"),
                ("INFO", "User admin exported table customers (1250000 rows)"),
                ("WARN", "Audit log retention policy disabled by admin"),
                ("INFO", "User admin accessed settings"),
            ]
            if m < len(acoes):
                nivel, msg = acoes[m]
                out.append((inc.inicio + timedelta(minutes=m, seconds=r.randint(0, 59)), nivel, msg))
            if m == inc.duracao_min - 1:
                out.append((inc.inicio + timedelta(minutes=m, seconds=50), "INFO", "User admin logged out"))
        elif inc.tipo == "esgotamento_pool":
            fase = m / inc.duracao_min
            em(m, r.randint(10, 18), lambda: ("WARN", f"Connection pool running low ({r.randint(45, 49)}/50 in use)"))
            if fase > 0.3:
                em(m, r.randint(8, 15), lambda: ("ERROR", "Connection pool exhausted (50/50 in use), request queued"))
                em(m, r.randint(5, 10), lambda: ("ERROR", "Timeout waiting for connection from pool after 30000ms"))
        elif inc.tipo == "exfiltracao":
            quem = "carla.mendes" if inc.inicio.day == 6 else "vitor.rocha"
            em(m, r.randint(20, 30), lambda: ("INFO", f"Report rpt-{r.randint(1000, 9999)} exported by user {quem} ({r.randint(400000, 900000)} rows)"))
    return out


# --------------------------------------------------------------------------------------
# Geração do arquivo completo
# --------------------------------------------------------------------------------------
def gerar_arquivo(total_linhas: int, saida: Path, gabarito: Path, seed: int = 42) -> None:
    t0 = time.time()
    r = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    normal = GeradorNormal(seed)

    # 1) incidentes primeiro, para fechar exatamente total_linhas
    extras: list[tuple[datetime, str, str]] = []
    resumo = []
    for i, inc in enumerate(AGENDA, 1):
        linhas = linhas_incidente(inc, r, normal.usuarios)
        extras.extend(linhas)
        resumo.append({
            "id": i, "tipo": inc.tipo, "descricao": DESCRICOES[inc.tipo],
            "inicio": inc.inicio.strftime("%Y-%m-%d %H:%M:%S"),
            "fim": inc.fim.strftime("%Y-%m-%d %H:%M:%S"),
            "linhas_injetadas": len(linhas),
        })

    n_normal = total_linhas - len(extras)
    if n_normal <= 0:
        raise ValueError("total de linhas pequeno demais para os incidentes")

    # 2) distribui o tráfego normal pelos minutos de acordo com a sazonalidade
    minutos = [INICIO + timedelta(minutes=m) for m in range(MINUTOS)]
    pesos = np.array([fator_horario(dt) for dt in minutos])
    pesos *= np_rng.lognormal(0, 0.08, size=MINUTOS)        # variação natural
    contagens = np_rng.multinomial(n_normal, pesos / pesos.sum())

    # 3) agrupa as linhas extras por minuto para intercalar ordenado
    extras_por_min: dict[int, list] = {}
    for ts, nivel, msg in extras:
        idx = int((ts - INICIO).total_seconds() // 60)
        extras_por_min.setdefault(idx, []).append((ts, nivel, msg))

    saida.parent.mkdir(parents=True, exist_ok=True)
    escritas = 0
    with open(saida, "w", encoding="utf-8", newline="\n") as f:
        for idx, dt in enumerate(minutos):
            linhas = []
            for s in np_rng.integers(0, 60, size=contagens[idx]):
                nivel, msg = normal.mensagem(dt)
                linhas.append((dt + timedelta(seconds=int(s)), nivel, msg))
            linhas.extend(extras_por_min.get(idx, []))
            linhas.sort(key=lambda x: x[0])
            f.writelines(f"{ts:%Y-%m-%d %H:%M:%S} {nivel} {msg}\n" for ts, nivel, msg in linhas)
            escritas += len(linhas)

    gabarito.write_text(json.dumps({
        "arquivo": saida.name, "linhas": escritas, "seed": seed,
        "periodo": [INICIO.strftime("%Y-%m-%d"), (INICIO + timedelta(days=DIAS - 1)).strftime("%Y-%m-%d")],
        "incidentes": resumo,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{escritas:,} linhas gravadas em {saida} ({saida.stat().st_size / 1e6:.1f} MB) "
          f"em {time.time() - t0:.1f}s")
    print(f"{len(resumo)} incidentes plantados ({len(extras):,} linhas) -> {gabarito}")


# --------------------------------------------------------------------------------------
# Modo ao vivo (Etapa 4): escreve linhas com o horário atual e injeta incidentes
# --------------------------------------------------------------------------------------
def modo_live(saida: Path, linhas_por_min: float, intervalo_incidente: int, seed: int) -> None:
    r = random.Random(seed)
    normal = GeradorNormal(seed)
    tipos = list(DESCRICOES)
    saida.parent.mkdir(parents=True, exist_ok=True)
    proximo_incidente = time.time() + intervalo_incidente
    pendentes: list[tuple[datetime, str, str]] = []
    print(f"[live] escrevendo em {saida} (~{linhas_por_min:.0f} linhas/min no pico, "
          f"incidente a cada {intervalo_incidente}s)", flush=True)

    with open(saida, "a", encoding="utf-8", newline="\n") as f:
        while True:
            agora = datetime.now().replace(microsecond=0)
            if time.time() >= proximo_incidente:
                tipo = r.choice(tipos)
                dur = 3 if tipo not in ("admin_fora_horario",) else 7
                inc = Incidente(tipo, agora.replace(second=0) + timedelta(minutes=1), dur)
                pendentes.extend(linhas_incidente(inc, r, normal.usuarios))
                print(f"[live] incidente agendado: {tipo} {inc.inicio:%H:%M}-{inc.fim:%H:%M}", flush=True)
                proximo_incidente = time.time() + intervalo_incidente

            n = np.random.poisson(linhas_por_min * fator_horario(agora) / 60)
            lote = [(agora, *normal.mensagem(agora)) for _ in range(n)]
            lote += [p for p in pendentes if p[0] <= agora]
            pendentes = [p for p in pendentes if p[0] > agora]
            lote.sort(key=lambda x: x[0])
            f.writelines(f"{agora:%Y-%m-%d %H:%M:%S} {nivel} {msg}\n" for _, nivel, msg in lote)
            f.flush()
            time.sleep(max(0.0, 1 - (time.time() % 1)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--linhas", type=int, default=1_000_000)
    p.add_argument("--saida", type=Path, default=Path(os.getenv("LOG_FILE", BASE_DIR / "data" / "app.log")))
    p.add_argument("--gabarito", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--live", action="store_true", help="gera logs continuamente com o horário atual")
    p.add_argument("--linhas-por-min", type=float, default=150.0)
    p.add_argument("--intervalo-incidente", type=int, default=300, help="segundos entre incidentes (live)")
    a = p.parse_args()

    if a.live:
        modo_live(a.saida, a.linhas_por_min, a.intervalo_incidente, a.seed)
    else:
        gabarito = a.gabarito or a.saida.with_name("incidentes.json")
        gerar_arquivo(a.linhas, a.saida, gabarito, a.seed)


if __name__ == "__main__":
    main()
