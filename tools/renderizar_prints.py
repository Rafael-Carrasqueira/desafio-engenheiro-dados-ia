"""
Renderiza a saída de texto de uma execução como imagem de terminal (o "print da execução" da entrega).

Uso:
    python tools/renderizar_prints.py <saida.txt> <imagem.png> "<comando exibido no topo>" [--linhas-por-pagina N]

Arquivos longos viram várias imagens (<imagem>_1.png, <imagem>_2.png, ...).
"""
import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FUNDO, BARRA, TEXTO = (12, 12, 12), (40, 40, 40), (204, 204, 204)
CORES = [
    (re.compile(r"^=+$"), (90, 90, 90)),
    (re.compile(r"^ \d\. "), (97, 214, 214)),                                   # títulos de seção
    (re.compile(r"^#\d+ "), (249, 241, 165)),                                    # cabeçalho de incidente
    (re.compile(r"Hipótese:"), (255, 170, 80)),
    (re.compile(r"^\s+\+ "), (180, 180, 255)),
    (re.compile(r"Ação sugerida"), (130, 220, 130)),
    (re.compile(r"Recall:|TREINADO|CARREGADO|=== "), (22, 198, 12)),
    (re.compile(r"\bNÃO\b|ERROR|\[aviso\]"), (231, 72, 86)),
    (re.compile(r"^Anomaly"), (220, 160, 160)),
]


def fonte(tam: int):
    for nome in ("C:/Windows/Fonts/consola.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                 "DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(nome, tam)
        except OSError:
            continue
    return ImageFont.load_default()


def cor(linha: str):
    for rx, c in CORES:
        if rx.search(linha):
            return c
    return TEXTO


def render(linhas: list[str], destino: Path, comando: str, titulo: str) -> None:
    f = fonte(15)
    alt_l = 19
    larg_c = f.getbbox("M")[2]
    colunas = max(102, max((len(l) for l in linhas), default=0) + 2)
    w = 24 + colunas * larg_c
    h = 44 + (len(linhas) + 2) * alt_l + 16
    img = Image.new("RGB", (w, h), FUNDO)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, 30], fill=BARRA)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([12 + i * 22, 9, 24 + i * 22, 21], fill=c)
    d.text((86, 7), titulo, font=fonte(14), fill=(200, 200, 200))
    y = 40
    d.text((12, y), "PS C:\\desafio-engenheiro-dados-ia> ", font=f, fill=(97, 214, 214))
    d.text((12 + len("PS C:\\desafio-engenheiro-dados-ia> ") * larg_c, y), comando, font=f, fill=(249, 241, 165))
    y += alt_l * 2
    for linha in linhas:
        d.text((12, y), linha, font=f, fill=cor(linha))
        y += alt_l
    img.save(destino, optimize=True)
    print(f"{destino} ({w}x{h})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("entrada", type=Path)
    p.add_argument("saida", type=Path)
    p.add_argument("comando")
    p.add_argument("--linhas-por-pagina", type=int, default=100)
    p.add_argument("--titulo", default="Windows PowerShell - docker compose")
    p.add_argument("--max-linhas", type=int, default=0, help="corta no meio arquivos enormes (cabeça + cauda)")
    a = p.parse_args()

    linhas = a.entrada.read_text(encoding="utf-8").expandtabs(4).splitlines()
    if a.max_linhas and len(linhas) > a.max_linhas:
        meio = a.max_linhas // 2
        omitidas = len(linhas) - a.max_linhas
        linhas = linhas[:meio] + [f"... ({omitidas:,} linhas omitidas no print; saída completa no .txt) ..."] + linhas[-meio:]
    paginas = [linhas[i:i + a.linhas_por_pagina] for i in range(0, len(linhas), a.linhas_por_pagina)]
    for n, pag in enumerate(paginas, 1):
        destino = a.saida if len(paginas) == 1 else a.saida.with_name(f"{a.saida.stem}_{n}{a.saida.suffix}")
        titulo = a.titulo + (f"  ({n}/{len(paginas)})" if len(paginas) > 1 else "")
        render(pag, destino, a.comando if n == 1 else f"{a.comando}   # continuação", titulo)


if __name__ == "__main__":
    main()
