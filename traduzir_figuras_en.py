"""Gera versoes em ingles das figuras de grade usadas no artigo.

As figuras originais (em portugues) sao mantidas; para cada uma e escrito um
arquivo com o sufixo _en, em que apenas os titulos dos paineis e os rotulos das
linhas sao substituidos pelo texto em ingles. As imagens em si nao sao alteradas.

    python traduzir_figuras_en.py
"""

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib

FIGURAS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figuras")
FONTE = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", "DejaVuSans.ttf")

BRANCO = 245          # limiar para considerar um pixel como fundo branco
MIN_PAINEL = 120      # altura/largura minima, em pixels, de um painel de imagem


def _grupos(mascara, min_tam=1):
    """Intervalos [inicio, fim] contiguos em que a mascara booleana e verdadeira."""
    grupos, inicio = [], None
    for i, v in enumerate(mascara):
        if v and inicio is None:
            inicio = i
        elif not v and inicio is not None:
            if i - inicio >= min_tam:
                grupos.append((inicio, i - 1))
            inicio = None
    if inicio is not None and len(mascara) - inicio >= min_tam:
        grupos.append((inicio, len(mascara) - 1))
    return grupos


def _linhas_de_texto(bloco):
    """Alturas das linhas de texto dentro de uma faixa de titulo."""
    tem = bloco.any(axis=1)
    return [(a, b) for a, b in _grupos(tem)]


def _fonte(tamanho):
    return ImageFont.truetype(FONTE, tamanho)


def _tamanho_fonte(altura_linha):
    """Maior tamanho de fonte cuja altura de linha nao ultrapassa a medida."""
    melhor, erro_min = 10, None
    for tamanho in range(6, 80):
        caixa = _fonte(tamanho).getbbox("Ay")
        erro = abs((caixa[3] - caixa[1]) - altura_linha)
        if erro_min is None or erro < erro_min:
            melhor, erro_min = tamanho, erro
    return melhor


def _desenha_centralizado(desenho, texto, centro_x, topo, fonte, cor=(0, 0, 0), espaco=1.35):
    linhas = texto.split("\n")
    altura = fonte.getbbox("Ay")[3] - fonte.getbbox("Ay")[1]
    y = topo
    for linha in linhas:
        caixa = fonte.getbbox(linha)
        largura = caixa[2] - caixa[0]
        desenho.text((centro_x - largura / 2 - caixa[0], y - caixa[1]), linha, font=fonte, fill=cor)
        y += altura * espaco


def _texto_rotacionado(imagem, texto, centro_y, centro_x, fonte, cor=(0, 0, 0), espaco=1.35):
    """Escreve texto girado 90 graus (rotulo de linha, como no matplotlib)."""
    linhas = texto.split("\n")
    altura = fonte.getbbox("Ay")[3] - fonte.getbbox("Ay")[1]
    largura = max(fonte.getbbox(l)[2] for l in linhas)
    bloco = Image.new("RGB", (largura + 8, int(altura * espaco * len(linhas)) + 8), "white")
    d = ImageDraw.Draw(bloco)
    y = 4
    for linha in linhas:
        d.text((4, y), linha, font=fonte, fill=cor)
        y += altura * espaco
    bloco = bloco.rotate(90, expand=True)
    imagem.paste(bloco, (int(centro_x - bloco.width / 2), int(centro_y - bloco.height / 2)))


def traduzir(nome, titulos, rotulos_linha=None, saida=None):
    """Substitui os titulos de uma figura em grade pelos titulos em ingles."""
    origem = os.path.join(FIGURAS, nome)
    imagem = Image.open(origem).convert("RGB")
    arr = np.asarray(imagem)
    conteudo = (arr < BRANCO).any(axis=2)

    faixas = _grupos(conteudo.any(axis=1))
    paineis_por_faixa, titulos_por_faixa = [], []
    for a, b in faixas:
        if b - a + 1 >= MIN_PAINEL:
            colunas = [(c0, c1) for c0, c1 in _grupos(conteudo[a:b + 1].any(axis=0))
                       if c1 - c0 + 1 >= MIN_PAINEL]
            paineis_por_faixa.append(((a, b), colunas))
        else:
            if titulos_por_faixa and a - titulos_por_faixa[-1][1] <= 1.8 * (b - a + 1):
                titulos_por_faixa[-1] = (titulos_por_faixa[-1][0], b)
            else:
                titulos_por_faixa.append((a, b))

    desenho = ImageDraw.Draw(imagem)
    indice = 0
    for (linha_a, linha_b), colunas in paineis_por_faixa:
        # faixa de titulo imediatamente acima desta linha de paineis
        acima = [(a, b) for a, b in titulos_por_faixa if b < linha_a]
        if not acima:
            indice += len(colunas)
            continue
        faixa_a, faixa_b = acima[-1]
        bloco = conteudo[faixa_a:faixa_b + 1]
        linhas_texto = _linhas_de_texto(bloco)
        tamanho = _tamanho_fonte(max(b - a + 1 for a, b in linhas_texto))
        fonte = _fonte(tamanho)
        altura_linha = fonte.getbbox("Ay")[3] - fonte.getbbox("Ay")[1]

        for c0, c1 in colunas:
            if indice >= len(titulos):
                break
            texto = titulos[indice]
            indice += 1
            margem = 70
            desenho.rectangle([max(0, c0 - margem), faixa_a - 4,
                               min(imagem.width - 1, c1 + margem), faixa_b + 4], fill="white")
            n = len(texto.split("\n"))
            altura_total = altura_linha * (1 + 1.35 * (n - 1))
            topo = faixa_a + ((faixa_b - faixa_a + 1) - altura_total) / 2
            _desenha_centralizado(desenho, texto, (c0 + c1) / 2, topo, fonte)

    if rotulos_linha:
        # rotulos girados, a esquerda da primeira coluna de paineis
        primeira_coluna = min(c0 for _, colunas in paineis_por_faixa for c0, _ in colunas)
        faixa_rotulos = conteudo[:, :primeira_coluna - 5]
        larg = primeira_coluna - 5
        tamanho = None
        for (linha_a, linha_b), _ in paineis_por_faixa:
            bloco = faixa_rotulos[linha_a:linha_b + 1]
            if not bloco.any():
                continue
            cols = [(a, b) for a, b in _grupos(bloco.any(axis=0))]
            if tamanho is None:
                larguras = [b - a + 1 for a, b in cols]
                tamanho = _tamanho_fonte(max(larguras) if len(larguras) == 1 else max(larguras))
            break
        fonte = _fonte(tamanho or 14)
        for i, ((linha_a, linha_b), _) in enumerate(paineis_por_faixa):
            if i >= len(rotulos_linha):
                break
            desenho.rectangle([0, linha_a, larg, linha_b], fill="white")
            _texto_rotacionado(imagem, rotulos_linha[i], (linha_a + linha_b) / 2, larg / 2, fonte)

    destino = os.path.join(FIGURAS, saida or nome.replace(".png", "_en.png"))
    imagem.save(destino)
    print("gerado:", os.path.basename(destino))


ORIGINAL = "Original image"
NOSSO = "Our method"
GG = "Gaussian Grouping"

FIGURAS_GRADE = {
    "exp1_hdbscan_comparacao.png": dict(titulos=[
        ORIGINAL,
        "Default\n(mcs=500, ms=3, eom)",
        "Low min_cluster_size\n(mcs=100, ms=3, eom)",
        "High min_cluster_size\n(mcs=1000, ms=3, eom)",
        "High min_samples\n(mcs=500, ms=15, eom)",
        "leaf selection\n(mcs=500, ms=3, leaf)",
    ]),
    "exp2_loss_comparacao.png": dict(titulos=[
        ORIGINAL,
        "Default\n(T=0.05  m+=0.2  m-=0.1  cent=0.3)",
        "Tight margins\n(T=0.04  m+=0.05  m-=0.05  cent=0.3)",
        "Relaxed margins\n(T=0.07  m+=0.5  m-=0.5  cent=0.3)",
        "High centroid repulsion\n(T=0.05  m+=0.2  m-=0.1  cent=1.0)",
    ]),
    "exp3_arquiteturas_comparacao.png": dict(titulos=[
        ORIGINAL, "GAT (8 heads)", "GAT (4 heads)", "GCN (2 layers)", "GCN (4 layers)",
    ]),
    "exp5_treino_comparacao.png": dict(titulos=[
        ORIGINAL, "HDBSCAN on the input features",
        "GAT with 20 epochs", "Default (full training)",
    ]),
    "comparacao_teatime_frame_00002.png": dict(titulos=[ORIGINAL, NOSSO, GG]),
    "comparacao_ramen_frame_00006.png": dict(titulos=[ORIGINAL, NOSSO, GG]),
    "comparacao_figurines_frame_00041.png": dict(titulos=[ORIGINAL, NOSSO, GG]),
    "comparacao_waldo_kitchen_frame_00053.png": dict(titulos=[ORIGINAL, NOSSO, GG]),
    "multivistas_teatime.png": dict(titulos=[ORIGINAL, NOSSO, GG]),
    "consistencia_multivista_teatime.png": dict(
        titulos=[ORIGINAL, NOSSO, GG],
        rotulos_linha=["frame 00008\n(GG segments)", "frame 00063\n(GG segments)",
                       "frame 00106\n(GG: background)", "frame 00139\n(GG: background)"],
    ),
}


if __name__ == "__main__":
    for nome, spec in FIGURAS_GRADE.items():
        traduzir(nome, **spec)
