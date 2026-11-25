"""
Figuras do artigo da Computational Visual Media (artigo_cvm/) no tamanho final
de impressão.

O checklist da CVM exige que o texto das figuras tenha pelo menos o tamanho do
corpo do artigo (10 pt). As figuras da tese eram desenhadas grandes e reduzidas
pelo LaTeX, o que deixava os rótulos com ~4 pt. Aqui cada figura é criada já
com a largura em que entra no artigo (\\columnwidth = 84,25 mm ou \\textwidth =
174 mm), com fonte Times (STIX) de 10 pt em todo o texto, inclusive no interior
dos diagramas, e deve ser incluída no .tex exatamente com essa largura.

As imagens de cada painel são as mesmas das figuras da tese (conferidas painel
a painel): images_coverage de cada configuração, objects_pred do Gaussian
Grouping e a imagem original em resolução cheia do LERF-OVS. As figuras
originais em figuras/ não são alteradas; a saída vai para artigo_cvm/figures/.

    python gerar_figuras_cvm.py                 # todas
    python gerar_figuras_cvm.py --so diagramas  # só um grupo
"""
import argparse
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

RAIZ = os.path.dirname(os.path.abspath(__file__))
SAIDA = os.path.join(RAIZ, "artigo_cvm", "figures")
EXP = os.path.join(RAIZ, "experimentos_finais")
ARQ = os.path.join(EXP, "comparação entre arquiteturas")
LERF = "/home/victor/Downloads/lear_ovs/lerf_ovs"

COLUNA = 84.25 / 25.4   # \columnwidth da CVM, em polegadas
PAGINA = 174.0 / 25.4   # \textwidth
T_TITULO = 10           # corpo do texto do artigo
T_TEXTO = 10            # texto interno dos diagramas
DPI = 600
ASPECTO = 730 / 988     # altura/largura das imagens do LERF-OVS

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": T_TITULO, "pdf.fonttype": 42, "savefig.dpi": DPI,
})


def ler(caminho):
    im = cv2.imread(caminho, cv2.IMREAD_COLOR)
    assert im is not None, f"não encontrado: {caminho}"
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def original(cena, frame):
    """Imagem original em resolução cheia (988x730)."""
    return ler(f"{LERF}/{cena}/images/{frame}.jpg")


def cobertura(pasta, frame="frame_00002"):
    return ler(os.path.join(pasta, "images_coverage", f"{frame}.png"))


def salvar(fig, nome):
    os.makedirs(SAIDA, exist_ok=True)
    caminho = os.path.join(SAIDA, nome)
    fig.savefig(caminho, dpi=DPI, facecolor="white")
    plt.close(fig)
    w, h = fig.get_size_inches()
    print(f"  {nome}: {w * 25.4:.1f} x {h * 25.4:.1f} mm")


def mostrar(ax, im, extent=None):
    # Máscaras de segmentação sem interpolação, para manter as bordas das instâncias.
    mascara = im.shape[1] < 500
    ax.imshow(im, interpolation="nearest" if mascara else "antialiased",
              aspect="auto", extent=extent)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("0.7"); sp.set_linewidth(0.4)


# ----------------------------------------------------------------------------
# Grades de segmentação (Figs. 4, 5, 6 e 10)
# ----------------------------------------------------------------------------

def grade(paineis, colunas, nome, largura=COLUNA, gap=0.05):
    """paineis: lista de (título, imagem). Títulos acima de cada painel; a faixa
    de título de cada fileira acompanha o título com mais linhas daquela fileira."""
    linhas = (len(paineis) + colunas - 1) // colunas
    w_p = (largura - (colunas - 1) * gap) / colunas
    h_p = w_p * ASPECTO
    n_lin = [max(t.count("\n") + 1 for t, _ in paineis[r * colunas:(r + 1) * colunas])
             for r in range(linhas)]
    h_t = [n * T_TITULO * 1.2 / 72 + 0.04 for n in n_lin]
    altura = sum(h_t) + linhas * h_p + 0.01
    fig = plt.figure(figsize=(largura, altura))
    for i, (titulo, im) in enumerate(paineis):
        r, c = divmod(i, colunas)
        x0 = c * (w_p + gap)
        topo = altura - sum(h_t[:r + 1]) - r * h_p
        ax = fig.add_axes([x0 / largura, (topo - h_p) / altura, w_p / largura, h_p / altura])
        mostrar(ax, im)
        ax.set_title(titulo, fontsize=T_TITULO, pad=2, linespacing=1.1)
    salvar(fig, nome)


def figuras_grades():
    orig = original("teatime", "frame_00002")
    e1 = lambda n: cobertura(os.path.join(EXP, "experimento1", n, "gat"))
    e2 = lambda n: cobertura(os.path.join(EXP, "experimento2", n, "gat"))
    grade([("(a) Original image", orig),
           ("(b) Default", e1("padrao")),
           ("(c) Low min_cluster_size", e1("mcs_baixo")),
           ("(d) High min_cluster_size", e1("mcs_alto")),
           ("(e) High min_samples", e1("ms_alto")),
           ("(f) $leaf$ selection", e1("leaf"))],
          2, "exp1_hdbscan_comparacao_en.png")
    grade([("(a) Original image", orig),
           ("(b) Default", e1("padrao")),
           ("(c) Tight margins", e2("margens_apertadas")),
           ("(d) Relaxed margins", e2("margens_relaxadas")),
           ("(e) High centroid repulsion", e2("centroides_alto"))],
          2, "exp2_loss_comparacao_en.png")
    grade([("(a) Original image", orig),
           ("(b) GAT (8 heads)", cobertura(os.path.join(ARQ, "padrao", "gat"))),
           ("(c) GAT (4 heads)", cobertura(os.path.join(ARQ, "gat_4_cabecas", "gat"))),
           ("(d) GCN (2 layers)", cobertura(os.path.join(ARQ, "gcn_2_camadas", "gcn"))),
           ("(e) GCN (4 layers)", cobertura(os.path.join(ARQ, "gcn_4_camadas", "gcn")))],
          2, "exp3_arquiteturas_comparacao_en.png")
    grade([("(a) Original image", orig),
           ("(b) HDBSCAN on the\ninput features",
            cobertura(os.path.join(EXP, "experimento5", "sem_rede_r1", "gat"))),
           ("(c) GAT with 20 epochs",
            cobertura(os.path.join(EXP, "experimento5", "epoca20_r1", "gat"))),
           ("(d) Default (full training)", e1("padrao"))],
          2, "exp5_treino_comparacao_en.png")


# ----------------------------------------------------------------------------
# Comparação com o Gaussian Grouping (Fig. 7): uma linha por cena
# ----------------------------------------------------------------------------

COLUNAS_METODOS = ["Original image", "Our method", "Gaussian Grouping"]


def linha_comparacao(cena, frame, nome, largura, cabecalho):
    ims = [original(cena, frame),
           cobertura(os.path.join(EXP, "experimento4", cena, "gat"), frame),
           ler(os.path.join(EXP, "experimento4", "gaussian_grouping", cena, f"{frame}.png"))]
    gap = 0.05
    w_p = (largura - 2 * gap) / 3
    h_p = w_p * ASPECTO
    h_t = (T_TITULO * 1.25 / 72 + 0.03) if cabecalho else 0.0
    altura = h_p + h_t + 0.01
    fig = plt.figure(figsize=(largura, altura))
    for c, im in enumerate(ims):
        ax = fig.add_axes([c * (w_p + gap) / largura, 0.005 / altura, w_p / largura, h_p / altura])
        mostrar(ax, im)
        if cabecalho:
            ax.set_title(COLUNAS_METODOS[c], fontsize=T_TITULO, pad=2)
    salvar(fig, nome)


def figuras_comparacao():
    largura = 0.82 * PAGINA   # \includegraphics[width=0.82\textwidth]
    cenas = [("teatime", "frame_00002"), ("ramen", "frame_00006"),
             ("figurines", "frame_00041"), ("waldo_kitchen", "frame_00053")]
    for i, (cena, frame) in enumerate(cenas):
        # o cabeçalho das colunas aparece só na primeira linha da figura
        linha_comparacao(cena, frame, f"comparacao_{cena}_{frame}_en.png", largura, cabecalho=(i == 0))


# ----------------------------------------------------------------------------
# Multivista (Fig. 8) e consistência entre vistas (Fig. 9)
# ----------------------------------------------------------------------------

def grade_vistas(linhas, rotulos, nome, retangulos=None, linhas_rotulo=1):
    """linhas: lista de [original, nosso, gg]; rótulos à esquerda de cada linha."""
    largura = COLUNA
    margem = linhas_rotulo * T_TITULO * 1.2 / 72 + 0.04
    gap = 0.03
    w_p = (largura - margem - 2 * gap) / 3
    h_p = w_p * ASPECTO
    h_t = T_TITULO * 1.25 / 72 + 0.03
    altura = h_t + len(linhas) * h_p + (len(linhas) - 1) * gap + 0.01
    fig = plt.figure(figsize=(largura, altura))
    for r, ims in enumerate(linhas):
        topo = altura - h_t - r * (h_p + gap)
        for c, im in enumerate(ims):
            ax = fig.add_axes([(margem + c * (w_p + gap)) / largura, (topo - h_p) / altura,
                               w_p / largura, h_p / altura])
            H, W = ims[1].shape[:2]      # sistema de coordenadas da renderização
            mostrar(ax, im, extent=(0, W, H, 0))
            if retangulos is not None:
                x0, y0, x1, y1 = retangulos[r]
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                       edgecolor="yellow", linewidth=1.0))
            if r == 0:
                # "Gaussian Grouping" é um pouco mais largo que o painel: alinhado à
                # direita para não ser cortado na borda da figura
                ax.set_title(COLUNAS_METODOS[c], fontsize=T_TITULO, pad=2,
                             loc="right" if c == 2 else "center")
            if c == 0:
                ax.set_ylabel(rotulos[r], fontsize=T_TITULO, labelpad=2, linespacing=1.05)
    salvar(fig, nome)


def figura_multivistas(cena="teatime", n=5):
    import glob
    from gerar_figuras_multivistas import GT, escolher_vistas
    cams = json.load(open(f"{RAIZ}/output/{cena}/cameras.json"))
    anotados = {os.path.basename(p)[:-5] for p in glob.glob(f"{GT}/{cena}/*.json")}
    vistas = escolher_vistas(cena, cams, anotados, n)
    linhas, rotulos = [], []
    for c in vistas:
        f = c["img_name"]
        linhas.append([original(cena, f),
                       cobertura(os.path.join(EXP, "experimento4", cena, "gat"), f),
                       ler(f"{RAIZ}/output/{cena}/train/ours_30000/objects_pred/{c['id']:05d}.png")])
        rotulos.append(f"Frame {int(f.split('_')[1])}")
    print(f"  vistas: {rotulos}")
    grade_vistas(linhas, rotulos, f"multivistas_{cena}_en.png")


def figura_consistencia(cena="teatime"):
    from gerar_figura_desaparecimento import escolher_exemplo, espalhadas, moda_por_splat, observar
    cams, xyz, vistas = observar(cena)
    g = np.concatenate([x["idx"] for x in vistas])
    moda_gg = moda_por_splat(g, np.concatenate([x["gg"] for x in vistas]), len(xyz))
    _, obj, presentes, ausentes, _ = escolher_exemplo(cena, xyz, vistas)
    sel = [(l, True) for l in espalhadas(presentes, 2)] + [(l, False) for l in espalhadas(ausentes, 2)]
    sel.sort(key=lambda s: s[0][0])

    linhas, rotulos, retangulos = [], [], []
    for l, presente in sel:
        x = vistas[l[0]]
        cam = x["cam"]
        membros = np.flatnonzero(np.isin(x["idx"], np.flatnonzero(moda_gg == obj)))
        uu, vv = x["u"][membros], x["v"][membros]
        x0, x1 = np.percentile(uu, [2, 98]); y0, y1 = np.percentile(vv, [2, 98])
        retangulos.append((x0 - 3, y0 - 3, x1 + 3, y1 + 3))
        f = cam["img_name"]
        linhas.append([original(cena, f),
                       cobertura(os.path.join(EXP, "experimento4", cena, "gat"), f),
                       ler(f"{RAIZ}/output/{cena}/train/ours_30000/objects_pred/{cam['id']:05d}.png")])
        # se o GG segmenta ou apaga o objeto em cada frame fica na legenda do artigo;
        # como rótulo de linha em duas linhas ele não cabe na altura do painel
        rotulos.append(f"Frame {int(f.split('_')[1])}")
        print(f"  {rotulos[-1]}: GG {'segmenta' if presente else 'prediz fundo'}")
    grade_vistas(linhas, rotulos, f"consistencia_multivista_{cena}_en.png", retangulos=retangulos)


# ----------------------------------------------------------------------------
# Segmentação 3D (Fig. 3): nuvem de centros de splats colorida por cluster
# ----------------------------------------------------------------------------

def projetar_seg3d(elevacao=35.0, recuo=1.15, largura_px=2400, aspecto=0.75):
    """Projeta o gat_clusters.ply da configuração de referência (Experimento 1,
    padrão) numa câmera pinhole. O ponto de vista parte da câmera do frame_00002
    (o mesmo das Figs. 4 a 6), recuado e elevado em torno do centro da nuvem para
    mostrar a profundidade da cena. Devolve, para os pontos à frente da câmera, as
    posições 3D, as coordenadas x, y na imagem W x H, a profundidade e a cor."""
    from plyfile import PlyData
    v = PlyData.read(os.path.join(EXP, "experimento1", "padrao", "gat", "pointclouds",
                                  "gat_clusters.ply"))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    rgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.uint8)

    cams = json.load(open(os.path.join(RAIZ, "output", "teatime", "cameras.json")))
    cam = next(c for c in cams if c["img_name"] == "frame_00002")
    cima = -np.mean([np.array(c["rotation"])[:, 1] for c in cams], axis=0)
    cima /= np.linalg.norm(cima)

    centro = np.median(xyz, axis=0)
    d = np.array(cam["position"]) - centro
    dist = np.linalg.norm(d) * recuo
    horiz = d - (d @ cima) * cima
    horiz /= np.linalg.norm(horiz)
    ang = np.radians(elevacao)
    pos = centro + dist * (np.cos(ang) * horiz + np.sin(ang) * cima)

    frente = centro - pos; frente /= np.linalg.norm(frente)
    direita = np.cross(frente, cima); direita /= np.linalg.norm(direita)
    baixo = np.cross(frente, direita)
    p = (xyz - pos) @ np.stack([direita, baixo, frente], 1)
    frente_ok = p[:, 2] > 1e-3
    p, xyz, cores = p[frente_ok], xyz[frente_ok], rgb[frente_ok]
    u, w = p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]

    # enquadramento pelos percentis 1-99, para que pontos isolados não encolham a cena
    W, H = largura_px, int(round(largura_px * aspecto))
    u0, u1 = np.percentile(u, [1, 99]); w0, w1 = np.percentile(w, [1, 99])
    f = 0.88 * min(W / (u1 - u0), H / (w1 - w0))
    x = (u - (u0 + u1) / 2) * f + W / 2
    y = (w - (w0 + w1) / 2) * f + H / 2
    return xyz, x, y, p[:, 2], cores, W, H


def segmentacao_3d(raio=3.6, **kw):
    """Fig. 3: a nuvem projetada, com z-buffer por ordem de profundidade."""
    _, x, y, prof, cores, W, H = projetar_seg3d(**kw)
    img = np.zeros((H, W, 3), np.uint8)
    for i in np.argsort(-prof):             # do mais distante ao mais próximo
        cx, cy = x[i], y[i]
        if -5 < cx < W + 5 and -5 < cy < H + 5:
            c = tuple(int(k) for k in cores[i][::-1])
            cv2.circle(img, (int(round(cx * 4)), int(round(cy * 4))), int(raio * 4), c,
                       -1, cv2.LINE_AA, shift=2)
    os.makedirs(SAIDA, exist_ok=True)
    cv2.imwrite(os.path.join(SAIDA, "segmentacao_3d.png"), img)
    cobertura_px = (img.max(axis=2) > 0).mean()
    print(f"  segmentacao_3d.png: {W}x{H} px, {len(x)} pontos, {cobertura_px:.0%} da imagem coberta")


# ----------------------------------------------------------------------------
# Graphical abstract: entrada -> aprendizado em grafo -> saída
# ----------------------------------------------------------------------------

def overlay_deva(frame="frame_00002", alfa=0.55, semente=3):
    """Imagem original em resolução cheia com as máscaras SAM/DEVA por cima. As
    máscaras (data/teatime/object_mask) estão na resolução reduzida do treino e
    são ampliadas sem interpolação; o id 0 é fundo e fica sem cor."""
    orig = original("teatime", frame).astype(np.float32)
    ids = cv2.imread(os.path.join(RAIZ, "data", "teatime", "object_mask", f"{frame}.png"), 0)
    ids = cv2.resize(ids, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_NEAREST)
    paleta = np.random.default_rng(semente).integers(40, 256, (256, 3)).astype(np.float32)
    cor = paleta[ids]
    saida = np.where(ids[..., None] > 0, (1 - alfa) * orig + alfa * cor, orig)
    return saida.astype(np.uint8)


def recorte_grafo(janela=(700, 470, 1600, 1145), celula=38, k=5, fator_aresta=2.5):
    """Nós e arestas de um recorte da nuvem da Fig. 3 (janela x0, y0, x1, y1 na
    imagem de 2400 x 1800). Para o grafo ficar legível, a janela é dividida em
    células de `celula` px e cada célula guarda só o splat mais próximo da câmera;
    as arestas ligam cada nó aos k vizinhos mais próximos em 3D, descartando as
    mais longas que `fator_aresta` vezes a distância mediana ao vizinho mais próximo."""
    from scipy.spatial import cKDTree
    xyz, x, y, prof, cores, _, _ = projetar_seg3d()
    x0, y0, x1, y1 = janela
    dentro = np.flatnonzero((x >= x0) & (x < x1) & (y >= y0) & (y < y1))
    cel = ((x[dentro] - x0) // celula).astype(int) * 10000 + ((y[dentro] - y0) // celula).astype(int)
    ordem = np.lexsort((prof[dentro], cel))           # por célula, do mais próximo ao mais distante
    _, primeiro = np.unique(cel[ordem], return_index=True)
    nos = dentro[ordem[primeiro]]
    arv = cKDTree(xyz[nos])
    dist, viz = arv.query(xyz[nos], k=k + 1)
    limite = fator_aresta * np.median(dist[:, 1])
    arestas = {(min(i, j), max(i, j)) for i in range(len(nos)) for d, j in zip(dist[i, 1:], viz[i, 1:])
               if d <= limite}
    return x[nos], y[nos], cores[nos], sorted(arestas)


def graphical_abstract():
    """Resumo gráfico em 8:13 (26 x 16 cm, JPG a 300 dpi), exigido pela CVM como
    arquivo separado. Três colunas: entrada (imagem original e máscaras SAM/DEVA),
    aprendizado em grafo (recorte do grafo real de splats e as etapas GAT ->
    embeddings -> HDBSCAN) e saída (segmentação 3D da Fig. 3 e máscara 2D)."""
    # No artigo a figura entra com \textwidth (174 mm = 0,67 x 26 cm): 15 pt aqui viram 10 pt,
    # o corpo do texto, como nas demais figuras.
    T_CAB, T_ROT, T_CX, T_SUB = 22, 16, 16, 15
    L, H = 26 / 2.54, 16 / 2.54
    fig, ax = eixo_polegadas(L, H)
    m, seta_w = 0.12, 0.5
    cw = (L - 2 * m - 2 * seta_w) / 3
    ih = cw * 0.75
    xs = [m + i * (cw + seta_w) for i in range(3)]
    topo1 = H - 0.66
    topo2 = topo1 - ih - 0.62

    def painel(x0, topo, im, rotulo, extent=None):
        a = fig.add_axes([x0 / L, (topo - ih) / H, cw / L, ih / H])
        mostrar(a, im, extent=extent)
        ax.text(x0 + cw / 2, topo - ih - 0.2, rotulo, ha="center", va="center", fontsize=T_ROT)
        return a

    for x0, cab in zip(xs, ["Input", "Graph learning", "Output"]):
        ax.text(x0 + cw / 2, H - 0.3, cab, ha="center", va="center", fontsize=T_CAB,
                fontweight="bold", color="#7A2A87")

    # 1. entrada
    painel(xs[0], topo1, original("teatime", "frame_00002"), "Multi-view images")
    painel(xs[0], topo2, overlay_deva(), "SAM + DEVA masks")

    # 2. grafo de splats (recorte da Fig. 3) e etapas do aprendizado
    gx, gy, gc, arestas = recorte_grafo()
    x0j, y0j, x1j, y1j = 700, 470, 1600, 1145
    a = fig.add_axes([xs[1] / L, (topo1 - ih) / H, cw / L, ih / H])
    a.set_facecolor("black")
    for i, j in arestas:
        a.plot([gx[i], gx[j]], [gy[i], gy[j]], color="0.72", linewidth=0.6, alpha=0.8, zorder=1)
    a.scatter(gx, gy, s=24, c=gc / 255.0, linewidths=0, zorder=2)
    a.set_xlim(x0j, x1j); a.set_ylim(y1j, y0j)
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_edgecolor("0.7"); sp.set_linewidth(0.4)
    ax.text(xs[1] + cw / 2, topo1 - ih - 0.2, "Splat graph (xyz + color)", ha="center",
            va="center", fontsize=T_ROT)

    etapas = [("GAT", "contrastive loss", VERDE), ("Embeddings", "one vector per splat", VERDE),
              ("HDBSCAN", "+ KNN propagation", ROXO)]
    hb, gap = 0.70, 0.24
    y = topo2 - 0.02
    for n, (tit, sub, cores) in enumerate(etapas):
        ax.add_patch(FancyBboxPatch((xs[1] + 0.25, y - hb), cw - 0.5, hb,
                                    boxstyle="round,pad=0,rounding_size=0.06", linewidth=1.2, **cores))
        ax.text(xs[1] + cw / 2, y - hb * 0.31, tit, ha="center", va="center", fontsize=T_CX,
                fontweight="bold")
        ax.text(xs[1] + cw / 2, y - hb * 0.72, sub, ha="center", va="center", fontsize=T_SUB)
        if n < len(etapas) - 1:
            ax.add_patch(FancyArrowPatch((xs[1] + cw / 2, y - hb), (xs[1] + cw / 2, y - hb - gap),
                                         arrowstyle="-|>", mutation_scale=16, linewidth=1.4,
                                         color="#111111", shrinkA=1, shrinkB=1))
        y -= hb + gap

    # 3. saída
    seg3d = cv2.cvtColor(cv2.imread(os.path.join(SAIDA, "segmentacao_3d.png")), cv2.COLOR_BGR2RGB)
    painel(xs[2], topo1, seg3d, "3D instances")
    painel(xs[2], topo2, cobertura(os.path.join(EXP, "experimento1", "padrao", "gat")),
           "Rendered 2D masks")

    # setas entre as colunas, na altura do meio dos painéis
    y_seta = (topo1 + topo2 - ih) / 2
    for i in range(2):
        ax.add_patch(FancyArrowPatch((xs[i] + cw + 0.07, y_seta), (xs[i + 1] - 0.07, y_seta),
                                     arrowstyle="-|>", mutation_scale=36, linewidth=4,
                                     color="#444444", shrinkA=0, shrinkB=0))

    caminho = os.path.join(SAIDA, "graphical_abstract.jpg")
    fig.savefig(caminho, dpi=300, facecolor="white", pil_kwargs={"quality": 95})
    plt.close(fig)
    print(f"  graphical_abstract.jpg: {L * 2.54:.0f} x {H * 2.54:.0f} cm, "
          f"{round(L * 300)} x {round(H * 300)} px, {len(gx)} nós e {len(arestas)} arestas no grafo")


# ----------------------------------------------------------------------------
# Diagramas (Figs. 1 e 2), desenhados em polegadas no tamanho de impressão
# ----------------------------------------------------------------------------

CINZA = dict(fc="#ededed", ec="#4d4d4d")
AZUL = dict(fc="#d9e6f7", ec="#2f6fc0")
LARANJA = dict(fc="#fae5cd", ec="#e08a2e")
VERDE = dict(fc="#dfeedd", ec="#3d7a3d")
ROXO = dict(fc="#e6dff3", ec="#7a5bbf")
BRANCO = dict(fc="#ffffff", ec="#111111")
LINHA = 0.8


def caixa(ax, x0, y0, x1, y1, titulo, linhas=(), cores=CINZA):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=0.04",
                                linewidth=LINHA, **cores))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h_tit, h_lin = T_TITULO * 1.3 / 72, T_TEXTO * 1.2 / 72
    total = (h_tit if titulo else 0) + len(linhas) * h_lin
    y = cy + total / 2
    if titulo:
        ax.text(cx, y - h_tit / 2, titulo, ha="center", va="center",
                fontsize=T_TITULO, fontweight="bold")
        y -= h_tit
    for linha in linhas:
        ax.text(cx, y - h_lin / 2, linha, ha="center", va="center", fontsize=T_TEXTO)
        y -= h_lin


def seta(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                 linewidth=LINHA, color="#111111", shrinkA=0, shrinkB=0))


def eixo_polegadas(largura, altura):
    fig = plt.figure(figsize=(largura, altura))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, largura); ax.set_ylim(0, altura); ax.axis("off")
    return fig, ax


def pipeline():
    L, H = PAGINA, 3.20
    fig, ax = eixo_polegadas(L, H)
    m = 0.03

    # Etapa 1
    ax.add_patch(FancyBboxPatch((m, 1.42), L - 2 * m, 1.75, boxstyle="round,pad=0,rounding_size=0.05",
                                fc="#f7f7f7", ec="#b8b8b8", linewidth=LINHA))
    ax.text(0.12, 3.01, "Stage 1: scene reconstruction and initial labels",
            fontsize=T_TITULO, fontweight="bold", va="center")
    hb = 0.54                     # altura das caixas
    y_sup, y_inf = 2.14, 1.51     # base das linhas de caixas
    caixa(ax, 0.12, y_inf, 0.92, y_sup + hb, "Scene", ["images"], CINZA)
    xs = [(1.12, 2.42), (2.62, 3.92)]
    caixa(ax, xs[0][0], y_sup, xs[0][1], y_sup + hb, "COLMAP (SfM)",
          ["camera poses and", "sparse point cloud"], AZUL)
    caixa(ax, xs[1][0], y_sup, xs[1][1], y_sup + hb, "3D splats",
          ["training with 3D", "Gaussian splatting"], AZUL)
    caixa(ax, xs[0][0], y_inf, xs[0][1], y_inf + hb, "SAM",
          ["2D segmentation", "masks"], LARANJA)
    caixa(ax, xs[1][0], y_inf, xs[1][1], y_inf + hb, "DEVA",
          ["consistent IDs", "across images"], LARANJA)
    caixa(ax, 4.12, y_inf, L - 0.12, y_sup + hb, r"2D $\rightarrow$ 3D projection",
          ["voting across views defines", "the initial label of each splat"], LARANJA)
    for yc in (y_sup + hb / 2, y_inf + hb / 2):
        seta(ax, 0.92, yc, 1.11, yc)
        seta(ax, 2.42, yc, 2.61, yc)
        seta(ax, 3.92, yc, 4.11, yc)

    # ligação entre as etapas
    xl = (4.12 + L - 0.12) / 2
    ax.plot([xl, xl], [y_inf, 1.36], color="#111111", linewidth=LINHA)
    ax.plot([xl, 0.07], [1.36, 1.36], color="#111111", linewidth=LINHA)
    ax.plot([0.07, 0.07], [1.36, 0.55], color="#111111", linewidth=LINHA)
    seta(ax, 0.07, 0.55, 0.11, 0.55)

    # Etapa 2
    ax.add_patch(FancyBboxPatch((m, 0.03), L - 2 * m, 1.27, boxstyle="round,pad=0,rounding_size=0.05",
                                fc="#f7f7f7", ec="#b8b8b8", linewidth=LINHA))
    ax.text(0.12, 1.14, "Stage 2: graph learning and segmentation",
            fontsize=T_TITULO, fontweight="bold", va="center")
    blocos = [
        ("Filtering", ["opacity, scale", "and outliers"], VERDE),
        ("Graph", ["KNN, distance", r"and color ($\Delta E$)"], VERDE),
        ("GAT or GCN", ["contrastive loss", r"$\rightarrow$ embeddings"], VERDE),
        ("HDBSCAN", ["clustering of", "the embeddings"], ROXO),
        ("Propagation", ["KNN for the", "noise splats"], ROXO),
        ("Segmentation", ["3D instances", "and 2D masks"], BRANCO),
    ]
    x, w, esp = 0.12, 0.99, 0.134
    for i, (titulo, linhas, cores) in enumerate(blocos):
        caixa(ax, x, 0.17, x + w, 0.93, titulo, linhas, cores)
        if i < len(blocos) - 1:
            seta(ax, x + w, 0.55, x + w + esp - 0.01, 0.55)
        x += w + esp
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(SAIDA, f"pipeline_metodo_en.{ext}"), dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"  pipeline_metodo_en.pdf/.png: {L * 25.4:.1f} x {H * 25.4:.1f} mm")


def arquiteturas():
    L, H = COLUNA, 4.14
    fig, ax = eixo_polegadas(L, H)
    wc = 1.44                         # largura das caixas
    ca, cb = 0.04 + wc / 2, L - 0.04 - wc / 2   # centros das colunas (a) e (b)
    xa0, xb0 = ca - wc / 2, cb - wc / 2

    ax.text(ca, H - 0.1, "(a) GAT with 4 heads", ha="center", va="center",
            fontsize=T_TITULO, fontweight="bold")
    # subtítulos em duas linhas: em 10 pt, uma linha só passa da largura de meia coluna
    ax.text(ca, H - 0.35, "learned attention\ncoefficients", ha="center", va="center",
            fontsize=T_TEXTO, style="italic", color="#333333", linespacing=1.1)
    ax.text(cb, H - 0.1, "(b) GCN with 2 layers", ha="center", va="center",
            fontsize=T_TITULO, fontweight="bold")
    ax.text(cb, H - 0.35, "aggregation weighted\n" + r"by $w_{ij}$", ha="center", va="center",
            fontsize=T_TEXTO, style="italic", color="#333333", linespacing=1.1)

    # (topo, base) das caixas
    y_in, y_c1, y_c2, y_lin, y_l2 = (3.58, 3.22), (3.04, 2.34), (2.16, 1.46), (0.96, 0.60), (0.42, 0.04)
    for x0, cor, nome, sub1, sub2, norm in [
        (xa0, LARANJA, "GATConv (4 heads)", r"$6 \rightarrow 32 \times 4 = 128$",
         r"$128 \rightarrow 64 \times 2 = 128$", ["BatchNorm, ELU,", "Dropout"]),
        (xb0, AZUL, "GCNConv", r"$6 \rightarrow 64$", r"$64 \rightarrow 128$", None),
    ]:
        caixa(ax, x0, y_in[1], x0 + wc, y_in[0], "Input", [r"$N \times 6$  (xyz + L*a*b*)"], CINZA)
        if norm:
            caixa(ax, x0, y_c1[1], x0 + wc, y_c1[0], nome, [sub1] + norm, cor)
            caixa(ax, x0, y_c2[1], x0 + wc, y_c2[0], "GATConv (2 heads)", [sub2] + norm, cor)
        else:
            caixa(ax, x0, y_c1[1], x0 + wc, y_c1[0], nome, [sub1, "LayerNorm, ELU,", "Dropout"], cor)
            caixa(ax, x0, y_c2[1], x0 + wc, y_c2[0], nome, [sub2, "LayerNorm"], cor)
        caixa(ax, x0, y_lin[1], x0 + wc, y_lin[0], "Linear", [r"$128 \rightarrow 32$"], VERDE)
        # \U0001D49B é o z em negrito itálico, como os vetores no texto do artigo (o mathtext
        # desta versão do matplotlib não tem \mathbfit)
        caixa(ax, x0, y_l2[1], x0 + wc, y_l2[0], "L2 normalization",
              ["embedding $\U0001D49B_i \\in \\mathbb{R}^{32}$"], CINZA)

    # coluna (a): setas diretas
    for y0, y1 in [(y_in[1], y_c1[0]), (y_c1[1], y_c2[0]), (y_c2[1], y_lin[0]), (y_lin[1], y_l2[0])]:
        seta(ax, ca, y0, ca, y1)

    # coluna (b): conexão residual entre as duas GCNConv
    y_ramo = (y_c1[1] + y_c2[0]) / 2
    y_soma = (y_c2[1] + y_lin[0]) / 2
    x_res = xb0 - 0.08                # deixa entre as colunas o espaço do rótulo em 10 pt
    r_soma = 0.07
    seta(ax, cb, y_in[1], cb, y_c1[0])
    seta(ax, cb, y_c1[1], cb, y_c2[0])
    ax.add_patch(Circle((cb, y_ramo), 0.022, fc="#111111", ec="#111111"))
    ax.plot([cb, x_res, x_res, cb - r_soma], [y_ramo, y_ramo, y_soma, y_soma],
            color="#111111", linewidth=LINHA)
    ax.text((xa0 + wc + x_res) / 2, (y_ramo + y_soma) / 2, r"Linear $64 \rightarrow 128$", rotation=90,
            ha="center", va="center", fontsize=T_TEXTO)
    seta(ax, cb, y_c2[1], cb, y_soma + r_soma)
    ax.add_patch(Circle((cb, y_soma), r_soma, fc="white", ec="#111111", linewidth=LINHA))
    ax.text(cb, y_soma, "+", ha="center", va="center", fontsize=T_TITULO)
    # rótulo à esquerda da seta que sai da soma (à direita não cabe na coluna)
    ax.text(cb - 0.05, (y_soma - r_soma + y_lin[0]) / 2, "ELU, Dropout", ha="right", va="center",
            fontsize=T_TEXTO)
    seta(ax, cb, y_soma - r_soma, cb, y_lin[0])
    seta(ax, cb, y_lin[1], cb, y_l2[0])

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(SAIDA, f"arquiteturas_gnn_en.{ext}"), dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"  arquiteturas_gnn_en.pdf/.png: {L * 25.4:.1f} x {H * 25.4:.1f} mm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", choices=["diagramas", "grades", "comparacao", "vistas", "seg3d",
                                     "resumo"], default=None)
    args = ap.parse_args()
    os.makedirs(SAIDA, exist_ok=True)
    if args.so in (None, "diagramas"):
        pipeline(); arquiteturas()
    if args.so in (None, "grades"):
        figuras_grades()
    if args.so in (None, "comparacao"):
        figuras_comparacao()
    if args.so in (None, "vistas"):
        figura_multivistas(); figura_consistencia()
    if args.so in (None, "seg3d"):
        segmentacao_3d()
    if args.so in (None, "resumo"):
        graphical_abstract()
