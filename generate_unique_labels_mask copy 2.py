"""
Reagrupa uma imagem de segmentação de instâncias (ex: renderizada via Gaussian
Splatting, onde a cor de uma mesma instância varia pixel a pixel por causa da
soma ponderada de contribuições) em máscaras de instância "limpas", cada uma
com uma única cor RGB nova.

Pipeline:
  1. Remove fundo (pixels muito escuros).
  2. Agrupa cores parecidas em clusters usando MeanShift no espaço Lab
     (perceptualmente mais uniforme que RGB) -- isso une as variações de tom
     de uma mesma instância (ex: os vários "roxos" do urso) num único cluster,
     sem precisar dizer a priori quantas instâncias existem.
  3. Para cada cluster de cor, roda connected components (8-conectividade).
     Se duas regiões têm cor parecida mas NÃO se tocam (ex: prato e
     guardanapo, ambos verdes), elas viram duas instâncias diferentes.
  4. Descarta componentes minúsculos (ruído / anti-aliasing) e opcionalmente
     funde neles ao componente vizinho maior.
  5. Sorteia uma cor RGB única e visualmente distinta para cada instância
     final e salva a nova imagem de segmentação.

Uso:
    python3 segment_instances.py entrada.png saida.png \
        --bandwidth 18 --min-area 40 --bg-thresh 12
"""

import argparse
import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy import ndimage

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels
    HAS_DENSECRF = True
except ImportError:
    HAS_DENSECRF = False


def build_distinct_colors(n, seed=0):
    """Gera n cores RGB bem distintas usando HSV espalhado + shuffle."""
    rng = np.random.default_rng(seed)
    hues = np.linspace(0, 179, n, endpoint=False).astype(np.uint8)
    rng.shuffle(hues)
    sats = rng.integers(170, 256, size=n, dtype=np.uint8)
    vals = rng.integers(170, 256, size=n, dtype=np.uint8)
    hsv = np.stack([hues, sats, vals], axis=1).reshape(1, n, 3).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).reshape(n, 3)
    return rgb


def refine_labels_densecrf(color_label_map, fg_mask, rgb, n_color_clusters,
                            crf_iters=5, gauss_sxy=3, gauss_compat=3,
                            bilateral_sxy=40, bilateral_srgb=13, bilateral_compat=8,
                            gt_prob=0.7):
    """
    Refina os rótulos de cor (por pixel) com um DenseCRF (Krähenbühl & Koltun,
    2011 -- o mesmo usado como pós-processamento no DeepLab).

    Diferença em relação ao filtro de voto majoritário: em vez de olhar só
    uma janela local fixa, o DenseCRF usa um kernel bilateral sobre a imagem
    inteira (posição + cor), então ele une variações de tom mesmo quando
    espalhadas por uma região grande, e ao mesmo tempo respeita bordas reais
    (onde a cor muda bruscamente o termo bilateral cai, então ele não "vaza"
    de um objeto pro outro).

    O fundo entra como uma classe extra no CRF (n_color_clusters), com alta
    confiança onde já foi detectado como fundo -- isso deixa o próprio CRF
    ajustar um pouco a fronteira objeto/fundo em vez de usar a máscara de
    fundo binária "crua".
    """
    if not HAS_DENSECRF:
        raise RuntimeError(
            "pydensecrf não está instalado. "
            "pip install git+https://github.com/lucasb-eyer/pydensecrf.git --break-system-packages"
        )
    h, w = color_label_map.shape
    n_labels = n_color_clusters + 1  # + classe "fundo"

    labels = np.where(fg_mask, color_label_map, n_color_clusters).astype(np.int64)

    unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)

    d = dcrf.DenseCRF2D(w, h, n_labels)
    d.setUnaryEnergy(unary)

    # termo espacial "puro" (suavidade local independente de cor)
    d.addPairwiseGaussian(sxy=gauss_sxy, compat=gauss_compat)

    # termo bilateral: funde pixels próximos em posição E em cor -- é aqui
    # que a variação de tom do Gaussian Splatting é unida de forma
    # edge-aware
    d.addPairwiseBilateral(
        sxy=(bilateral_sxy, bilateral_sxy),
        srgb=(bilateral_srgb, bilateral_srgb, bilateral_srgb),
        rgbim=np.ascontiguousarray(rgb),
        compat=bilateral_compat,
    )

    Q = d.inference(crf_iters)
    map_labels = np.argmax(Q, axis=0).reshape(h, w)

    refined = np.where(map_labels == n_color_clusters, -1, map_labels)
    return refined


def segment(
    in_path,
    out_path,
    bandwidth=None,
    min_area=100,
    bg_thresh=12,
    connectivity=8,
    merge_small_into_neighbor=True,
    seed=0,
    save_debug_clusters=None,
    majority_radius=4,
    majority_iters=2,
    refine="majority",  # "majority" | "crf" | "none"
    crf_iters=10,
    crf_gauss_sxy=3,
    crf_gauss_compat=15,
    crf_bilateral_sxy=15,
    crf_bilateral_srgb=60,
    crf_bilateral_compat=0,
    crf_gt_prob=0.4,
    close_bg_holes_max_area=500,
    median_blur_ksize=15,
    median_blur_iters=3,
):
    bgr = cv2.imread(in_path)
    if bgr is None:
        raise FileNotFoundError(in_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # 0. suaviza fortemente ANTES de agrupar por cor, usando um filtro de
    #    mediana (não gaussiano/bilateral): mediana pega o valor mais comum
    #    ("moda") da vizinhança, sem inventar cor nova por média -- é o jeito
    #    certo de atacar o ruído pixel-a-pixel do Gaussian Splatting na
    #    origem, em vez de tentar limpar rótulos de cluster depois. Isso
    #    reduz drasticamente quantos clusters de cor espúrios aparecem.
    rgb_smooth = rgb.copy()
    for _ in range(median_blur_iters):
        rgb_smooth = cv2.medianBlur(rgb_smooth, median_blur_ksize)

    # 1. máscara de fundo (pixels quase pretos)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    fg_mask = gray > bg_thresh

    # 2. clustering de cor no espaço Lab, apenas nos pixels de foreground
    lab = cv2.cvtColor(rgb_smooth, cv2.COLOR_RGB2LAB).astype(np.float32)
    fg_pixels = lab[fg_mask]

    if bandwidth is None:
        # estima automaticamente a partir de uma amostra
        sample = fg_pixels[:: max(1, len(fg_pixels) // 3000)]
        bandwidth = estimate_bandwidth(sample, quantile=0.1, n_samples=min(2000, len(sample)))
        if bandwidth <= 0:
            bandwidth = 15.0

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
    color_labels_fg = ms.fit_predict(fg_pixels)

    color_label_map = np.full((h, w), -1, dtype=np.int32)
    color_label_map[fg_mask] = color_labels_fg
    n_color_clusters = color_labels_fg.max() + 1

    # 2b. refinamento espacial dos rótulos de cor
    if refine == "majority" and majority_radius > 0:
        ksize = 2 * majority_radius + 1
        for _ in range(majority_iters):
            counts = np.zeros((n_color_clusters, h, w), dtype=np.float32)
            for c in range(n_color_clusters):
                counts[c] = cv2.boxFilter(
                    (color_label_map == c).astype(np.float32), ddepth=-1,
                    ksize=(ksize, ksize), normalize=False,
                )
            majority_label = counts.argmax(axis=0)
            color_label_map = np.where(fg_mask, majority_label, -1)
    elif refine == "crf":
        color_label_map = refine_labels_densecrf(
            color_label_map, fg_mask, rgb, n_color_clusters,
            crf_iters=crf_iters,
            gauss_sxy=crf_gauss_sxy, gauss_compat=crf_gauss_compat,
            bilateral_sxy=crf_bilateral_sxy, bilateral_srgb=crf_bilateral_srgb,
            bilateral_compat=crf_bilateral_compat, gt_prob=crf_gt_prob,
        )
        fg_mask = color_label_map >= 0  # o CRF pode reclassificar pixels como fundo

    if save_debug_clusters:
        dbg = np.zeros((h, w, 3), dtype=np.uint8)
        cluster_colors = build_distinct_colors(n_color_clusters, seed=seed)
        for c in range(n_color_clusters):
            dbg[color_label_map == c] = cluster_colors[c]
        cv2.imwrite(save_debug_clusters, cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR))

    # 3. dentro de cada cluster de cor, separa por componentes conectados
    struct = ndimage.generate_binary_structure(2, 2 if connectivity == 8 else 1)
    instance_map = np.zeros((h, w), dtype=np.int32)  # 0 = fundo
    next_id = 1
    instance_info = []  # (id, mean_color_rgb, area, color_cluster_id)

    clean_kernel = np.ones((3, 3), np.uint8)
    for c in range(n_color_clusters):
        mask_c = (color_label_map == c).astype(np.uint8)
        if not mask_c.any():
            continue
        # remove speckles isolados (abertura) e fecha pequenos buracos/gaps
        # de anti-aliasing na borda (fechamento), sem "vazar" para outro
        # cluster de cor
        mask_c = cv2.morphologyEx(mask_c, cv2.MORPH_OPEN, clean_kernel)
        mask_c = cv2.morphologyEx(mask_c, cv2.MORPH_CLOSE, clean_kernel)
        mask_c = mask_c.astype(bool)
        if not mask_c.any():
            continue
        labeled, n_comp = ndimage.label(mask_c, structure=struct)
        for comp_id in range(1, n_comp + 1):
            comp_mask = labeled == comp_id
            instance_map[comp_mask] = next_id
            mean_color = rgb[comp_mask].mean(axis=0)
            instance_info.append([next_id, mean_color, comp_mask.sum(), c])
            next_id += 1

    # 3b. preenche buracos deixados pela limpeza morfológica: qualquer pixel
    #     de foreground que ficou sem rótulo herda o rótulo do pixel
    #     rotulado mais próximo (nearest-neighbor fill)
    unlabeled_fg = fg_mask & (instance_map == 0)
    if unlabeled_fg.any() and (instance_map > 0).any():
        _, (iy, ix) = ndimage.distance_transform_edt(
            instance_map == 0, return_indices=True
        )
        filled = instance_map[iy, ix]
        instance_map[unlabeled_fg] = filled[unlabeled_fg]

    # 4. descarta / funde componentes minúsculos
    final_info = []
    for inst_id, mean_color, area, c in instance_info:
        if area < min_area:
            comp_mask = instance_map == inst_id
            if merge_small_into_neighbor:
                # funde no vizinho de maior área que faz fronteira com ele
                dil = ndimage.binary_dilation(comp_mask, structure=struct)
                border = dil & ~comp_mask
                neighbor_ids = instance_map[border]
                neighbor_ids = neighbor_ids[neighbor_ids > 0]
                if len(neighbor_ids) > 0:
                    vals, counts = np.unique(neighbor_ids, return_counts=True)
                    target = vals[np.argmax(counts)]
                    instance_map[comp_mask] = target
                else:
                    instance_map[comp_mask] = 0
            else:
                instance_map[comp_mask] = 0
        else:
            final_info.append([inst_id, mean_color, area, c])

    # 4b. fecha buracos internos de fundo (pixels pretos "vazando" por dentro
    #     de um objeto, comum em renders de Gaussian Splatting onde há
    #     frestas entre splats). Só preenche buraco de FUNDO cercado por uma
    #     única instância -- ilhas de OUTRA cor em primeiro plano (ex: o
    #     nariz do urso, cercado pelo corpo) não são tocadas, porque não são
    #     "buraco de fundo", são outro cluster de cor.
    if close_bg_holes_max_area > 0:
        final_ids_sorted_by_area = sorted(
            final_info, key=lambda x: -x[2]  # maior área primeiro
        )
        for inst_id, _, _, _ in final_ids_sorted_by_area:
            mask = instance_map == inst_id
            filled = ndimage.binary_fill_holes(mask)
            holes = filled & ~mask
            if not holes.any():
                continue
            hole_labels, n_holes = ndimage.label(holes, structure=struct)
            for hole_id in range(1, n_holes + 1):
                hole_mask = hole_labels == hole_id
                # só fecha se o buraco inteiro for região de fundo (não outro
                # cluster de cor já rotulado como instância própria)
                if (instance_map[hole_mask] == 0).all() and hole_mask.sum() <= close_bg_holes_max_area:
                    instance_map[hole_mask] = inst_id

    # 5. sorteia cores novas e únicas por instância final
    final_ids = sorted(set(instance_map[instance_map > 0].tolist()))
    n_final = len(final_ids)
    new_colors = build_distinct_colors(n_final, seed=seed)

    out_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    id_to_color = {}
    for idx, inst_id in enumerate(final_ids):
        color = new_colors[idx]
        id_to_color[inst_id] = tuple(int(v) for v in color)
        out_rgb[instance_map == inst_id] = color

    cv2.imwrite(out_path, cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))

    print(f"bandwidth usado: {bandwidth:.2f}")
    print(f"clusters de cor encontrados: {n_color_clusters}")
    print(f"instâncias finais (após separar por conectividade e filtrar ruído < {min_area}px): {n_final}")
    for inst_id in final_ids:
        area = int((instance_map == inst_id).sum())
        print(f"  id={inst_id:3d}  area={area:6d}px  nova_cor_rgb={id_to_color[inst_id]}")

    return instance_map, id_to_color


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--bandwidth", type=float, default=None,
                     help="Raio de agrupamento de cor no espaço Lab. Maior = funde mais variações. "
                          "Se omitido, é estimado automaticamente.")
    ap.add_argument("--min-area", type=int, default=100,
                     help="Componentes menores que isso (em pixels) são tratados como ruído.")
    ap.add_argument("--bg-thresh", type=int, default=12,
                     help="Limiar de brilho abaixo do qual o pixel é considerado fundo.")
    ap.add_argument("--connectivity", type=int, default=4, choices=[4, 8])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug-clusters", default=None,
                     help="Se passado, salva também uma imagem mostrando só os clusters de cor (antes de separar por conectividade).")
    ap.add_argument("--majority-radius", type=int, default=4,
                     help="Raio (px) do filtro de voto majoritário aplicado nos rótulos de cor. 0 desliga.")
    ap.add_argument("--majority-iters", type=int, default=2,
                     help="Quantas vezes repetir o filtro de voto majoritário.")
    ap.add_argument("--refine", choices=["majority", "crf", "none"], default="majority",
                     help="Método de refinamento espacial dos rótulos de cor antes de separar por conectividade.")
    ap.add_argument("--crf-iters", type=int, default=10)
    ap.add_argument("--crf-gauss-sxy", type=float, default=3)
    ap.add_argument("--crf-gauss-compat", type=float, default=15)
    ap.add_argument("--crf-bilateral-sxy", type=float, default=15,
                     help="Raio espacial (px) do kernel bilateral do CRF. Maior = une variações de tom mais espalhadas.")
    ap.add_argument("--crf-bilateral-srgb", type=float, default=60,
                     help="Tolerância de cor (0-255) do kernel bilateral do CRF. Maior = funde tons mais diferentes.")
    ap.add_argument("--crf-bilateral-compat", type=float, default=0,
                     help="0 desliga o termo de cor do CRF (recomendado p/ renders muito ruidosos -- ver nota no topo do arquivo).")
    ap.add_argument("--crf-gt-prob", type=float, default=0.4,
                     help="Confiança inicial no rótulo do MeanShift (0-1). Mais baixo = CRF pode mudar mais.")
    ap.add_argument("--close-bg-holes-max-area", type=int, default=500,
                     help="Fecha buracos de fundo (px) cercados por uma única instância, até esse tamanho. 0 desliga.")
    ap.add_argument("--median-blur-ksize", type=int, default=15,
                     help="Tamanho (ímpar) do filtro de mediana aplicado ANTES do clustering de cor. Maior = funde variações de tom mais fortes.")
    ap.add_argument("--median-blur-iters", type=int, default=1,
                     help="Quantas vezes repetir o filtro de mediana.")
    args = ap.parse_args()

    segment(
        args.input, args.output,
        bandwidth=args.bandwidth,
        min_area=args.min_area,
        bg_thresh=args.bg_thresh,
        connectivity=args.connectivity,
        seed=args.seed,
        save_debug_clusters=args.debug_clusters,
        majority_radius=args.majority_radius,
        majority_iters=args.majority_iters,
        refine=args.refine,
        crf_iters=args.crf_iters,
        crf_gauss_sxy=args.crf_gauss_sxy,
        crf_gauss_compat=args.crf_gauss_compat,
        crf_bilateral_sxy=args.crf_bilateral_sxy,
        crf_bilateral_srgb=args.crf_bilateral_srgb,
        crf_bilateral_compat=args.crf_bilateral_compat,
        crf_gt_prob=args.crf_gt_prob,
        close_bg_holes_max_area=args.close_bg_holes_max_area,
        median_blur_ksize=args.median_blur_ksize,
        median_blur_iters=args.median_blur_iters,
    )