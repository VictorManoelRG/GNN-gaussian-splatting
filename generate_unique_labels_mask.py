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
):
    bgr = cv2.imread(in_path)
    if bgr is None:
        raise FileNotFoundError(in_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # 0. suaviza levemente preservando bordas, para tirar ruído de
    #    anti-aliasing / speckle de pixel isolado antes de agrupar por cor
    rgb_smooth = cv2.bilateralFilter(rgb, d=5, sigmaColor=40, sigmaSpace=5)

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

    # 2b. filtro de voto majoritário espacial: cada pixel passa a ter o
    #     rótulo de cor mais comum na sua vizinhança. Isso é essencial aqui
    #     porque a variação de cor pixel-a-pixel do Gaussian Splatting não é
    #     ruído tipo "sal e pimenta" isolado, e sim correlacionado
    #     espacialmente em pequenas manchas -- abertura/fechamento morfológico
    #     simples não dá conta disso, mas contagem local por vizinhança sim.
    if majority_radius > 0:
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
    ap.add_argument("--connectivity", type=int, default=8, choices=[4, 8])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug-clusters", default=None,
                     help="Se passado, salva também uma imagem mostrando só os clusters de cor (antes de separar por conectividade).")
    ap.add_argument("--majority-radius", type=int, default=4,
                     help="Raio (px) do filtro de voto majoritário aplicado nos rótulos de cor. 0 desliga.")
    ap.add_argument("--majority-iters", type=int, default=2,
                     help="Quantas vezes repetir o filtro de voto majoritário.")
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
    )