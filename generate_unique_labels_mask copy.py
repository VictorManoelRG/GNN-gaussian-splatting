import argparse
import numpy as np
import cv2
from sklearn.cluster import KMeans
from scipy import ndimage


def build_distinct_colors(n, seed=0):
    """
    Gera cores RGB distintas para representar as instâncias.
    """
    rng = np.random.default_rng(seed)

    hues = np.linspace(0, 179, n, endpoint=False)
    rng.shuffle(hues)

    hsv = np.zeros((n, 1, 3), dtype=np.uint8)

    hsv[:, 0, 0] = hues.astype(np.uint8)
    hsv[:, 0, 1] = rng.integers(180, 256, n)
    hsv[:, 0, 2] = rng.integers(180, 256, n)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb[:, 0]


def segment(
    input_path,
    output_path,
    n_colors=12,
    color_threshold=18,
    min_area=30,
    seed=0,
):
    # =========================================================
    # 1. CARREGAR
    # =========================================================

    bgr = cv2.imread(input_path)

    if bgr is None:
        raise FileNotFoundError(input_path)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    h, w = rgb.shape[:2]

    # =========================================================
    # 2. RGB -> LAB
    # =========================================================

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    lab_float = lab.astype(np.float32)

    pixels = lab_float.reshape(-1, 3)

    # =========================================================
    # 3. DESCOBRIR CORES REPRESENTATIVAS
    # =========================================================

    print("Descobrindo cores predominantes...")

    # Amostra para acelerar KMeans
    max_samples = 10000

    if len(pixels) > max_samples:

        rng = np.random.default_rng(seed)

        indices = rng.choice(
            len(pixels),
            max_samples,
            replace=False
        )

        sample = pixels[indices]

    else:
        sample = pixels

    kmeans = KMeans(
        n_clusters=n_colors,
        random_state=seed,
        n_init=10
    )

    kmeans.fit(sample)

    centers = kmeans.cluster_centers_

    # =========================================================
    # 4. CADA PIXEL → COR BASE MAIS PRÓXIMA
    # =========================================================

    # distância Lab para cada centro
    distances = np.linalg.norm(
        pixels[:, None, :] - centers[None, :, :],
        axis=2
    )

    nearest_cluster = distances.argmin(axis=1)

    nearest_distance = distances.min(axis=1)

    cluster_map = nearest_cluster.reshape(h, w)

    distance_map = nearest_distance.reshape(h, w)

    # =========================================================
    # 5. CRIAR INSTÂNCIAS
    # =========================================================

    instance_map = np.zeros(
        (h, w),
        dtype=np.int32
    )

    next_instance = 1

    instance_data = []

    # =========================================================
    # 6. PROCESSAR CADA FAMÍLIA DE COR
    # =========================================================

    for color_id in range(n_colors):

        # pixels próximos desta cor
        mask = (
            (cluster_map == color_id) &
            (distance_map <= color_threshold)
        )

        if not mask.any():
            continue

        # -----------------------------------------------------
        # IMPORTANTÍSSIMO:
        #
        # não fazemos majority filter
        # não fazemos fill
        # não fazemos dilation
        # -----------------------------------------------------

        labeled, n_components = ndimage.label(
            mask,
            structure=np.ones((3, 3), dtype=np.uint8)
        )

        for component_id in range(
            1,
            n_components + 1
        ):

            component = (
                labeled == component_id
            )

            area = int(component.sum())

            # remove apenas coisas realmente minúsculas
            if area < min_area:
                continue

            instance_map[component] = next_instance

            mean_color = rgb[component].mean(axis=0)

            instance_data.append({
                "id": next_instance,
                "color_cluster": color_id,
                "area": area,
                "mean_rgb": mean_color,
            })

            next_instance += 1

    # =========================================================
    # 7. CORES RGB ÚNICAS
    # =========================================================

    n_instances = next_instance - 1

    colors = build_distinct_colors(
        n_instances,
        seed=seed
    )

    output = np.zeros(
        (h, w, 3),
        dtype=np.uint8
    )

    for i in range(1, next_instance):

        output[
            instance_map == i
        ] = colors[i - 1]

    # =========================================================
    # 8. SALVAR
    # =========================================================

    cv2.imwrite(
        output_path,
        cv2.cvtColor(
            output,
            cv2.COLOR_RGB2BGR
        )
    )

    # =========================================================
    # 9. LOG
    # =========================================================

    print()
    print("====================================")
    print("RESULTADO")
    print("====================================")

    print(
        f"clusters de cor: {n_colors}"
    )

    print(
        f"instâncias: {n_instances}"
    )

    for data in instance_data:

        print(
            f"id={data['id']:3d} "
            f"area={data['area']:6d} "
            f"cluster={data['color_cluster']:3d} "
            f"mean_rgb={data['mean_rgb'].astype(int)}"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input"
    )

    parser.add_argument(
        "output"
    )

    parser.add_argument(
        "--n-colors",
        type=int,
        default=12
    )

    parser.add_argument(
        "--color-threshold",
        type=float,
        default=18
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=30
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0
    )

    args = parser.parse_args()

    segment(
        args.input,
        args.output,
        n_colors=args.n_colors,
        color_threshold=args.color_threshold,
        min_area=args.min_area,
        seed=args.seed,
    )