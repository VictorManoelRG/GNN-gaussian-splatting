import cv2
import numpy as np

mask = cv2.imread("/home/victor/Documentos/gaussian_grouping/gaussian-grouping/Tracking-Anything-with-DEVA/example/output_gaussian_dataset/ramen/Annotations_color/frame_00002.png", cv2.IMREAD_UNCHANGED)
print("shape:", mask.shape, "dtype:", mask.dtype)

if mask.ndim == 3:
    # cores únicas de verdade (R,G,B combinados)
    flat = mask.reshape(-1, mask.shape[-1])
    unique_colors = np.unique(flat, axis=0)
    print("nº de cores únicas (RGB completo):", len(unique_colors))

    # o que seu código atual está usando (só canal 0)
    unique_ch0 = np.unique(mask[..., 0])
    print("nº de valores únicos (só canal 0):", len(unique_ch0))