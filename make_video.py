import os
import cv2
import numpy as np

# ==========================
# Configurações
# ==========================

DIR1 = "/home/victor/Documentos/gaussian_grouping/gaussian-grouping/data/ramen/images"
DIR2 = "/home/victor/Documentos/gaussian_grouping/gaussian-grouping/GAT/output_seg-ramen/gat"

OUTPUT_VIDEO = "comparacao.mp4"
FPS = 4

EXTENSOES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

TARGET_W = 1920
TARGET_H = 1080


# ==========================
# Utils
# ==========================

def listar_imagens_dict(diretorio):
    arquivos = {}
    for f in os.listdir(diretorio):
        if f.lower().endswith(EXTENSOES):
            nome_base = os.path.splitext(f)[0]  # <<< chave importante
            arquivos[nome_base] = os.path.join(diretorio, f)
    return arquivos


def encaixar_no_frame(img, largura, altura):
    h, w = img.shape[:2]

    escala = min(largura / w, altura / h)
    novo_w = int(w * escala)
    novo_h = int(h * escala)

    img_resized = cv2.resize(
        img,
        (novo_w, novo_h),
        interpolation=cv2.INTER_LANCZOS4
    )

    frame = np.zeros((altura, largura, 3), dtype=np.uint8)

    y_offset = (altura - novo_h) // 2
    x_offset = (largura - novo_w) // 2

    frame[y_offset:y_offset + novo_h, x_offset:x_offset + novo_w] = img_resized

    return frame


# ==========================
# Carregar como dicionário
# ==========================

dict1 = listar_imagens_dict(DIR1)
dict2 = listar_imagens_dict(DIR2)

# interseção dos nomes (só frames que existem nos dois)
common_keys = sorted(set(dict1.keys()) & set(dict2.keys()))

if len(common_keys) == 0:
    raise RuntimeError("Nenhum frame em comum entre os diretórios.")

print(f"Frames em comum: {len(common_keys)}")

# ==========================
# Vídeo
# ==========================

half_w = TARGET_W // 2

video = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS,
    (TARGET_W, TARGET_H)
)


# ==========================
# Loop robusto
# ==========================

for i, key in enumerate(common_keys):

    img1 = cv2.imread(dict1[key])
    img2 = cv2.imread(dict2[key])

    if img1 is None or img2 is None:
        print(f"Frame {key} inválido, pulando...")
        continue

    img1 = encaixar_no_frame(img1, half_w, TARGET_H)
    img2 = encaixar_no_frame(img2, half_w, TARGET_H)

    frame = np.hstack((img1, img2))
    video.write(frame)

    print(f"{i+1}/{len(common_keys)} - {key}")


video.release()

print(f"\nVídeo salvo em: {OUTPUT_VIDEO}")