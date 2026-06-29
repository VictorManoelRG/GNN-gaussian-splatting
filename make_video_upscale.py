import os
import cv2
import numpy as np
import torch

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

import sys

if len(sys.argv) < 2:
    print("Uso: python3 make_video.py <nome_dataset>")
    sys.exit(1)

NOME = sys.argv[1]

# ==========================
# Configurações
# ==========================

DIR1 = f"/home/victor/Documentos/gaussian_grouping/gaussian-grouping/data/{NOME}/images"
DIR2 = f"/home/victor/Documentos/gaussian_grouping/gaussian-grouping/GAT/output_seg-{NOME}/gat"

OUTPUT_VIDEO = f"comparacao_{NOME}.mp4"

TARGET_W = 1920
TARGET_H = 1080

FPS = 30
FRAME_TIME = 0.3
FRAMES_REPEAT = int(FPS * FRAME_TIME)

EXTENSOES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ==========================
# Device
# ==========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================
# Modelo Real-ESRGAN correto
# ==========================

model_path = "weights/RealESRGAN_x4plus.pth"

rrdb_model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=4
)

upsampler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=rrdb_model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=torch.cuda.is_available(),
    device=device
)


# ==========================
# Utils
# ==========================

def listar_imagens_dict(diretorio):
    arquivos = {}
    for f in os.listdir(diretorio):
        if f.lower().endswith(EXTENSOES):
            nome_base = os.path.splitext(f)[0]
            arquivos[nome_base] = os.path.join(diretorio, f)
    return arquivos


def upscale_ai(img):
    """Upscale com Real-ESRGAN"""
    output, _ = upsampler.enhance(img, outscale=4)
    return output


def encaixar_no_frame(img, largura, altura):
    """Mantém proporção e centraliza no canvas"""
    h, w = img.shape[:2]

    escala = min(largura / w, altura / h)
    novo_w = int(w * escala)
    novo_h = int(h * escala)

    img_resized = cv2.resize(
        img,
        (novo_w, novo_h),
        interpolation=cv2.INTER_AREA
    )

    frame = np.zeros((altura, largura, 3), dtype=np.uint8)

    y_offset = (altura - novo_h) // 2
    x_offset = (largura - novo_w) // 2

    frame[y_offset:y_offset + novo_h, x_offset:x_offset + novo_w] = img_resized

    return frame


# ==========================
# Carregar imagens
# ==========================

dict1 = listar_imagens_dict(DIR1)
dict2 = listar_imagens_dict(DIR2)

common_keys = sorted(set(dict1.keys()) & set(dict2.keys()))

if len(common_keys) == 0:
    raise RuntimeError("Nenhum frame em comum entre os diretórios.")

print(f"Frames em comum: {len(common_keys)}")


# ==========================
# Video writer
# ==========================

half_w = TARGET_W // 2

video = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS,
    (TARGET_W, TARGET_H)
)


# ==========================
# Loop principal
# ==========================

for i, key in enumerate(common_keys):

    img1 = cv2.imread(dict1[key])
    img2 = cv2.imread(dict2[key])

    if img1 is None or img2 is None:
        print(f"Frame {key} inválido, pulando...")
        continue

    # ==========================
    # IA UPSCALE (4x)
    # ==========================
    img1 = upscale_ai(img1)
    img2 = upscale_ai(img2)

    # ==========================
    # FIT no canvas
    # ==========================
    img1 = encaixar_no_frame(img1, half_w, TARGET_H)
    img2 = encaixar_no_frame(img2, half_w, TARGET_H)

    frame = np.hstack((img1, img2))

    # ==========================
    # duração de 0.8s por frame
    # ==========================
    for _ in range(FRAMES_REPEAT):
        video.write(frame)

    print(f"{i+1}/{len(common_keys)} - {key}")


video.release()

print(f"\nVídeo salvo em: {OUTPUT_VIDEO}")