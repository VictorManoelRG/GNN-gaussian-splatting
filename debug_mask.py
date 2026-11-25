"""
Script de diagnóstico — rode isso ANTES de mexer mais na lógica de voto.
Objetivo: confirmar se a máscara do DEVA está sendo lida com os IDs corretos.

Uso:
    python debug_mask_encoding.py /caminho/masks/000001.png /caminho/deva.json 000001.jpg
"""
import sys
import json
import numpy as np
import cv2

mask_path = sys.argv[1]
deva_json_path = sys.argv[2]
frame_key = sys.argv[3]

# 1. Lê a máscara exatamente como o pipeline faz
mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
print(f"Arquivo: {mask_path}")
print(f"  shape: {mask.shape}, dtype: {mask.dtype}")

if mask.ndim == 3:
    print(f"  ⚠️ Máscara tem {mask.shape[2]} canais — verifique se o ID está")
    print(f"     codificado em mais de um canal (R + G*256 + B*65536, etc).")
    for c in range(mask.shape[2]):
        u = np.unique(mask[..., c])
        print(f"  canal {c}: {len(u)} valores únicos, min={u.min()}, max={u.max()}")
    mask_single_ch0 = mask[..., 0]
    print(f"  Se usarmos só o canal 0 -> {len(np.unique(mask_single_ch0))} IDs únicos")
else:
    mask_single_ch0 = mask

uniq_raw = np.unique(mask_single_ch0)
print(f"  IDs únicos na máscara (canal usado pelo pipeline): {len(uniq_raw)}")
print(f"  min={uniq_raw.min()}, max={uniq_raw.max()}")
if mask.dtype == np.uint8:
    print(f"  ⚠️ dtype é uint8 -> qualquer ID do DEVA acima de 255 seria truncado/wrapped!")

# 2. Compara com os IDs que o JSON do DEVA diz que deveriam existir nesse frame
with open(deva_json_path) as f:
    deva_data = json.load(f)

frame_ann = None
for ann in deva_data.get("annotations", []):
    if ann["file_name"] == frame_key:
        frame_ann = ann
        break

if frame_ann is None:
    print(f"  ⚠️ Frame '{frame_key}' não encontrado no JSON (confira o nome exato usado como chave)")
else:
    ids_json = sorted(seg["id"] for seg in frame_ann.get("segments_info", []))
    print(f"\n  IDs esperados pelo JSON para este frame: {ids_json}")
    print(f"  IDs realmente lidos da imagem PNG:        {sorted(int(x) for x in uniq_raw if x != 0)}")

    faltando = set(ids_json) - set(int(x) for x in uniq_raw)
    sobrando = set(int(x) for x in uniq_raw if x != 0) - set(ids_json)
    if faltando:
        print(f"  ❌ IDs no JSON que NÃO aparecem na imagem: {sorted(faltando)}")
        print(f"     -> forte indício de truncamento/perda de ID na leitura da máscara")
    if sobrando:
        print(f"  ⚠️ IDs na imagem que NÃO estão no JSON: {sorted(sobrando)}")
    if not faltando and not sobrando:
        print(f"  ✅ Os IDs batem perfeitamente entre JSON e imagem")