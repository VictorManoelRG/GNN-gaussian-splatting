#!/usr/bin/env bash
# ============================================================================
# Pipeline completo de preparação + treino de uma cena nova.
#
#   1. COLMAP (convert.py)      -> poses de câmera e nuvem esparsa
#   2. resize                   -> imagens pequenas o bastante para SAM/DEVA
#   3. prepare_pseudo_label.sh  -> máscaras do SAM rastreadas pelo DEVA
#   4. train.sh                 -> Gaussian Grouping + render
#
# Precisa ser chamado através do run_colmap_env.sh, que limpa as variáveis do
# VSCode snap (ver o cabeçalho daquele arquivo) e mantém o conda
# gaussian_grouping:
#
#   ./run_colmap_env.sh ./pipeline_cena.sh bonsai
#
# Cada etapa só roda se ainda não tiver sido concluída, então dá para
# reexecutar o script depois de uma interrupção sem refazer o que já passou.
# ============================================================================
set -u -o pipefail

CENA="${1:?uso: pipeline_cena.sh <nome-da-cena> [fator-de-resize]}"
# Fator de redução das imagens. As cenas do LERF usadas na tese ficaram em
# torno de 182 px de altura; com bonsai e room saindo do COLMAP em ~2078 px,
# o fator 9 leva a ~231 px, um pouco acima delas mas ainda leve o bastante
# para o SAM e o DEVA caberem nos 8 GB da placa.
FATOR="${2:-9}"
RAIZ="/home/victor/Documentos/gaussian_grouping/gaussian-grouping"
DADOS="${RAIZ}/data/${CENA}"
LOGS="${RAIZ}/logs_cenas"

mkdir -p "${LOGS}"
cd "${RAIZ}" || exit 1

etapa () { echo -e "\n=========== [$CENA] $* ==========="; }

# ---------------------------------------------------------------- 1. COLMAP
if [ -f "${DADOS}/sparse/0/cameras.bin" ]; then
  etapa "1/4 COLMAP já feito, pulando"
else
  etapa "1/4 COLMAP (convert.py)"
  python convert.py -s "data/${CENA}" 2>&1 | tee "${LOGS}/${CENA}_1_convert.log"
  [ -f "${DADOS}/sparse/0/cameras.bin" ] || { echo "❌ COLMAP não produziu sparse/0"; exit 1; }
  # O banco de features não é usado no treino e ocupa vários GB.
  rm -f "${DADOS}/distorted/database.db"
fi

# ---------------------------------------------------------------- 2. resize
if [ -d "${DADOS}/images_full" ]; then
  etapa "2/4 resize já feito, pulando"
else
  etapa "2/4 resize das imagens (fator ${FATOR})"
  python resize_images.py "${DADOS}/images" "${FATOR}" 2>&1 | tail -3
  SAIDA="${DADOS}/images/resized_x${FATOR}"
  [ -d "$SAIDA" ] || { echo "❌ resize não gerou ${SAIDA}"; exit 1; }
  # As reduzidas passam a ser `images` (é o que o DEVA e o train.py leem) e as
  # originais sem distorção ficam guardadas em images_full.
  mv "$SAIDA" "${DADOS}/images_resized_tmp"
  mv "${DADOS}/images" "${DADOS}/images_full"
  mv "${DADOS}/images_resized_tmp" "${DADOS}/images"
  python -c "
import cv2, glob
fs = sorted(glob.glob('${DADOS}/images/*'))
im = cv2.imread(fs[0])
print(f'   {len(fs)} imagens em {im.shape[1]}x{im.shape[0]}')
"
fi

# ------------------------------------------------------ 3. SAM + DEVA
if [ -n "$(ls -A "${DADOS}/object_mask"/*.png 2>/dev/null)" ]; then
  etapa "3/4 máscaras já geradas, pulando"
else
  etapa "3/4 pseudo-labels (SAM + DEVA)"
  bash script/prepare_pseudo_label.sh "${CENA}" 1 2>&1 | tee "${LOGS}/${CENA}_3_deva.log" | tail -5
  # O script copia para object_mask/Annotations/, mas o dataset_readers.py
  # procura object_mask/<nome-da-imagem>.png, um nível acima.
  if [ -d "${DADOS}/object_mask/Annotations" ]; then
    mv "${DADOS}/object_mask/Annotations"/*.png "${DADOS}/object_mask"/ 2>/dev/null
    rmdir "${DADOS}/object_mask/Annotations" 2>/dev/null
  fi
  N=$(ls -1 "${DADOS}/object_mask"/*.png 2>/dev/null | wc -l)
  echo "   ${N} máscaras em object_mask/"
  [ "$N" -gt 0 ] || { echo "❌ nenhuma máscara gerada"; exit 1; }
fi

# ---------------------------------------------------------------- 4. treino
if [ -d "${RAIZ}/output/${CENA}/point_cloud/iteration_30000" ]; then
  etapa "4/4 treino já concluído, pulando"
else
  etapa "4/4 treino do Gaussian Grouping"
  # Cenas do MipNeRF-360 entram com muito mais pontos esparsos que as do LERF
  # (bonsai: 209 mil, contra 28 mil do ramen), porque o COLMAP roda em 6,5 MP
  # em vez de 0,7 MP. Com a densificação por cima, o backward estourava os
  # 8 GB da placa lá pela iteração 1900 — com 6,29 GB reservados para 5,03 GB
  # de fato alocados, ou seja, mais de 1 GB perdido em fragmentação. Limitar o
  # tamanho dos blocos do alocador recupera essa folga e o treino passa do
  # pico da densificação sem precisar mexer em nenhum hiperparâmetro do método,
  # o que manteria estas cenas comparáveis às da tese.
  export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
  bash script/train.sh "${CENA}" 1 2>&1 | tee "${LOGS}/${CENA}_4_train.log" | tail -20
fi

echo -e "\n🏁 [$CENA] pipeline concluído."
