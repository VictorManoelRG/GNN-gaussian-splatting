#!/usr/bin/env bash
# ============================================================================
# Encadeia o que falta: espera o treino do bonsai (rodado à parte, depois do
# OOM) terminar, faz o render dele, e só então começa o room inteiro — para as
# duas cenas não disputarem a GPU.
# ============================================================================
set -u

RAIZ="/home/victor/Documentos/gaussian_grouping/gaussian-grouping"
LOGS="${RAIZ}/logs_cenas"
cd "${RAIZ}" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

while pgrep -f "train.py -s data/bonsai" > /dev/null; do
  sleep 60
done

if [ ! -d "${RAIZ}/output/bonsai/point_cloud/iteration_30000" ]; then
  echo "❌ o treino do bonsai não chegou a 30000 — room NÃO foi iniciado."
  exit 1
fi
echo "✅ bonsai treinado."

echo "=========== [bonsai] render ==========="
./run_colmap_env.sh env PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  python render.py -m output/bonsai --num_classes 256 \
  > "${LOGS}/bonsai_5_render.log" 2>&1 \
  && echo "🏁 [bonsai] render concluído" || echo "❌ render do bonsai falhou"

./run_colmap_env.sh ./pipeline_cena.sh room 9
