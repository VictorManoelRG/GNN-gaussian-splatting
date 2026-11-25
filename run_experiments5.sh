#!/usr/bin/env bash
# ============================================================================
# EXPERIMENTO 5 - Contribuição do treinamento da GNN
#
# Três pontos da MESMA pipeline, mudando apenas QUANTO a rede foi treinada.
# Todo o resto (grafo, features, HDBSCAN, bbox, filtros) é idêntico ao
# Experimento 1, para que a única variável seja o treino:
#
#   sem_rede  -> HDBSCAN direto nas features de entrada (xyz normalizado + Lab),
#                descartando a GNN por completo (--cluster_on_raw_features com
#                --train_epochs 0, já que o embedding treinado seria descartado)
#   epoca20   -> embedding da GAT após apenas 20 épocas de treino
#   padrao    -> treino completo; os números vêm do Experimento 1 (mesma
#                configuração), conforme a prática de reaproveitar a baseline
#
# Cena: Teatime, recortada pela bounding box do ground truth (--use_bbox).
# 5 repetições por configuração, como nos demais experimentos da tese.
# ============================================================================
set -u

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAIDA="${RAIZ}/experimentos_finais/experimento5"
LOGS="${SAIDA}/logs"
METRICAS="${SAIDA}/metricas"
GT="${RAIZ}/data/lerf_ovs/label/teatime"
DEVA="${RAIZ}/Tracking-Anything-with-DEVA/example/output_gaussian_dataset/teatime"

mkdir -p "${LOGS}" "${METRICAS}"

BASE=(python train_gnn_final_version.py
  -m output/teatime --iteration 30000
  --masks_path "${DEVA}/Annotations"
  --deva_json  "${DEVA}/pred.json"
  --model_type_choice gat --gat_heads 8
  --no_visualize_3d
  --target_gaussians 250000
  --use_bbox
  --no_render_full_pointcloud
  --edge_distance_mode local_scaling
  --min_cluster_size 500 --hdbscan_min_samples 3
  --hdbscan_cluster_selection_method eom
  --diagnose_embedding
  --output "${SAIDA}")

# roda <nome> <repeticao> <flags específicas...>
roda () {
  local nome="$1"; shift
  local rep="$1"; shift
  local tag="${nome}_r${rep}"

  echo "▶️  ${tag}"
  "${BASE[@]}" --experiment_name "${tag}" "$@" > "${LOGS}/${tag}.log" 2>&1
  if [ $? -ne 0 ]; then
    echo "❌ falhou: ${tag} (ver ${LOGS}/${tag}.log)"
    return 1
  fi

  python calculate_miou_claude_full_table.py \
    --gt  "${GT}" \
    --pred "${SAIDA}/${tag}/gat/images_coverage" \
    --out  "${METRICAS}/${tag}_miou.json" \
    > "${LOGS}/${tag}_miou.log" 2>&1
  if [ $? -ne 0 ]; then
    echo "❌ falhou o mIoU: ${tag} (ver ${LOGS}/${tag}_miou.log)"
    return 1
  fi
  echo "✅ ${tag}"
}

for rep in 1 2 3 4 5; do
  roda sem_rede "${rep}" --cluster_on_raw_features --train_epochs 0
done

for rep in 1 2 3 4 5; do
  roda epoca20 "${rep}" --train_epochs 20
done

echo "🏁 Experimento 5 concluído."
