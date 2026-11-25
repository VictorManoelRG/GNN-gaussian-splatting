#!/usr/bin/env bash
# ============================================================================
# EXPERIMENTO 5 - braços adicionais: época 0 e época 5
#
# epoca0 -> a GAT roda com os pesos da inicialização, sem nenhuma época de
#           treino. Separa o que a ARQUITETURA (propagação de mensagem no
#           grafo) entrega sozinha do que a LOSS efetivamente aprende: a
#           configuração sem_rede não passa pela rede, esta passa mas não
#           treina, e a diferença entre as duas é o efeito da arquitetura.
# epoca5 -> um ponto bem no começo do treino, entre a época 0 e a época 20.
#
# Mesma configuração das demais rodadas do Experimento 5: Teatime recortado
# pela bounding box do ground truth, GAT de 8 cabeças, loss e HDBSCAN padrão.
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

# avalia <tag>: calcula as métricas 2D de uma rodada que já foi treinada
avalia () {
  local tag="$1"
  python calculate_miou_claude_full_table.py \
    --gt  "${GT}" \
    --pred "${SAIDA}/${tag}/gat/images_coverage" \
    --out  "${METRICAS}/${tag}_miou.json" \
    > "${LOGS}/${tag}_miou.log" 2>&1 \
    && echo "✅ ${tag}" || echo "❌ falhou o mIoU: ${tag}"
}

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
  avalia "${tag}"
}

# A rodada epoca0_r1 já foi executada isoladamente para verificar que o
# caminho sem treino funciona; falta só a avaliação 2D dela.
[ -f "${METRICAS}/epoca0_r1_miou.json" ] || avalia epoca0_r1

for rep in 2 3 4 5; do
  roda epoca0 "${rep}" --train_epochs 0
done

for rep in 1 2 3 4 5; do
  roda epoca5 "${rep}" --train_epochs 5
done

echo "🏁 Braços de época 0 e época 5 concluídos."
