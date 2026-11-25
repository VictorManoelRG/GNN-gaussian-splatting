#!/usr/bin/env bash
# ============================================================================
# EXPERIMENTO 5 - complemento: diagnóstico do embedding da configuração Padrão
#
# As métricas (ARI/NMI/Silhouette e as 2D) da configuração Padrão são
# reaproveitadas do Experimento 1, que rodou exatamente esta configuração.
# O que o Experimento 1 NÃO mediu foi a separação do espaço de embeddings
# (similaridade intra vs inter-rótulo), porque não usou --diagnose_embedding.
#
# Estas 5 rodadas existem só para obter esse número na mesma configuração,
# permitindo completar a tabela de separação com as três configurações. A
# pasta de saída é separada (padraodiag_r*) justamente para deixar claro que
# elas não substituem os números do Experimento 1.
# ============================================================================
set -u

RAIZ="/home/victor/Documentos/gaussian_grouping/gaussian-grouping"
SAIDA="${RAIZ}/experimentos_finais/experimento5"
LOGS="${SAIDA}/logs"
DEVA="${RAIZ}/Tracking-Anything-with-DEVA/example/output_gaussian_dataset/teatime"

mkdir -p "${LOGS}"

for rep in 1 2 3 4 5; do
  tag="padraodiag_r${rep}"
  echo "▶️  ${tag}"
  python train_gnn_final_version.py \
    -m output/teatime --iteration 30000 \
    --masks_path "${DEVA}/Annotations" \
    --deva_json  "${DEVA}/pred.json" \
    --model_type_choice gat --gat_heads 8 \
    --no_visualize_3d \
    --target_gaussians 250000 \
    --use_bbox \
    --no_render_full_pointcloud \
    --edge_distance_mode local_scaling \
    --min_cluster_size 500 --hdbscan_min_samples 3 \
    --hdbscan_cluster_selection_method eom \
    --diagnose_embedding \
    --output "${SAIDA}" \
    --experiment_name "${tag}" > "${LOGS}/${tag}.log" 2>&1 \
    && echo "✅ ${tag}" || echo "❌ falhou: ${tag}"
done

echo "🏁 Diagnóstico da configuração Padrão concluído."
