#!/usr/bin/env bash
# ============================================================================
# EXPERIMENTO 4 - refazer Waldo Kitchen com as 187 câmeras registradas
#
# O COLMAP anterior desta cena rodou sobre imagens já reduzidas (246x181) e
# registrou só 156/187 frames, deixando de fora 2 dos 5 frames anotados no
# ground truth (frame_00140 e frame_00154). Aqui a cena usa a reconstrução
# COLMAP oficial do LERF-OVS (resolução cheia, 187/187 registradas), já copiada
# para data/waldo_kitchen/sparse/0, com as imagens reduzidas por 4 em images/.
#
# Etapas (cada uma pula se já estiver concluída):
#   1. SAM + DEVA sobre data/waldo_kitchen/images
#   2. treino do Gaussian Grouping + render.py (todas as câmeras)
#   3. métricas 2D do Gaussian Grouping nos 5 frames anotados
#   4. nosso método, 5 repetições, mesma configuração do Experimento 4
#      (GAT 8 cabeças, loss padrão, HDBSCAN mcs=200/ms=3/eom, sem recorte)
#
# Versão 2: o min_cluster_size vem como argumento e a etapa 4 grava em
# experimentos_finais/experimento4/waldo_kitchen_mcs<valor>/ (logs e métricas
# próprios), sem sobrescrever os resultados de outros valores.
#
# Uso: ./run_colmap_env.sh ./refazer_waldo_kitchen_187cams_2.sh <min_cluster_size>
# ============================================================================
set -u -o pipefail

CENA=waldo_kitchen
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GT=$ROOT/data/lerf_ovs/label/$CENA
DEVA=Tracking-Anything-with-DEVA/example/output_gaussian_dataset/$CENA
MCS="${1:?uso: refazer_waldo_kitchen_187cams_2.sh <min_cluster_size>}"
GGDIR=$ROOT/experimentos_finais/experimento4/gaussian_grouping
OUTDIR=$ROOT/experimentos_finais/experimento4/${CENA}_mcs${MCS}
LOGDIR=$OUTDIR/logs; REPDIR=$OUTDIR/metricas
SEED=42; REPS=(1 2 3 4 5); MS=3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
mkdir -p "$LOGDIR" "$REPDIR/relatorios" "$GGDIR/logs" logs_cenas
cd "$ROOT" || exit 1

etapa () { echo -e "\n=========== [$CENA] $* ===========  $(date '+%H:%M:%S')"; }

# Confere que as 5 predições anotadas existem numa pasta (nomes frame_XXXXX.png)
confere_frames () {
  local pasta="$1" falta=0
  for j in "$GT"/*.json; do
    f=$(basename "$j" .json)
    [ -f "$pasta/$f.png" ] || { echo "❌ falta $pasta/$f.png"; falta=1; }
  done
  return $falta
}

# ------------------------------------------------------------- 1. SAM + DEVA
if [ -n "$(ls -A data/$CENA/object_mask/*.png 2>/dev/null)" ]; then
  etapa "1/4 máscaras já geradas, pulando"
else
  etapa "1/4 pseudo-labels (SAM + DEVA)"
  bash script/prepare_pseudo_label.sh $CENA 1 > logs_cenas/${CENA}_187_deva.log 2>&1
  if [ -d data/$CENA/object_mask/Annotations ]; then
    mv data/$CENA/object_mask/Annotations/*.png data/$CENA/object_mask/ && rmdir data/$CENA/object_mask/Annotations
  fi
  N=$(ls -1 data/$CENA/object_mask/*.png 2>/dev/null | wc -l)
  echo "   $N máscaras em object_mask/"
  [ "$N" -eq 187 ] || { echo "❌ esperava 187 máscaras"; exit 1; }
fi

# ------------------------------------------- 2. Gaussian Grouping treino+render
if [ -d output/$CENA/train/ours_30000/objects_pred ] && \
   [ "$(ls output/$CENA/train/ours_30000/objects_pred | wc -l)" -eq 187 ]; then
  etapa "2/4 Gaussian Grouping já treinado e renderizado, pulando"
else
  etapa "2/4 treino + render do Gaussian Grouping"
  bash script/train.sh $CENA 1 > logs_cenas/${CENA}_187_train.log 2>&1
  N=$(ls output/$CENA/train/ours_30000/objects_pred 2>/dev/null | wc -l)
  echo "   $N máscaras renderizadas pelo Gaussian Grouping"
  [ "$N" -eq 187 ] || { echo "❌ esperava 187 renders (ver logs_cenas/${CENA}_187_train.log)"; exit 1; }
fi

# ------------------------------------------------ 3. métricas Gaussian Grouping
if [ -f "$GGDIR/${CENA}_miou.json" ] && confere_frames "$GGDIR/$CENA" > /dev/null; then
  etapa "3/4 métricas do Gaussian Grouping já calculadas, pulando"
else
etapa "3/4 métricas 2D do Gaussian Grouping"
python - << 'PY'
import json, os, glob, shutil
s = "waldo_kitchen"
cams = json.load(open(f"output/{s}/cameras.json"))
inv = {e["img_name"]: e["id"] for e in cams}
gt = sorted(os.path.basename(p)[:-5] for p in glob.glob(f"data/lerf_ovs/label/{s}/*.json"))
dst = f"experimentos_finais/experimento4/gaussian_grouping/{s}"; os.makedirs(dst, exist_ok=True)
for f in gt:
    shutil.copy(f"output/{s}/train/ours_30000/objects_pred/{inv[f]:05d}.png", f"{dst}/{f}.png")
print(f"   {len(gt)} predições do GG copiadas: {gt}")
PY
confere_frames "$GGDIR/$CENA" || exit 1
python calculate_miou_claude_full_table.py --pred "$GGDIR/$CENA" --gt "$GT" \
  --out "$GGDIR/${CENA}_miou.json" --quiet > "$GGDIR/logs/${CENA}.log" 2>&1 \
  || { echo "❌ métricas do GG falharam"; exit 1; }
fi

# ------------------------------------------------------------ 4. nosso método
etapa "4/4 nosso método, ${#REPS[@]} repetições (mcs=$MCS ms=$MS, sem recorte)"
FALHAS=()
for REP in "${REPS[@]}"; do
  RUN="${CENA}_r${REP}"
  echo "  -> repetição $REP  $(date '+%H:%M:%S')"
  python train_gnn_final_version.py -m output/$CENA --iteration 30000 --seed "$SEED" \
      --masks_path "$DEVA/Annotations" --deva_json "$DEVA/pred.json" \
      --model_type_choice gat --gat_heads 8 --no_visualize_3d --target_gaussians 250000 \
      --no_bbox --no_render_full_pointcloud --edge_distance_mode local_scaling \
      --min_cluster_size "$MCS" --hdbscan_min_samples "$MS" \
      --hdbscan_cluster_selection_method eom --loss_color_weight 0.0 \
      --output "$OUTDIR" --experiment_name "$RUN" > "$LOGDIR/${RUN}.log" 2>&1 || {
        echo "  !! treino falhou: $RUN"; FALHAS+=("$RUN"); continue; }
  cp "$OUTDIR/$RUN/gat/"*_report_*.txt "$REPDIR/relatorios/${RUN}.txt" 2>/dev/null
  confere_frames "$OUTDIR/$RUN/gat/images_coverage" || { FALHAS+=("$RUN frames"); continue; }
  python calculate_miou_claude_full_table.py --pred "$OUTDIR/$RUN/gat/images_coverage" \
      --gt "$GT" --out "$REPDIR/${RUN}_miou.json" --quiet > "$LOGDIR/${RUN}_miou.log" 2>&1 \
      && echo "     ok" || { echo "  !! metricas falharam: $RUN"; FALHAS+=("${RUN} 2D"); }
done

# Mantém em disco só a repetição de maior mIoU, como nas outras cenas
MELHOR=$(python - "$REPDIR" "$CENA" << 'PY'
import json, os, sys
rep, nome = sys.argv[1], sys.argv[2]
melhor, valor = None, -1
for r in "12345":
    p = os.path.join(rep, f"{nome}_r{r}_miou.json")
    if os.path.exists(p):
        v = json.load(open(p))["aggregated"]["mean_iou_all_matches"]
        if v > valor: melhor, valor = r, v
print(melhor or "")
PY
)
if [ -n "$MELHOR" ] && [ ${#FALHAS[@]} -eq 0 ]; then
  echo "  melhor mIoU: repetição $MELHOR"
  rm -rf "$OUTDIR/$CENA"; mv "$OUTDIR/${CENA}_r${MELHOR}" "$OUTDIR/$CENA"
  echo "repeticao $MELHOR (semente $SEED) -- sem recorte, 187 cameras, mcs $MCS" > "$OUTDIR/$CENA/REPETICAO_PRESERVADA.txt"
  for R in "${REPS[@]}"; do [ "$R" != "$MELHOR" ] && rm -rf "$OUTDIR/${CENA}_r${R}"; done
fi

echo "=================================================================="
[ ${#FALHAS[@]} -eq 0 ] && echo "🏁 TODAS AS RODADAS OK." || echo "CONCLUIDO com problemas: ${FALHAS[*]}"
