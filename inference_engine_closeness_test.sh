#!/bin/bash

DATASET="datasets/img_preprocess"

OUTDIR="outputs"
OUTFILE="$OUTDIR/inference_engine_closeness_results.txt"

mkdir -p "$OUTDIR"

echo "===== CLOSENESS TEST =====" > "$OUTFILE"
echo "Dataset: $DATASET" >> "$OUTFILE"
echo "Fecha ejecución: $(date)" >> "$OUTFILE"
echo "-----------------------------------------------" >> "$OUTFILE"

python3 eval.py \
  --dataset $DATASET \
  --closeness \
  2>&1 | tee -a "$OUTFILE"

echo "-----------------------------------------------" >> "$OUTFILE"
echo "Fin: $(date)" >> "$OUTFILE"

echo "Resultados guardados en: $OUTFILE"