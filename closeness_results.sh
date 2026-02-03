#!/bin/bash

OUTDIR="outputs/closeness_results"
OUTFILE="$OUTDIR/closeness_img_preprocess.log"

mkdir -p "$OUTDIR"

echo "=== Ejecutando Regression Closeness (img_preprocess) ==="
echo "Inicio: $(date)"
echo ""

python3 eval.py --dataset datasets/img_preprocess --closeness 2>&1 | tee "$OUTFILE"

echo ""
echo "Fin: $(date)"
echo "Resultados guardados en: $OUTFILE"
