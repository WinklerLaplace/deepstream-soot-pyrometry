#!/bin/bash

OUTDIR="outputs/accuracy_results"
CASES=("A" "B" "C")
mkdir -p "$OUTDIR"

declare -A MODELS=(
  ["unet_base"]="--weights weights/unet.pth --model unet --dataset datasets/data_experimental --experiment"
  ["unet_fp32"]="--weights weights/unet_fp32.engine --model tensorrt --dataset datasets/data_experimental --experiment"
  ["unet_fp16"]="--weights weights/unet_fp16.engine --model tensorrt --dataset datasets/data_experimental --experiment"
  ["unet_int8"]="--weights weights/unet_int8.engine --model tensorrt --dataset datasets/data_experimental --experiment"
  ["attunet_base"]="--weights weights/attunet.pth --model attunet --dataset datasets/data_experimental --experiment"
  ["attunet_fp32"]="--weights weights/attunet_fp32.engine --model tensorrt --dataset datasets/data_experimental --experiment"
  ["attunet_fp16"]="--weights weights/attunet_fp16.engine --model tensorrt --dataset datasets/data_experimental --experiment"
  ["attunet_int8"]="--weights weights/attunet_int8.engine --model tensorrt --dataset datasets/data_experimental --experiment"
)

for CASE in "${CASES[@]}"; do
  for NAME in "${!MODELS[@]}"; do
    OUTFILE="$OUTDIR/${CASE}_${NAME}.log"
    echo "=== Evaluando $NAME (case $CASE) ==="
    CMD="python3 eval.py ${MODELS[$NAME]} --case $CASE 2>&1 | tee -a $OUTFILE"
    echo "Ejecutando: $CMD"
    eval $CMD
    echo "Guardado en: $OUTFILE"
    echo ""
  done
done

echo "Evaluación terminada para todos los modelos en los casos A, B y C."
