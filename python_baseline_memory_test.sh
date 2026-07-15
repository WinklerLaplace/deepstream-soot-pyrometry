#!/bin/bash

export LC_NUMERIC=C

DATASET="datasets/img_preprocess"
OUTDIR="outputs"
LOGDIR="$OUTDIR/python_baseline_memory_logs"
RESULTS_FILE="$OUTDIR/python_baseline_memory_results.txt"

mkdir -p "$LOGDIR"

BATCH_SIZE=1
REPEATS=10
TEGRA_INTERVAL=1000

# Muestras de idle que se toman ANTES de cada repetición individual
# (en vez de una sola medición global al inicio del script)
IDLE_SAMPLES_PER_REP=3

MODEL_PT_UNET="unet.pth"
MODEL_PT_ATT="attunet.pth"

MODEL_TRT_UNET_FP32="unet_fp32.engine"
MODEL_TRT_UNET_FP16="unet_fp16.engine"
MODEL_TRT_UNET_INT8="unet_int8.engine"

MODEL_TRT_ATT_FP32="attunet_fp32.engine"
MODEL_TRT_ATT_FP16="attunet_fp16.engine"
MODEL_TRT_ATT_INT8="attunet_int8.engine"

echo "===== PYTHON BASELINE MEMORY TEST =====" > "$RESULTS_FILE"
echo "Dataset: $DATASET" >> "$RESULTS_FILE"
echo "Batch size: $BATCH_SIZE" >> "$RESULTS_FILE"
echo "Repeticiones por modelo: $REPEATS" >> "$RESULTS_FILE"
echo "Intervalo tegrastats: ${TEGRA_INTERVAL} ms" >> "$RESULTS_FILE"
echo "Muestras idle por repeticion: $IDLE_SAMPLES_PER_REP" >> "$RESULTS_FILE"
echo "Fecha: $(date)" >> "$RESULTS_FILE"
echo "=================================================" >> "$RESULTS_FILE"

# =========================
#  Función: medir idle puntual (usada antes de cada repetición)
# =========================

measure_idle () {
    # $1 = ruta del log donde guardar la medición idle

    local IDLE_LOG=$1

    tegrastats --interval "$TEGRA_INTERVAL" --logfile "$IDLE_LOG" &
    local IDLE_PID=$!

    local sleep_time
    sleep_time=$(awk "BEGIN {print ($IDLE_SAMPLES_PER_REP + 1) * $TEGRA_INTERVAL / 1000}")
    sleep "$sleep_time"

    kill "$IDLE_PID" 2>/dev/null
    wait "$IDLE_PID" 2>/dev/null

    local idle_values
    idle_values=$(grep -oP 'RAM \K[0-9]+' "$IDLE_LOG" | head -n "$IDLE_SAMPLES_PER_REP")

    if [[ -z "$idle_values" ]]; then
        echo ""
        return 1
    fi

    echo "$idle_values" | awk '{sum+=$1; n+=1} END {printf "%.2f", sum/n}'
    return 0
}

# =========================
#  Test por modelo
# =========================

run_test () {

    NAME=$1
    WEIGHTS=$2
    MODEL_TYPE=$3

    LOG_NAME=$(echo "$NAME" | tr ' ' '_' | tr -d '()')

    echo ""
    echo "[MODEL: $NAME]" | tee -a "$RESULTS_FILE"

    max_values=()
    avg_values=()
    net_max_values=()
    net_avg_values=()
    idle_values_per_rep=()

    for ((rep = 1; rep <= REPEATS; rep++)); do

        # --- Idle local, medido justo antes de esta repetición ---
        IDLE_LOG="$LOGDIR/${LOG_NAME}_idle_rep${rep}.txt"
        rep_idle=$(measure_idle "$IDLE_LOG")

        if [[ -z "$rep_idle" ]]; then
            echo "  Repetición ${rep}: error midiendo idle local (se omite)" | tee -a "$RESULTS_FILE"
            continue
        fi

        # --- Test de inferencia ---
        TEGRALOG="$LOGDIR/${LOG_NAME}_tegrastats_rep${rep}.txt"
        INFERLOG="$LOGDIR/${LOG_NAME}_inference_rep${rep}.txt"

        tegrastats --interval "$TEGRA_INTERVAL" --logfile "$TEGRALOG" &
        PID=$!

        sleep 1

        python3 eval.py \
            --weights "weights/$WEIGHTS" \
            --model "$MODEL_TYPE" \
            --dataset "$DATASET" \
            --memory \
            --batch_size "$BATCH_SIZE" \
            > "$INFERLOG"

        kill "$PID" 2>/dev/null
        wait "$PID" 2>/dev/null

        RAM_VALUES=$(grep -oP 'RAM \K[0-9]+' "$TEGRALOG")

        if [[ -z "$RAM_VALUES" ]]; then
            echo "  Repetición ${rep}: error, sin datos RAM (se omite)" | tee -a "$RESULTS_FILE"
            continue
        fi

        rep_max=$(echo "$RAM_VALUES" | sort -nr | head -1)
        rep_avg=$(echo "$RAM_VALUES" | awk '{sum+=$1} END {printf "%.2f", sum/NR}')

        # Neto = usando el idle medido justo antes de ESTA repetición,
        # no un idle global fijo. Se protege contra negativos por
        # seguridad (variabilidad residual de tegrastats).
        rep_net_max=$(awk -v a="$rep_max" -v b="$rep_idle" 'BEGIN {v=a-b; if (v<0) v=0; printf "%.2f", v}')
        rep_net_avg=$(awk -v a="$rep_avg" -v b="$rep_idle" 'BEGIN {v=a-b; if (v<0) v=0; printf "%.2f", v}')

        max_values+=("$rep_max")
        avg_values+=("$rep_avg")
        net_max_values+=("$rep_net_max")
        net_avg_values+=("$rep_net_avg")
        idle_values_per_rep+=("$rep_idle")

        echo "  Repetición ${rep}/${REPEATS}:" | tee -a "$RESULTS_FILE"
        echo "    idle          : ${rep_idle} MB" | tee -a "$RESULTS_FILE"
        echo "    RAM max total : ${rep_max} MB" | tee -a "$RESULTS_FILE"
        echo "    RAM avg total : ${rep_avg} MB" | tee -a "$RESULTS_FILE"
        echo "    RAM max neta  : ${rep_net_max} MB" | tee -a "$RESULTS_FILE"
        echo "    RAM avg neta  : ${rep_net_avg} MB" | tee -a "$RESULTS_FILE"

    done

    if [[ ${#max_values[@]} -eq 0 ]]; then
        echo "Error: ninguna repetición produjo datos válidos" | tee -a "$RESULTS_FILE"
        echo "-----------------------------------------------" >> "$RESULTS_FILE"
        return
    fi

    RAM_MAX_NET_AVG=$(printf '%s\n' "${net_max_values[@]}" | awk '{sum+=$1; n+=1} END {printf "%.2f", sum/n}')
    RAM_MAX_NET_STD=$(printf '%s\n' "${net_max_values[@]}" | awk -v mean="$RAM_MAX_NET_AVG" \
        '{sum+=($1-mean)^2; n+=1} END {printf "%.2f", sqrt(sum/n)}')

    RAM_AVG_NET_AVG=$(printf '%s\n' "${net_avg_values[@]}" | awk '{sum+=$1; n+=1} END {printf "%.2f", sum/n}')
    RAM_AVG_NET_STD=$(printf '%s\n' "${net_avg_values[@]}" | awk -v mean="$RAM_AVG_NET_AVG" \
        '{sum+=($1-mean)^2; n+=1} END {printf "%.2f", sqrt(sum/n)}')

    echo "" | tee -a "$RESULTS_FILE"
    echo "  Resumen (${#max_values[@]}/${REPEATS} repeticiones válidas):" | tee -a "$RESULTS_FILE"
    echo "    RAM max neta : ${RAM_MAX_NET_AVG} +- ${RAM_MAX_NET_STD} MB" | tee -a "$RESULTS_FILE"
    echo "    RAM avg neta : ${RAM_AVG_NET_AVG} +- ${RAM_AVG_NET_STD} MB" | tee -a "$RESULTS_FILE"

    echo "-----------------------------------------------" >> "$RESULTS_FILE"
}

# =========================
#  U-Net
# =========================

# run_test "U-Net PyTorch Base" $MODEL_PT_UNET "unet"

# run_test "U-Net TensorRT FP32" $MODEL_TRT_UNET_FP32 "tensorrt"
# run_test "U-Net TensorRT FP16" $MODEL_TRT_UNET_FP16 "tensorrt"
# run_test "U-Net TensorRT INT8" $MODEL_TRT_UNET_INT8 "tensorrt"

# =========================
#  Attention U-Net
# =========================

run_test "Attention U-Net PyTorch Base" $MODEL_PT_ATT "attunet"

# run_test "Attention U-Net TensorRT FP32" $MODEL_TRT_ATT_FP32 "tensorrt"
# run_test "Attention U-Net TensorRT FP16" $MODEL_TRT_ATT_FP16 "tensorrt"
# run_test "Attention U-Net TensorRT INT8" $MODEL_TRT_ATT_INT8 "tensorrt"

echo ""
echo "Fin: $(date)" >> "$RESULTS_FILE"
echo "Resultados guardados en: $RESULTS_FILE"