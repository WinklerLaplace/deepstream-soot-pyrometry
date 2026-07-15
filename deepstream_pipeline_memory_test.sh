#!/bin/bash

export LC_NUMERIC=C

# ===========================================
# Configuración
# ===========================================

INPUT="/home/shannon/Documents/flame/UNetTRT/datasets/test.mp4"

OUTDIR="outputs"
LOGDIR="$OUTDIR/deepstream_pipeline_memory_logs"
RESULTS_FILE="$OUTDIR/deepstream_pipeline_memory_results.txt"

mkdir -p "$LOGDIR"

REPEATS=10
TEGRA_INTERVAL=1000

# Muestras de idle que se toman antes de cada repetición individual
IDLE_SAMPLES_PER_REP=3

MUX_WIDTH=2048
MUX_HEIGHT=1536
BITRATE=4000000
UDP_HOST="100.115.33.111"
UDP_PORT=5000

PREPROCESS_CFG="/home/shannon/Documents/flame/UNetTRT/deepstream/config_preprocess.txt"
INFER_CFG="/home/shannon/Documents/flame/UNetTRT/deepstream/config_infer_primary.txt"
POSTPROCESS_LIB="/home/shannon/Documents/flame/UNetTRT/deepstream/gst-nvdspostprocess/postprocesslib_impl/libpostprocess_impl.so"
POSTPROCESS_CFG="/home/shannon/Documents/flame/UNetTRT/deepstream/config_postprocess.txt"
FILESINK_OUTPUT="/home/shannon/Documents/flame/UNetTRT/outputs/test_render.mp4"

MODEL_LABEL="U-Net_FP16"

echo "===== DEEPSTREAM PIPELINE MEMORY TEST =====" > "$RESULTS_FILE"
echo "Input: $INPUT" >> "$RESULTS_FILE"
echo "Repeticiones por combinación: $REPEATS" >> "$RESULTS_FILE"
echo "Intervalo tegrastats: ${TEGRA_INTERVAL} ms" >> "$RESULTS_FILE"
echo "Muestras idle por repeticion: $IDLE_SAMPLES_PER_REP" >> "$RESULTS_FILE"
echo "Fecha: $(date)" >> "$RESULTS_FILE"
echo "=================================================" >> "$RESULTS_FILE"

# ===========================================
# Función: medir idle puntual 
# ===========================================

measure_idle () {
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

# ===========================================
# Construcción de gst-launch-1.0 según sink
# ===========================================

build_pipeline () {
    local SINK=$1
    local INFER_CFG=$2
 
    local COMMON="gst-launch-1.0 -e filesrc location=$INPUT ! qtdemux ! h264parse ! "
    COMMON+="nvv4l2decoder ! mux.sink_0 nvstreammux name=mux batch-size=1 "
    COMMON+="width=$MUX_WIDTH height=$MUX_HEIGHT ! "
    COMMON+="nvdspreprocess config-file=$PREPROCESS_CFG ! "
    COMMON+="nvinfer config-file-path=$INFER_CFG ! "
    COMMON+="nvdspostprocess postprocesslib-name=$POSTPROCESS_LIB "
    COMMON+="postprocesslib-config-file=$POSTPROCESS_CFG ! "
 
    case "$SINK" in
        fake)
            echo "${COMMON}fakesink sync=false"
            ;;
        file)
            echo "${COMMON}nvv4l2h264enc bitrate=$BITRATE ! h264parse ! mp4mux ! filesink location=$FILESINK_OUTPUT"
            ;;
        udp)
            echo "${COMMON}nvv4l2h264enc bitrate=$BITRATE ! h264parse ! rtph264pay config-interval=1 pt=96 ! udpsink host=$UDP_HOST port=$UDP_PORT sync=false"
            ;;
        *)
            echo "[ERROR] Sink desconocido: $SINK" >&2
            return 1
            ;;
    esac
}
 
# ===========================================
# Test sobre el modelo
# ===========================================
 
run_test () {
    local NAME=$1
    local INFER_CFG=$2
    local SINK=$3
 
    local LOG_NAME
    LOG_NAME=$(echo "$NAME" | tr ' ' '_' | tr -d '()')
 
    local CMD
    CMD=$(build_pipeline "$SINK" "$INFER_CFG") || return 1
 
    echo ""
    echo "[MODEL: $NAME | SINK: $SINK]" | tee -a "$RESULTS_FILE"
 
    local max_values=()
    local avg_values=()
    local net_max_values=()
    local net_avg_values=()
 
    for ((rep = 1; rep <= REPEATS; rep++)); do
        # --- Idle local, medido justo antes de la repetición ---
        local IDLE_LOG="$LOGDIR/${LOG_NAME}_idle_rep${rep}.txt"
        local rep_idle
        rep_idle=$(measure_idle "$IDLE_LOG")

        if [[ -z "$rep_idle" ]]; then
            echo "  Repetición ${rep}: error midiendo idle local (se omite)" | tee -a "$RESULTS_FILE"
            continue
        fi

        # --- Test del pipeline ---
        local TEGRALOG="$LOGDIR/${LOG_NAME}_tegrastats_rep${rep}.txt"
        local INFERLOG="$LOGDIR/${LOG_NAME}_gstlaunch_rep${rep}.txt"
 
        tegrastats --interval "$TEGRA_INTERVAL" --logfile "$TEGRALOG" &
        local PID=$!
 
        sleep 1
 
        eval "$CMD" > "$INFERLOG" 2>&1
 
        kill "$PID" 2>/dev/null
        wait "$PID" 2>/dev/null
 
        local RAM_VALUES
        RAM_VALUES=$(grep -oP 'RAM \K[0-9]+' "$TEGRALOG")
 
        if [[ -z "$RAM_VALUES" ]]; then
            echo "  Repetición ${rep}: error, sin datos RAM (se omite)" | tee -a "$RESULTS_FILE"
            continue
        fi
 
        local rep_max
        rep_max=$(echo "$RAM_VALUES" | sort -nr | head -1)
        local rep_avg
        rep_avg=$(echo "$RAM_VALUES" | awk '{sum+=$1} END {printf "%.2f", sum/NR}')
 
        local rep_net_max
        rep_net_max=$(awk -v a="$rep_max" -v b="$rep_idle" 'BEGIN {v=a-b; if (v<0) v=0; printf "%.2f", v}')
        local rep_net_avg
        rep_net_avg=$(awk -v a="$rep_avg" -v b="$rep_idle" 'BEGIN {v=a-b; if (v<0) v=0; printf "%.2f", v}')
 
        max_values+=("$rep_max")
        avg_values+=("$rep_avg")
        net_max_values+=("$rep_net_max")
        net_avg_values+=("$rep_net_avg")

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
 
    local RAM_MAX_NET_AVG
    RAM_MAX_NET_AVG=$(printf '%s\n' "${net_max_values[@]}" | awk '{sum+=$1; n+=1} END {printf "%.2f", sum/n}')
    local RAM_MAX_NET_STD
    RAM_MAX_NET_STD=$(printf '%s\n' "${net_max_values[@]}" | awk -v mean="$RAM_MAX_NET_AVG" \
        '{sum+=($1-mean)^2; n+=1} END {printf "%.2f", sqrt(sum/n)}')
 
    local RAM_AVG_NET_AVG
    RAM_AVG_NET_AVG=$(printf '%s\n' "${net_avg_values[@]}" | awk '{sum+=$1; n+=1} END {printf "%.2f", sum/n}')
    local RAM_AVG_NET_STD
    RAM_AVG_NET_STD=$(printf '%s\n' "${net_avg_values[@]}" | awk -v mean="$RAM_AVG_NET_AVG" \
        '{sum+=($1-mean)^2; n+=1} END {printf "%.2f", sqrt(sum/n)}')
 
    echo "" | tee -a "$RESULTS_FILE"
    echo "  Resumen (${#max_values[@]}/${REPEATS} repeticiones válidas):" | tee -a "$RESULTS_FILE"
    echo "    RAM max neta : ${RAM_MAX_NET_AVG} +- ${RAM_MAX_NET_STD} MB" | tee -a "$RESULTS_FILE"
    echo "    RAM avg neta : ${RAM_AVG_NET_AVG} +- ${RAM_AVG_NET_STD} MB" | tee -a "$RESULTS_FILE"
 
    echo "-----------------------------------------------" >> "$RESULTS_FILE"
}

# ===========================================
# Ejecución según modo
# ===========================================

run_test "$MODEL_LABEL" "$INFER_CFG" "fake"
run_test "$MODEL_LABEL" "$INFER_CFG" "file"
run_test "$MODEL_LABEL" "$INFER_CFG" "udp"
 
echo ""
echo "Fin: $(date)" >> "$RESULTS_FILE"
echo "Resultados guardados en: $RESULTS_FILE"