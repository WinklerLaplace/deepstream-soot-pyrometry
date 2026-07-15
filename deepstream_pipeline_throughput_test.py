#!/usr/bin/env python3

"""
Throughput de referencia para DeepStream vía ejecución nativa. 
Se usa gst-launch-1.0 sin pad probes ni overhead de GIL.

Se apoya en la línea que gst-launch-1.0 imprime al finalizar:
    Execution ended after H:MM:SS.nnnnnnnnn

El pipeline se trunca a --frames frames insertando `identity eos-after=N` 
justo después del decoder forzando un EOS tras N buffers decodificados.

Modo simple (sin --calibrate):
  Un único truncamiento a --frames, repetido --repeats veces. 
  Útil para validar el script contra una corrida manual conocida.

Modo --calibrate (truncated replications):
  Corre dos longitudes del mismo video (--warmup-frames y --frames),
  cada una --repeats veces, empareja cada repetición de warm-up con la
  correspondiente del total, y calcula el fps de estado estacionario
  por par:
      fps_estable_i = (frames_total - frames_warmup) / (T_total_i - T_warmup_i)
  Esto cancela cualquier costo fijo de arranque sin necesidad de descartar
  frames individuales (inviable en un pipeline paralelo sin pad probes).
  Además reporta el margen de incertidumbre (SE) de delta_time, para
  distinguir si una diferencia observada es señal real o ruido de fondo.

Ejemplo:
  python deepstream_pipeline_throughput_test.py --sink fake --calibrate --frames 900 --warmup-frames 90 --repeats 10
"""

import argparse
import re
import subprocess
import statistics
import sys
import os

OUTFILE = "outputs/deepstream_pipeline_throughput_results.txt"

class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

DEFAULTS = {
    "input":              "/home/shannon/Documents/flame/UNetTRT/datasets/test.mp4",
    "output":             "/home/shannon/Documents/flame/UNetTRT/outputs/test_render.mp4",
    "preprocess":         "/home/shannon/Documents/flame/UNetTRT/deepstream/config_preprocess.txt",
    "infer":              "/home/shannon/Documents/flame/UNetTRT/deepstream/config_infer_primary.txt",
    "postprocess_lib":    "/home/shannon/Documents/flame/UNetTRT/deepstream/gst-nvdspostprocess/postprocesslib_impl/libpostprocess_impl.so",
    "postprocess_config": "/home/shannon/Documents/flame/UNetTRT/deepstream/config_postprocess.txt",
    "mux_width":          2048,
    "mux_height":         1536,
    "bitrate":            4000000,
    "udp_host":           "100.115.33.111",
    "udp_port":           5000,
}

EXEC_TIME_RE = re.compile(
    r"Execution ended after (\d+):(\d+):(\d+(?:\.\d+)?)"
)

def build_command(args, frames: int) -> str:
    common = (
        f"gst-launch-1.0 -e "
        f"filesrc location={args.input} ! qtdemux ! h264parse ! "
        f"nvv4l2decoder"
        f"{' disable-dpb=true' if args.disable_dpb else ''} ! "
        f"identity eos-after={frames} ! "
        f"mux.sink_0 nvstreammux name=mux batch-size=1 "
        f"width={args.mux_width} height={args.mux_height} ! "
        f"nvdspreprocess config-file={args.preprocess} ! "
        f"nvinfer config-file-path={args.infer} ! "
        f"nvdspostprocess postprocesslib-name={args.postprocess_lib} "
        f"postprocesslib-config-file={args.postprocess_config} ! "
    )

    if args.sink == "fake":
        return common + "fakesink sync=false"

    elif args.sink == "file":
        return (
            common
            + f"nvv4l2h264enc bitrate={args.bitrate} ! h264parse ! mp4mux ! "
              f"filesink location={args.output}"
        )

    elif args.sink == "udp":
        return (
            common
            + f"nvv4l2h264enc bitrate={args.bitrate} ! h264parse ! "
              f"rtph264pay config-interval=1 pt=96 ! "
              f"udpsink host={args.udp_host} port={args.udp_port} sync=false"
        )

    else:
        sys.exit(f"[ERROR] Modo sink desconocido: {args.sink}")

def parse_execution_time(output: str):
    match = EXEC_TIME_RE.search(output)
    if not match:
        return None
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)

def run_once(cmd: str, timeout: int):
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print("  [ERROR] Timeout esperando a que el pipeline termine.")
        return None

    output = result.stdout + result.stderr
    exec_time = parse_execution_time(output)

    if exec_time is None:
        print("  [ERROR] No se encontró 'Execution ended after' en la salida.")
        print("  --- stdout/stderr de las últimas 20 líneas ---")
        for line in output.strip().splitlines()[-20:]:
            print(f"  {line}")
        return None

    return exec_time

def run_repeated(args, frames: int, repeats: int, label: str):
    cmd = build_command(args, frames)

    exec_times = []
    for i in range(repeats):
        print(f"=== [{label}] Repetición {i + 1}/{repeats} ===")
        t = run_once(cmd, timeout=args.timeout)
        if t is None:
            print(f"  Repetición {i + 1} descartada (sin datos válidos).")
            continue
        fps = frames / t
        exec_times.append(t)
        print(f"  Execution ended after: {t:.6f}s  ->  {fps:.2f} fps")

    return exec_times

def main():
    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)
    logfile = open(OUTFILE, "w")
    sys.stdout = _Tee(sys.__stdout__, logfile)

    p = argparse.ArgumentParser(description="Throughput nativo DeepStream (gst-launch-1.0)")
    p.add_argument("--input",               default=DEFAULTS["input"])
    p.add_argument("--output",              default=DEFAULTS["output"])
    p.add_argument("--sink",                default="fake", choices=["fake", "file", "udp"])
    p.add_argument("--udp-host",            default=DEFAULTS["udp_host"])
    p.add_argument("--udp-port",            default=DEFAULTS["udp_port"], type=int)
    p.add_argument("--bitrate",             default=DEFAULTS["bitrate"], type=int)
    p.add_argument("--mux-width",           default=DEFAULTS["mux_width"], type=int)
    p.add_argument("--mux-height",          default=DEFAULTS["mux_height"], type=int)
    p.add_argument("--preprocess",          default=DEFAULTS["preprocess"])
    p.add_argument("--infer",               default=DEFAULTS["infer"])
    p.add_argument("--postprocess-lib",     default=DEFAULTS["postprocess_lib"], dest="postprocess_lib")
    p.add_argument("--postprocess-config",  default=DEFAULTS["postprocess_config"], dest="postprocess_config")
    p.add_argument("--disable-dpb",         action="store_true", dest="disable_dpb")
    p.add_argument("--frames",              type=int, required=True,
                    help="Frames a los que se trunca el pipeline (modo simple), o frames totales de la corrida larga (modo --calibrate)")
    p.add_argument("--calibrate",           action="store_true",
                    help="Método de truncated replications: corre --warmup-frames y --frames por separado y calcula el fps de estado estacionario por diferencia")
    p.add_argument("--warmup-frames",       type=int, default=90,
                    help="Frames usados como referencia de warm-up en modo --calibrate (ajustar al 10% del total de frames)")
    p.add_argument("--repeats",             type=int, default=10)
    p.add_argument("--timeout",             type=int, default=300)
    args = p.parse_args()

    print(f"====== DEEPSTREAM PIPELINE THROUGHPUT TEST ======")
    print(f"[INFO] Sink: {args.sink}  |  Repeticiones: {args.repeats}")
    print(f"=================================================")

    if not args.calibrate:
        # Modo simple: un solo truncamiento a --frames
        exec_times = run_repeated(args, args.frames, args.repeats, label="único")

        if not exec_times:
            print("\n[ERROR] Ninguna repetición produjo datos válidos.")
            return

        fps_values = [args.frames / t for t in exec_times]
        exec_std = statistics.stdev(exec_times) if len(exec_times) > 1 else 0.0
        fps_std = statistics.stdev(fps_values) if len(fps_values) > 1 else 0.0

        print("\n[RESUMEN THROUGHPUT NATIVO — promedio entre repeticiones]")
        print(f"N repeticiones válidas : {len(exec_times)}/{args.repeats}")
        print(f"Tiempo ejecución (avg) : {statistics.mean(exec_times):.4f}s  (std: {exec_std:.4f}s)")
        print(f"FPS (avg)              : {statistics.mean(fps_values):.2f}  (std: {fps_std:.2f})")
        return

    # Modo --calibrate: truncated replications
    if args.warmup_frames >= args.frames:
        sys.exit("[ERROR] --warmup-frames debe ser menor que --frames")

    t_warmup = run_repeated(args, args.warmup_frames, args.repeats, label="warm-up")
    t_total  = run_repeated(args, args.frames,         args.repeats, label="total")

    if not t_warmup or not t_total:
        print("\n[ERROR] Alguna de las dos fases no produjo datos válidos.")
        return

    n_pairs = min(len(t_warmup), len(t_total))
    if n_pairs < len(t_warmup) or n_pairs < len(t_total):
        print(f"[AVISO] Repeticiones descartadas por fallos: "
              f"{len(t_warmup)} warm-up válidas, {len(t_total)} total válidas, "
              f"{n_pairs} pares emparejados.")

    delta_frames = args.frames - args.warmup_frames
    steady_fps_values = []
    for i in range(n_pairs):
        delta_time_i = t_total[i] - t_warmup[i]
        if delta_time_i <= 0:
            print(f"  [AVISO] Par {i+1} descartado: delta_time <= 0 "
                  f"(t_total={t_total[i]:.4f}s, t_warmup={t_warmup[i]:.4f}s).")
            continue
        steady_fps_values.append(delta_frames / delta_time_i)

    if not steady_fps_values:
        print("\n[ERROR] Ningún par produjo un delta_time válido (> 0).")
        return

    mean_warmup = statistics.mean(t_warmup)
    mean_total  = statistics.mean(t_total)
    std_warmup  = statistics.stdev(t_warmup) if len(t_warmup) > 1 else 0.0
    std_total   = statistics.stdev(t_total)  if len(t_total)  > 1 else 0.0

    naive_fps = args.frames / mean_total

    mean_steady = statistics.mean(steady_fps_values)
    std_steady  = statistics.stdev(steady_fps_values) if len(steady_fps_values) > 1 else 0.0

    # ── Margen de incertidumbre de delta_time (propagación de error) ──
    # SE de cada media = std / sqrt(n). Al restar dos medias independientes,
    # los errores se combinan como raíz de la suma de cuadrados.
    se_warmup = std_warmup / (len(t_warmup) ** 0.5)
    se_total  = std_total  / (len(t_total)  ** 0.5)
    se_delta_time = (se_warmup ** 2 + se_total ** 2) ** 0.5

    mean_delta_time = mean_total - mean_warmup
    se_delta_pct = (se_delta_time / mean_delta_time * 100) if mean_delta_time > 0 else float("nan")

    print("\n[RESUMEN THROUGHPUT NATIVO — truncated replications]")
    print(f"Warm-up   : N={args.warmup_frames:4d} frames  |  tiempo (avg) = {mean_warmup:.4f}s  (std: {std_warmup:.4f}s)  |  n={len(t_warmup)}/{args.repeats}")
    print(f"Total     : N={args.frames:4d} frames  |  tiempo (avg) = {mean_total:.4f}s  (std: {std_total:.4f}s)  |  n={len(t_total)}/{args.repeats}")
    print(f"delta_time: {mean_delta_time:.4f}s  "
          f"(margen de incertidumbre SE: ±{se_delta_time:.4f}s, ±{se_delta_pct:.2f}%)")
    print()
    print(f"FPS ingenuo            : {naive_fps:.2f}")
    print(f"FPS estado estacionario: {mean_steady:.2f}  (std: {std_steady:.2f})  |  n={len(steady_fps_values)}/{n_pairs} pares")
    diff_pct = (naive_fps - mean_steady) / mean_steady * 100
    print(f"Diferencia ingenuo vs. estacionario: {diff_pct:+.2f}%")

    if abs(diff_pct) < se_delta_pct:
        print(f"-> La diferencia observada ({diff_pct:+.2f}%) es MENOR que el margen de "
              f"incertidumbre de delta_time (±{se_delta_pct:.2f}%): no se puede distinguir de "
              f"ruido con esta cantidad de repeticiones. Esto NO es evidencia de que el costo "
              f"de arranque sea cero, solo de que -si existe- es menor de lo que este "
              f"experimento puede resolver. Considera aumentar --repeats o usar un delta_frames "
              f"mayor para reducir el margen.")
    else:
        print(f"-> La diferencia observada ({diff_pct:+.2f}%) SUPERA el margen de incertidumbre "
              f"(±{se_delta_pct:.2f}%): hay evidencia de un costo de arranque real y no "
              f"despreciable. Usa el fps de estado estacionario como métrica de referencia.")

if __name__ == "__main__":
    main()
