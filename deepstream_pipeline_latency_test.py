#!/usr/bin/env python3

"""
Latencia por etapa de un pipeline DeepStream, medida con pad probes.

Cada repetición construye el pipeline, instala pad probes de entrada/salida 
en decoder, preprocess, infer y postprocess, corre hasta EOS, descarta los 
primeros --warmup frames de todas las estadísticas por etapa y end-to-end, 
y calcula, sobre los frames restantes: mean, median, p95, max.

El proceso padre agrega estos resultados entre las --repeats repeticiones
promediando cada estadístico (mean/median/p95/max) por separado y reportando
también su desviación estándar entre corridas.

Ejemplo:
  python deepstream_pipeline_latency_test.py --sink fake --warmup 90 --repeats 10
  python deepstream_pipeline_latency_test.py --sink file
  python deepstream_pipeline_latency_test.py --sink udp --no-profile
"""

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time

OUTFILE = "outputs/deepstream_pipeline_latency_results.txt"

RESULT_MARKER = "@@DS_LATENCY_RESULT_JSON@@"

STAGES = ["decode", "preprocess", "infer", "postprocess"]

STAGE_LABELS = {
    "decode":      "Decodificación     (nvv4l2decoder)  ",
    "preprocess":  "Preprocesamiento   (nvdspreprocess) ",
    "infer":       "Inferencia         (nvinfer FP16)   ",
    "postprocess": "Postprocesamiento  (nvdspostprocess)",
}

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
    "warmup":             90,
    "repeats":            10,
    "timeout":            300,
}

class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except ValueError:
                pass 

def stats(vals):
    s = sorted(vals)
    n = len(s)
    return (
        statistics.mean(s),
        statistics.median(s),
        s[max(0, int(n * 0.95) - 1)],
        s[-1],
        statistics.stdev(s) if n > 1 else 0.0,
    )

# # =======================================================================
# Construcción de argumentos
# # =======================================================================

def build_arg_parser():
    p = argparse.ArgumentParser(description="Latencia por etapa de pipeline DeepStream (pad probes)")
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
    p.add_argument("--warmup",              default=DEFAULTS["warmup"], type=int,
                    help="Frames de warm-up excluidos de las estadísticas por repetición "
                         "(recomendado: 10%% del total de frames del video)")
    p.add_argument("--repeats",             default=DEFAULTS["repeats"], type=int)
    p.add_argument("--timeout",             default=DEFAULTS["timeout"], type=int,
                    help="Timeout en segundos por repetición (subproceso)")
    p.add_argument("--no-profile",          action="store_true")
    p.add_argument("--_worker",             action="store_true", dest="_worker", help=argparse.SUPPRESS)
    return p

def worker_argv(args):
    argv = [
        "--_worker",
        "--input", args.input,
        "--output", args.output,
        "--sink", args.sink,
        "--udp-host", args.udp_host,
        "--udp-port", str(args.udp_port),
        "--bitrate", str(args.bitrate),
        "--mux-width", str(args.mux_width),
        "--mux-height", str(args.mux_height),
        "--preprocess", args.preprocess,
        "--infer", args.infer,
        "--postprocess-lib", args.postprocess_lib,
        "--postprocess-config", args.postprocess_config,
        "--warmup", str(args.warmup),
    ]
    if args.disable_dpb:
        argv.append("--disable-dpb")
    if args.no_profile:
        argv.append("--no-profile")
    return argv

# # =======================================================================
# Modo WORKER (hijo): construye y corre una sola repetición
# # =======================================================================

def _run_worker(args):
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("GLib", "2.0")
    from gi.repository import Gst, GLib

    Gst.init(None)

    WARMUP_FRAMES = args.warmup

    entry_times = {s: [] for s in STAGES}
    timings = {s: [] for s in STAGES}
    stage_seen_count = {s: 0 for s in STAGES}
    e2e_entry = []
    e2e_timings = []
    state = {"total_frames_seen": 0, "e2e_seen_count": 0}

    def make_elem(factory, name=None):
        elem = Gst.ElementFactory.make(factory, name)
        if elem is None:
            sys.exit(f"[ERROR] No se pudo crear '{factory}'.")
        return elem

    def link_or_die(a, b):
        # A diferencia de a.link(b) solo, esto NO se queda callado si el
        # link falla (p.ej. por incompatibilidad de caps entre h264parse
        # y mp4mux/qtmux). Antes, un link fallido dejaba el pad sin
        # conectar, lo que produce un GST_FLOW_NOT_LINKED silencioso más
        # adelante: el pipeline termina rápido, con 0 frames medidos, y
        # el motivo real nunca se veía en el reporte.
        if not a.link(b):
            sys.exit(
                f"[ERROR] Fallo al linkear '{a.get_name()}' -> '{b.get_name()}' "
                f"(probable incompatibilidad de caps entre ambos elementos)."
            )

    def build_pipeline():
        pipeline = Gst.Pipeline.new("ds-profile-pipeline")

        src     = make_elem("filesrc",       "src")
        demux   = make_elem("qtdemux",       "demux")
        h264p   = make_elem("h264parse",     "h264parse_in")
        decoder = make_elem("nvv4l2decoder", "decoder")
        if args.disable_dpb:
            decoder.set_property("disable-dpb", True)
        src.set_property("location", args.input)

        mux = make_elem("nvstreammux", "mux")
        mux.set_property("batch-size",  1)
        mux.set_property("width",       args.mux_width)
        mux.set_property("height",      args.mux_height)
        mux.set_property("live-source", 0)

        preproc  = make_elem("nvdspreprocess",  "preproc")
        infer0   = make_elem("nvinfer",         "infer0")
        postproc = make_elem("nvdspostprocess", "postproc")
        preproc.set_property("config-file",                args.preprocess)
        infer0.set_property("config-file-path",            args.infer)
        postproc.set_property("postprocesslib-name",        args.postprocess_lib)
        postproc.set_property("postprocesslib-config-file", args.postprocess_config)

        encoder = h264p2 = mp4mux = rtppay = sink_elem = None

        if args.sink == "fake":
            sink_elem = make_elem("fakesink", "sink")
            sink_elem.set_property("sync",  False)
            sink_elem.set_property("async", False)
        elif args.sink == "file":
            encoder   = make_elem("nvv4l2h264enc", "encoder")
            h264p2    = make_elem("h264parse",     "h264parse_out")
            mp4mux    = make_elem("mp4mux",        "mp4mux")
            sink_elem = make_elem("filesink",      "sink")
            encoder.set_property("bitrate", args.bitrate)
            out_dir = os.path.dirname(os.path.abspath(args.output))
            os.makedirs(out_dir, exist_ok=True)
            sink_elem.set_property("location", args.output)
            sink_elem.set_property("sync", False)
        elif args.sink == "udp":
            encoder   = make_elem("nvv4l2h264enc", "encoder")
            h264p2    = make_elem("h264parse",     "h264parse_out")
            rtppay    = make_elem("rtph264pay",    "rtppay")
            sink_elem = make_elem("udpsink",       "sink")
            encoder.set_property("bitrate", args.bitrate)
            sink_elem.set_property("host", args.udp_host)
            sink_elem.set_property("port", args.udp_port)
            sink_elem.set_property("sync", False)
        else:
            sys.exit(f"[ERROR] Modo sink desconocido: {args.sink}")

        for e in [src, demux, h264p, decoder, mux, preproc, infer0, postproc]:
            pipeline.add(e)
        for e in [encoder, h264p2, mp4mux, rtppay, sink_elem]:
            if e is not None:
                pipeline.add(e)

        link_or_die(src, demux)
        link_or_die(h264p, decoder)

        mux_sink = mux.get_request_pad("sink_0")
        if mux_sink is None:
            sys.exit("[ERROR] No se pudo obtener mux.sink_0")
        if decoder.get_static_pad("src").link(mux_sink) != Gst.PadLinkReturn.OK:
            sys.exit("[ERROR] Error al linkear decoder → mux")

        link_or_die(mux, preproc)
        link_or_die(preproc, infer0)
        link_or_die(infer0, postproc)

        if args.sink == "fake":
            link_or_die(postproc, sink_elem)
        elif args.sink == "file":
            link_or_die(postproc, encoder)
            link_or_die(encoder, h264p2)
            link_or_die(h264p2, mp4mux)
            link_or_die(mp4mux, sink_elem)
        elif args.sink == "udp":
            link_or_die(postproc, encoder)
            link_or_die(encoder, h264p2)
            link_or_die(h264p2, rtppay)
            link_or_die(rtppay, sink_elem)

        def on_pad_added(element, pad):
            caps = pad.get_current_caps() or pad.query_caps(None)
            if caps and "video" in caps.get_structure(0).get_name():
                sink_pad = h264p.get_static_pad("sink")
                if not sink_pad.is_linked():
                    ret = pad.link(sink_pad)
                    if ret != Gst.PadLinkReturn.OK:
                        print(f"  [WARN] Error al linkear qtdemux pad: {ret}")

        demux.connect("pad-added", on_pad_added)

        return pipeline

    def probe_entry(stage):
        def cb(pad, info):
            now = time.perf_counter_ns()
            entry_times[stage].append(now)
            if stage == "decode":
                e2e_entry.append(now)
            return Gst.PadProbeReturn.OK
        return cb

    def probe_exit(stage):
        def cb(pad, info):
            now = time.perf_counter_ns()

            if entry_times[stage]:
                t_in = entry_times[stage].pop(0)
                stage_seen_count[stage] += 1
                if stage_seen_count[stage] > WARMUP_FRAMES:
                    timings[stage].append((now - t_in) / 1e6)

            if stage == "postprocess" and e2e_entry:
                t_in = e2e_entry.pop(0)
                state["total_frames_seen"] += 1
                state["e2e_seen_count"] += 1
                if state["e2e_seen_count"] > WARMUP_FRAMES:
                    e2e_timings.append((now - t_in) / 1e6)

            return Gst.PadProbeReturn.OK
        return cb

    def attach_probes(pipeline):
        stage_map = [
            ("decode",      "decoder",    "sink",    "src"),
            ("preprocess",  "preproc",    "sink",    "src"),
            ("infer",       "infer0",     "sink",    "src"),
            ("postprocess", "postproc",   "sink",    "src"),
        ]
        for stage, elem_name, sink_name, src_name in stage_map:
            elem = pipeline.get_by_name(elem_name)
            if elem is None:
                print(f"  [WARN] Elemento '{elem_name}' no encontrado — probe '{stage}' omitida")
                continue
            sp = elem.get_static_pad(sink_name)
            if sp:
                sp.add_probe(Gst.PadProbeType.BUFFER, probe_entry(stage))
            else:
                print(f"  [WARN] Pad '{sink_name}' no disponible en '{elem_name}'")
            srcp = elem.get_static_pad(src_name)
            if srcp:
                srcp.add_probe(Gst.PadProbeType.BUFFER, probe_exit(stage))
            else:
                print(f"  [WARN] Pad '{src_name}' no disponible en '{elem_name}'")

    pipeline = build_pipeline()

    if not args.no_profile:
        attach_probes(pipeline)

    holder = {"pipeline": pipeline}

    def on_sigint(sig, frame):
        p = holder["pipeline"]
        if p is not None:
            p.send_event(Gst.Event.new_eos())

    signal.signal(signal.SIGINT, on_sigint)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_message(bus, msg):
        if msg.type == Gst.MessageType.EOS:
            loop.quit()
        elif msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            print(f"  [ERROR] {err.message}")
            if dbg:
                print(f"  [DEBUG] {dbg}")
            loop.quit()

    bus.connect("message", on_message)

    t_start = time.perf_counter()
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
    t_wall = time.perf_counter() - t_start

    rep_stats = {
        "t_wall": t_wall,
        "total_frames_seen": state["total_frames_seen"],
        "n_measured": len(e2e_timings),
    }
    for stage in STAGES:
        rep_stats[stage] = stats(timings[stage]) if timings[stage] else None
    rep_stats["e2e"] = stats(e2e_timings) if e2e_timings else None

    print(RESULT_MARKER + json.dumps(rep_stats))

# =======================================================================
# Modo PADRE: lanza subprocesos, uno por repetición, y agrega resultados
# =======================================================================

def _parse_worker_output(output):
    for line in output.splitlines():
        if line.startswith(RESULT_MARKER):
            try:
                return json.loads(line[len(RESULT_MARKER):])
            except json.JSONDecodeError:
                return None
    return None

def _print_output_tail(output, n=20):
    for line in output.strip().splitlines()[-n:]:
        print(f"  {line}")

def run_repeated(args):
    script_path = os.path.abspath(__file__)
    rep_results = []

    for i in range(args.repeats):
        print(f"=== Repetición {i + 1}/{args.repeats} ===")
        cmd = [sys.executable, script_path] + worker_argv(args)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
        except subprocess.TimeoutExpired:
            print("  [ERROR] Timeout esperando a que el pipeline termine.")
            continue

        output = result.stdout + result.stderr
        rep = _parse_worker_output(output)

        if rep is None:
            print("  [ERROR] No se encontró resultado JSON en la salida del subproceso.")
            print("  --- stdout/stderr de las últimas 20 líneas ---")
            _print_output_tail(output)
            continue

        # El worker puede terminar "normalmente" (imprime su JSON) pero sin
        # haber medido ningún frame, típicamente porque el bus recibió un
        # GST_MESSAGE_ERROR muy temprano (link fallido, caps incompatibles,
        # no se pudo abrir el archivo de salida, etc). Antes ese caso pasaba
        # desapercibido porque solo se mostraba la salida cruda cuando el
        # JSON no se podía parsear. Ahora también se muestra aquí.
        has_data = rep.get("n_measured", 0) > 0 or any(rep.get(s) is not None for s in STAGES)
        if not has_data:
            print("  [WARN] Repetición sin frames medidos — mostrando salida completa del worker:")
            _print_output_tail(output)

        e2e_info = ""
        if rep.get("e2e") is not None:
            mn, md, p95, mx, sd = rep["e2e"]
            e2e_info = f"  |  e2e: median={md:.2f}ms p95={p95:.2f}ms max={mx:.2f}ms"
        print(f"  Frames: {rep['total_frames_seen']}  |  "
              f"medidos: {rep['n_measured']}  |  tiempo total: {rep['t_wall']:.2f}s{e2e_info}")

        rep_results.append(rep)

    return rep_results

def _agg_line(label, values):
    if not values:
        print(f"  {label}  (sin datos)")
        return
    avg = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    print(f"    {label:8s}: {avg:8.2f}ms  (std entre repeticiones: {sd:.2f}ms)")

def print_aggregate_report(rep_results, args):
    n_valid = len(rep_results)

    W = 72
    print()
    print("=" * W)
    print("  PROFILING DE LATENCIA POR ETAPA — RESUMEN AGREGADO")
    print(f"  ({n_valid} repeticiones válidas, promedio de estadísticos entre corridas)")
    print("=" * W)
    print(f"  Repeticiones válidas: {n_valid}/{args.repeats}")
    print(f"  Warm-up descartado por repetición: {args.warmup} frames")
    print()

    for stage in STAGES:
        label = STAGE_LABELS[stage]
        vals = [r[stage] for r in rep_results if r.get(stage) is not None]
        if not vals:
            print(f"  {label}  (sin datos)")
            print()
            continue
        print(f"  {label}")
        _agg_line("mean",   [v[0] for v in vals])
        _agg_line("median", [v[1] for v in vals])
        _agg_line("p95",    [v[2] for v in vals])
        _agg_line("max",    [v[3] for v in vals])
        print()

    e2e_vals = [r["e2e"] for r in rep_results if r.get("e2e") is not None]
    if e2e_vals:
        print("  End-to-end  (decode entrada → postprocess salida)")
        _agg_line("mean",   [v[0] for v in e2e_vals])
        _agg_line("median", [v[1] for v in e2e_vals])
        _agg_line("p95",    [v[2] for v in e2e_vals])
        _agg_line("max",    [v[3] for v in e2e_vals])
    else:
        print("  End-to-end: sin datos válidos en ninguna repetición.")

# =======================================================================
# Main
# =======================================================================

def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args._worker:
        _run_worker(args)
        return

    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)
    logfile = open(OUTFILE, "w")
    sys.stdout = _Tee(sys.__stdout__, logfile)

    print(f"======== DEEPSTREAM PIPELINE LATENCY TEST ========")
    print(f"[INFO] Sink        : {args.sink}")
    print(f"[INFO] Profiling   : {'DESACTIVADO' if args.no_profile else 'ACTIVADO'}")
    print(f"[INFO] Warm-up     : {args.warmup} frames excluidos por repetición")
    print(f"[INFO] Repeticiones: {args.repeats}")
    print(f"==================================================")

    rep_results = run_repeated(args)

    if not rep_results:
        print("\n[ERROR] Ninguna repetición produjo datos válidos.")
        sys.stdout = sys.__stdout__
        logfile.close()
        return

    if not args.no_profile:
        print_aggregate_report(rep_results, args)

    sys.stdout = sys.__stdout__
    logfile.close()

if __name__ == "__main__":
    main()