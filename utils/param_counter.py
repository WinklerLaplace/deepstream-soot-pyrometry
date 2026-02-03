import sys
from pathlib import Path
import tensorrt as trt
import torch
import json
import argparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "utils"))

train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if train_on_gpu else "cpu")

def main(opt):
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, namespace='')

    engine_path = Path(opt.engine).resolve()

    with trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())

    inspector = engine.create_engine_inspector()

    total_weights_count = 0

    for layer_index in range(engine.num_layers):
        layer_info_json = inspector.get_layer_information(layer_index, trt.LayerInformationFormat.JSON)
        layer_info = json.loads(layer_info_json)

        if 'Weights' in layer_info and 'Count' in layer_info['Weights']:
            total_weights_count += int(layer_info['Weights']['Count'])

    print(f"Number of parameters in the model: {total_weights_count}")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', default='weights/best.engine', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
    