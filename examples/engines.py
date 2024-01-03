import os
import copy
import time
import torch
import cupy as cp
import numpy as np
import tensorrt as trt
from typing import List
from collections import OrderedDict
from torch.cuda import nvtx
from safetensors.numpy import load_file
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import engine_from_bytes
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool


def merge_loras(base: str, loras: List[str], scales: List[str]):
    """
    param base: common base engine that contains those key-weights that change in lora
    param lora: lora engine that contains key-weights
    param scales: lora scale
    todo: 需要确定不同lora的keys是否相同
    """
    start = time.time()
    base_dict = load_file(base)
    print(f"**** load base model time cost: {time.time() - start}s ****")
    refit_dict = {}
    start = time.time()
    for lora, scale in zip(loras, scales):
        second = time.time()
        lora_dict = load_file(lora)
        print(f"**** load lora file count: {len(lora_dict)}, time cost: {time.time() - second}s ****")
        for k, v in lora_dict.items():
            if k in refit_dict:
                refit_dict[k] += scale * cp.array(v)
            else:
                refit_dict[k] = cp.array(base_dict[k]) + scale * cp.array(v)
    print(f"**** merge weight dict time cost: {time.time() - start}s ****")
    return refit_dict


class TrtEngine:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.cuda_graph_instance = None  # cuda graph

    def load(self):
        print(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self, reuse_device_memory=None):
        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
        #    self.context.device_memory = reuse_device_memory
        else:
            self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device="cuda"):
        nvtx.range_push("allocate_buffers")
        for idx in range(self.engine.num_io_tensors):
            binding = self.engine[idx]
            shape = self.context.get_binding_shape(idx)
            if shape_dict and binding in shape_dict:
                shape = shape_dict[binding].shape
            else:
                shape = self.context.get_binding_shape(idx)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            if self.engine.binding_is_input(binding):
                self.context.set_binding_shape(idx, shape)
            tensor = torch.empty(tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]).to(device=device)
            self.tensors[binding] = tensor
        nvtx.range_pop()

    def infer(self, feed_dict, stream, use_cuda_graph=False):
        nvtx.range_push("set_tensors")
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)

        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())
        nvtx.range_pop()
        nvtx.range_push("execute")
        noerror = self.context.execute_async_v3(stream)
        if not noerror:
            raise ValueError("ERROR: inference failed.")
        nvtx.range_pop()
        return self.tensors

    def refit_from_dict(self, refit_dict):
        refitter = trt.Refitter(self.engine, TRT_LOGGER)
        all_weights = refitter.get_all()

        # TODO ideally iterate over refit_dict as len(refit_dict) < len(all_weights)
        for layer_name, weights_role in zip(all_weights[0], all_weights[1]):
            if weights_role == trt.WeightsRole.KERNEL:
                custom_name = layer_name + "_TRTKERNEL"
            elif weights_role == trt.WeightsRole.BIAS:
                custom_name = layer_name + "_TRTBIAS"
            else:
                custom_name = layer_name

            if layer_name.startswith("onnx::Trilu"):
                continue

            if custom_name in refit_dict:
                refitter.set_weights(layer_name, weights_role, refit_dict[custom_name].get())
        start = time.time()
        failed = refitter.refit_cuda_engine()
        print(f"**** refitter.refit_cuda_engine() time cost: {time.time() - start}s ****")
        if not failed:
            print("Failed to refit!")
            exit(0)

            
global base_unet_engine
global controlnet_engine
base_unet_engine = None
controlnet_engine = None

root_path = "./extensions/models/Unet-trt"
controlunet_name = "xxx.trt"
controlunet_path = os.path.join(root_path, controlunet_name)

if os.path.exists(controlunet_path):
    base_unet_engine = TrtEngine(controlunet_path)
    base_unet_engine.load()
    base_unet_engine.activate(True)

    print(f"==== try to merge lora")
    # 加载lora
    refit_dict = merge_loras(
        os.path.join(root_path, "common_base.trt"),
        [os.path.join(root_path, "refit_lora.trt")],
        [1.0]
    )
    print(f"==== try to refit from dict")
    base_unet_engine.refit_from_dict(refit_dict)
    print(f"==== refit success")

else:
    print(f"**** controlunet_path: {controlunet_path} not exists")

controlnet_name = "yyyy.trt"
controlnet_path = os.path.join(root_path, controlnet_name)
if os.path.exists(controlnet_path):
    controlnet_engine = TrtEngine(controlnet_path)
    controlnet_engine.load()
    controlnet_engine.activate(True)
else:
    print(f"**** controlnet_path: {controlnet_path} not exists")
