import time
from typing import List
from safetensors.numpy import load_file
import cupy as cp
import numpy as np


# TODO really not that efficient, but improves usability greatly
def merge_loras(base: str, loras: List[str], scales: List[str]):
    """
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


def apply_loras(base_path: str, loras: List[str], scales: List[str]) -> dict:
    if base_path.endswith(".onnx"):
        base_path = base_path.replace("onnx", "trt")

    refit_dict = merge_loras(base_path, loras, scales)

    return refit_dict