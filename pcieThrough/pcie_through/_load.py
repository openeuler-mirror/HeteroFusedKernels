import os
import pathlib
from functools import wraps, lru_cache
import torch
import torch_npu


def _load_opextension_so():
    npu_path = pathlib.Path(__file__).parents[0]
    _so_path = os.path.join(npu_path, 'lib', 'libpcie_through.so')
    torch.ops.load_library(_so_path)