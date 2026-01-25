from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured
from mlir.execution_engine import ExecutionEngine
from mlir.passmanager import PassManager
from mlir.runtime.np_to_memref import get_ranked_memref_descriptor

from cuda.bindings import runtime as cudart

import argparse
import ctypes
import numpy as np

from utils import memref


def create_kernel(ctx: ir.Context) -> ir.Module:
    with ctx:
        module = ir.Module.parse(
            r"""
    // Compute element-wise addition.
    func.func @add(%a: memref<4xf32>, %b: memref<4xf32>, %out: memref<4xf32>) {
        linalg.add ins(%a, %b : memref<4xf32>, memref<4xf32>)
                   outs(%out : memref<4xf32>)
        return
    }
"""
        )
    return module


def simulate_linalg_add(a, b, out):
    np.add(a, b, out=out)


def main():
    ctx = ir.Context()
    kernel = create_kernel(ctx)

    # CPU buffers
    a = np.ascontiguousarray(np.array([1,1,1,1], dtype=np.float32))
    b = np.ascontiguousarray(np.array([1,1,1,1], dtype=np.float32))
    out = np.ascontiguousarray(np.zeros(4, dtype=np.float32))


    simulate_linalg_add(a, b, out)
    print("out =", out)
    return


if __name__ == "__main__":
    main()
