from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured
from mlir.execution_engine import ExecutionEngine
from mlir.passmanager import PassManager
from mlir.runtime.np_to_memref import get_ranked_memref_descriptor

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


def create_jit_pipeline(ctx: ir.Context) -> PassManager:
    """
    Lower linalg to LLVM dialect.
    """
    with ctx:
        pm = PassManager("builtin.module")

        pm.add("func.func(llvm-request-c-wrappers)")
        pm.add("convert-linalg-to-loops")
        pm.add("convert-scf-to-cf")
        pm.add("convert-to-llvm")
        pm.add("reconcile-unrealized-casts")
        pm.add("cse")
        pm.add("canonicalize")
    return pm


def main():
    ctx = ir.Context()
    kernel = create_kernel(ctx)

    # CPU buffers
    a = np.ascontiguousarray(np.array([1,1,1,1], dtype=np.float32))
    b = np.ascontiguousarray(np.array([1,1,1,1], dtype=np.float32))
    out = np.ascontiguousarray(np.zeros(4, dtype=np.float32))

    pm = create_jit_pipeline(ctx)
    pm.run(kernel.operation)

    eng = ExecutionEngine(kernel)
    eng.initialize()
    add_func = eng.lookup("add")

    a_memref = get_ranked_memref_descriptor(a)
    b_memref = get_ranked_memref_descriptor(b)
    out_memref = get_ranked_memref_descriptor(out)

    # Pack args for MLIR JIT
    args = memref.to_packed_args([a_memref, b_memref, out_memref])
    add_func(args)

    print("out =", out)


if __name__ == "__main__":
    main()
