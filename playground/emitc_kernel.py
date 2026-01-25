from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured
from mlir.execution_engine import ExecutionEngine
from mlir.passmanager import PassManager
from mlir.runtime.np_to_memref import get_ranked_memref_descriptor

import argparse
import ctypes
import numpy as np
from pathlib import Path
import subprocess
import tempfile

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


def create_strided_kernel(ctx: ir.Context) -> ir.Module:
    with ctx:
        module = ir.Module.parse(
            r"""
    // Compute element-wise addition.
    func.func @add(
            %a:   memref<2x3xf32, strided<[8, 2], offset: 4>>,
            %b:   memref<2x3xf32, strided<[8, 2], offset: 4>>,
            %out: memref<2x3xf32, strided<[8, 2], offset: 4>>
            ) {
              linalg.add
                ins(%a, %b : memref<2x3xf32, strided<[8, 2], offset: 4>>,
                             memref<2x3xf32, strided<[8, 2], offset: 4>>)
                outs(%out : memref<2x3xf32, strided<[8, 2], offset: 4>>)
              return
            }
"""
        )
    return module


def create_emitc_pipeline(ctx: ir.Context) -> PassManager:
    with ctx:
        pm = PassManager("builtin.module")

        pm.add("convert-linalg-to-loops")

        # pm.add("convert-scf-to-emitc")
        # pm.add("convert-memref-to-emitc")
        # pm.add("convert-arith-to-emitc")
        # pm.add("convert-func-to-emitc")
        pm.add("convert-to-emitc")

        pm.add("cse")
        pm.add("canonicalize")

    return pm


def emit_cpp_from_mlir_module(module):
    with tempfile.TemporaryDirectory() as tmpdir:
        mlir_path = Path(tmpdir) / "kernel_emitc.mlir"
        cpp_path = Path(tmpdir) / "kernel.cpp"

        with open(mlir_path, "w") as f:
            print(module, file=f)

        subprocess.check_call([
            "/home/alex/sources/ML/mlirPython2/.venv/lib/python3.10/site-packages/mlir/bin/mlir-translate",
            "-mlir-to-cpp",
            str(mlir_path),
            "-o", str(cpp_path),
        ])

        with open(cpp_path, "r") as f:
            return f.read()


def main():
    ctx = ir.Context()
    kernel = create_kernel(ctx)

    pm = create_emitc_pipeline(ctx)
    pm.run(kernel.operation)

    code = emit_cpp_from_mlir_module(kernel.operation)
    print(code)


if __name__ == "__main__":
    main()
