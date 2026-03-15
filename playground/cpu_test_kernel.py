from mlir import ir
from mlir.execution_engine import ExecutionEngine
from mlir.passmanager import PassManager
from mlir.runtime.np_to_memref import get_ranked_memref_descriptor

import numpy as np

from utils import memref


def create_kernel(ctx: ir.Context) -> ir.Module:
    with ctx:
        module = ir.Module.parse(
            r"""
module {
  func.func private @outlined_group_atomic_kernel_convolution_0(
      %arg0: memref<1x3x64x64xf16>,
      %arg1: memref<64x3x1x1xf16>,
      %arg2: memref<1x64x64x64xf16>) {
    %cst = arith.constant 0.0 : f16
    linalg.fill ins(%cst : f16) outs(%arg2 : memref<1x64x64x64xf16>)
    linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%arg0, %arg1 : memref<1x3x64x64xf16>, memref<64x3x1x1xf16>)
      outs(%arg2 : memref<1x64x64x64xf16>)
    return
  }

  func.func @forward(
      %arg0: memref<1x3x64x64xf16>,
      %arg1: memref<64x3x1x1xf16>,
      %arg2: memref<1x64x64x64xf16>) {
    %buf_in = memref.alloc() : memref<1x3x64x64xf16>
    %buf_w = memref.alloc() : memref<64x3x1x1xf16>
    %buf_out = memref.alloc() : memref<1x64x64x64xf16>

    func.call @copy_kernel_0(%arg0, %buf_in)
      : (memref<1x3x64x64xf16>, memref<1x3x64x64xf16>) -> ()
    func.call @copy_kernel_1(%arg1, %buf_w)
      : (memref<64x3x1x1xf16>, memref<64x3x1x1xf16>) -> ()
    func.call @outlined_group_atomic_kernel_convolution_0(%buf_in, %buf_w, %buf_out)
      : (memref<1x3x64x64xf16>, memref<64x3x1x1xf16>, memref<1x64x64x64xf16>) -> ()
    func.call @copy_kernel_2(%buf_out, %arg2)
      : (memref<1x64x64x64xf16>, memref<1x64x64x64xf16>) -> ()

    memref.dealloc %buf_in : memref<1x3x64x64xf16>
    memref.dealloc %buf_w : memref<64x3x1x1xf16>
    memref.dealloc %buf_out : memref<1x64x64x64xf16>
    return
  }

  func.func private @copy_kernel_0(%arg0: memref<1x3x64x64xf16>, %arg1: memref<1x3x64x64xf16>) {
    memref.copy %arg0, %arg1 : memref<1x3x64x64xf16> to memref<1x3x64x64xf16>
    return
  }

  func.func private @copy_kernel_1(%arg0: memref<64x3x1x1xf16>, %arg1: memref<64x3x1x1xf16>) {
    memref.copy %arg0, %arg1 : memref<64x3x1x1xf16> to memref<64x3x1x1xf16>
    return
  }

  func.func private @copy_kernel_2(%arg0: memref<1x64x64x64xf16>, %arg1: memref<1x64x64x64xf16>) {
    memref.copy %arg0, %arg1 : memref<1x64x64x64xf16> to memref<1x64x64x64xf16>
    return
  }
}
"""
        )
    return module


def create_jit_pipeline(ctx: ir.Context) -> PassManager:
    with ctx:
        pm = PassManager("builtin.module")
        pm.add("func.func(llvm-request-c-wrappers)")
        pm.add("convert-linalg-to-loops")
        pm.add("lower-affine")
        pm.add("convert-scf-to-cf")
        pm.add("convert-to-llvm")
        pm.add("reconcile-unrealized-casts")
        pm.add("canonicalize")
        pm.add("cse")
    return pm


def main():
    ctx = ir.Context()
    kernel = create_kernel(ctx)

    inp = np.ascontiguousarray(np.ones((1, 3, 64, 64), dtype=np.float16))
    weight = np.ascontiguousarray(np.ones((64, 3, 1, 1), dtype=np.float16))
    out = np.ascontiguousarray(np.zeros((1, 64, 64, 64), dtype=np.float16))

    pm = create_jit_pipeline(ctx)
    pm.run(kernel.operation)

    eng = ExecutionEngine(kernel)
    eng.initialize()
    forward = eng.lookup("forward")

    inp_memref = get_ranked_memref_descriptor(inp)
    weight_memref = get_ranked_memref_descriptor(weight)
    out_memref = get_ranked_memref_descriptor(out)
    args = memref.to_packed_args([inp_memref, weight_memref, out_memref])
    forward(args)

    print("out shape =", out.shape)
    print("out[0,0,0,0:8] =", out[0, 0, 0, 0:8])
    print("mean =", float(out.mean()))
    print("allclose_to_3 =", bool(np.allclose(out, 3.0)))


if __name__ == "__main__":
    main()
