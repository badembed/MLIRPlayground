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
  func.func private @outlined_group_atomic_kernel_sdpa_0(
      %arg0: memref<8x128x64xf16>,
      %arg1: memref<8x64x128xf16>,
      %arg2: memref<8x128x64xf16>,
      %arg3: memref<8x128x128xf16>,
      %arg4: memref<8x128x128xf16>,
      %arg5: memref<8x128x64xf16>) {
    %cst0 = arith.constant 0.0 : f16
    %cst_scale = arith.constant 1.250000e-01 : f16

    linalg.fill ins(%cst0 : f16) outs(%arg3 : memref<8x128x128xf16>)
    linalg.batch_matmul
      ins(%arg0, %arg1 : memref<8x128x64xf16>, memref<8x64x128xf16>)
      outs(%arg3 : memref<8x128x128xf16>)

    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%arg3 : memref<8x128x128xf16>) outs(%arg4 : memref<8x128x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      %scaled = arith.mulf %in, %cst_scale : f16
      linalg.yield %scaled : f16
    }

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c128 = arith.constant 128 : index
    %neg_inf = arith.constant -65504.0 : f16
    %f32_zero = arith.constant 0.0 : f32
    scf.for %b = %c0 to %c8 step %c1 {
      scf.for %i = %c0 to %c128 step %c1 {
        %max = scf.for %j = %c0 to %c128 step %c1 iter_args(%m = %neg_inf) -> (f16) {
          %v = memref.load %arg4[%b, %i, %j] : memref<8x128x128xf16>
          %m2 = arith.maximumf %m, %v : f16
          scf.yield %m2 : f16
        }
        %max32 = arith.extf %max : f16 to f32
        %sum = scf.for %j = %c0 to %c128 step %c1 iter_args(%s = %f32_zero) -> (f32) {
          %v = memref.load %arg4[%b, %i, %j] : memref<8x128x128xf16>
          %v32 = arith.extf %v : f16 to f32
          %x = arith.subf %v32, %max32 : f32
          %e = math.exp %x : f32
          %s2 = arith.addf %s, %e : f32
          scf.yield %s2 : f32
        }
        scf.for %j = %c0 to %c128 step %c1 {
          %v = memref.load %arg4[%b, %i, %j] : memref<8x128x128xf16>
          %v32 = arith.extf %v : f16 to f32
          %x = arith.subf %v32, %max32 : f32
          %e = math.exp %x : f32
          %y = arith.divf %e, %sum : f32
          %y16 = arith.truncf %y : f32 to f16
          memref.store %y16, %arg4[%b, %i, %j] : memref<8x128x128xf16>
        }
      }
    }

    linalg.fill ins(%cst0 : f16) outs(%arg5 : memref<8x128x64xf16>)
    linalg.batch_matmul
      ins(%arg4, %arg2 : memref<8x128x128xf16>, memref<8x128x64xf16>)
      outs(%arg5 : memref<8x128x64xf16>)
    return
  }

  func.func @forward(
      %arg0: memref<8x128x64xf16>,
      %arg1: memref<8x64x128xf16>,
      %arg2: memref<8x128x64xf16>,
      %arg3: memref<8x128x64xf16>) {
    %view_q = memref.alloc() : memref<8x128x64xf16>
    %view_k = memref.alloc() : memref<8x64x128xf16>
    %view_v = memref.alloc() : memref<8x128x64xf16>
    %view_scores = memref.alloc() : memref<8x128x128xf16>
    %view_probs = memref.alloc() : memref<8x128x128xf16>
    %view_out = memref.alloc() : memref<8x128x64xf16>

    func.call @copy_kernel_0(%arg0, %view_q)
      : (memref<8x128x64xf16>, memref<8x128x64xf16>) -> ()
    func.call @copy_kernel_1(%arg1, %view_k)
      : (memref<8x64x128xf16>, memref<8x64x128xf16>) -> ()
    func.call @copy_kernel_2(%arg2, %view_v)
      : (memref<8x128x64xf16>, memref<8x128x64xf16>) -> ()

    func.call @outlined_group_atomic_kernel_sdpa_0(
      %view_q, %view_k, %view_v, %view_scores, %view_probs, %view_out)
      : (
        memref<8x128x64xf16>, memref<8x64x128xf16>, memref<8x128x64xf16>,
        memref<8x128x128xf16>, memref<8x128x128xf16>, memref<8x128x64xf16>) -> ()

    func.call @copy_kernel_3(%view_out, %arg3)
      : (memref<8x128x64xf16>, memref<8x128x64xf16>) -> ()

    memref.dealloc %view_q : memref<8x128x64xf16>
    memref.dealloc %view_k : memref<8x64x128xf16>
    memref.dealloc %view_v : memref<8x128x64xf16>
    memref.dealloc %view_scores : memref<8x128x128xf16>
    memref.dealloc %view_probs : memref<8x128x128xf16>
    memref.dealloc %view_out : memref<8x128x64xf16>
    return
  }

  func.func private @copy_kernel_0(%arg0: memref<8x128x64xf16>, %arg1: memref<8x128x64xf16>) {
    memref.copy %arg0, %arg1 : memref<8x128x64xf16> to memref<8x128x64xf16>
    return
  }

  func.func private @copy_kernel_1(%arg0: memref<8x64x128xf16>, %arg1: memref<8x64x128xf16>) {
    memref.copy %arg0, %arg1 : memref<8x64x128xf16> to memref<8x64x128xf16>
    return
  }

  func.func private @copy_kernel_2(%arg0: memref<8x128x64xf16>, %arg1: memref<8x128x64xf16>) {
    memref.copy %arg0, %arg1 : memref<8x128x64xf16> to memref<8x128x64xf16>
    return
  }

  func.func private @copy_kernel_3(%arg0: memref<8x128x64xf16>, %arg1: memref<8x128x64xf16>) {
    memref.copy %arg0, %arg1 : memref<8x128x64xf16> to memref<8x128x64xf16>
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
        pm.add("convert-math-to-llvm")
        pm.add("convert-to-llvm")
        pm.add("reconcile-unrealized-casts")
        pm.add("canonicalize")
        pm.add("cse")
    return pm


def softmax_last_dim(x: np.ndarray) -> np.ndarray:
    x32 = x.astype(np.float32)
    x32 = x32 - np.max(x32, axis=-1, keepdims=True)
    ex = np.exp(x32)
    return (ex / np.sum(ex, axis=-1, keepdims=True)).astype(np.float32)


def main():
    ctx = ir.Context()
    kernel = create_kernel(ctx)

    q = np.ascontiguousarray(np.random.randn(8, 128, 64).astype(np.float16))
    k = np.ascontiguousarray(np.random.randn(8, 64, 128).astype(np.float16))
    v = np.ascontiguousarray(np.random.randn(8, 128, 64).astype(np.float16))
    out = np.ascontiguousarray(np.zeros((8, 128, 64), dtype=np.float16))

    pm = create_jit_pipeline(ctx)
    pm.run(kernel.operation)

    eng = ExecutionEngine(kernel)
    eng.initialize()
    forward = eng.lookup("forward")

    q_memref = get_ranked_memref_descriptor(q)
    k_memref = get_ranked_memref_descriptor(k)
    v_memref = get_ranked_memref_descriptor(v)
    out_memref = get_ranked_memref_descriptor(out)
    args = memref.to_packed_args([q_memref, k_memref, v_memref, out_memref])
    forward(args)

    # Reference in float32 for stability, then compare in float16 tolerance.
    scores = np.matmul(q.astype(np.float32), k.astype(np.float32)) * 0.125
    probs = softmax_last_dim(scores)
    ref = np.matmul(probs, v.astype(np.float32)).astype(np.float16)

    max_abs = float(np.max(np.abs(out.astype(np.float32) - ref.astype(np.float32))))
    print("out shape =", out.shape)
    print("out[0,0,0:8] =", out[0, 0, 0:8])
    print("ref[0,0,0:8] =", ref[0, 0, 0:8])
    print("max_abs_err =", max_abs)
    print("allclose =", bool(np.allclose(out, ref, rtol=1e-2, atol=1e-2)))


if __name__ == "__main__":
    main()
