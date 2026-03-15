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
      %arg0: memref<1x2x5x8xf32>,
      %arg1: memref<1x2x5x8xf32>,
      %arg2: memref<1x2x5x8xf32>,
      %arg3: memref<1x2x5x5xf32>,
      %arg4: memref<1x2x5x5xf32>,
      %arg5: memref<1x2x5x8xf32>) {
    %cst0 = arith.constant 0.0 : f32
    %cst_scale = arith.constant 1.250000e-01 : f32

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cb = arith.constant 1 : index
    %ch = arith.constant 2 : index
    %cs = arith.constant 5 : index
    %cd = arith.constant 8 : index
    %neg_inf = arith.constant -3.4028235E+38 : f32
    %f32_zero = arith.constant 0.0 : f32
    scf.for %b = %c0 to %cb step %c1 {
      scf.for %h = %c0 to %ch step %c1 {
        scf.for %i = %c0 to %cs step %c1 {
          scf.for %j = %c0 to %cs step %c1 {
            %acc = scf.for %d = %c0 to %cd step %c1 iter_args(%sum0 = %f32_zero) -> (f32) {
              %q = memref.load %arg0[%b, %h, %i, %d] : memref<1x2x5x8xf32>
              %k = memref.load %arg1[%b, %h, %j, %d] : memref<1x2x5x8xf32>
              %p = arith.mulf %q, %k : f32
              %sum1 = arith.addf %sum0, %p : f32
              scf.yield %sum1 : f32
            }
            %scaled = arith.mulf %acc, %cst_scale : f32
            memref.store %scaled, %arg3[%b, %h, %i, %j] : memref<1x2x5x5xf32>
          }
          %max = scf.for %j = %c0 to %cs step %c1 iter_args(%m = %neg_inf) -> (f32) {
            %v = memref.load %arg3[%b, %h, %i, %j] : memref<1x2x5x5xf32>
            %m2 = arith.maximumf %m, %v : f32
            scf.yield %m2 : f32
          }
          %sum = scf.for %j = %c0 to %cs step %c1 iter_args(%s = %f32_zero) -> (f32) {
            %v = memref.load %arg3[%b, %h, %i, %j] : memref<1x2x5x5xf32>
            %x = arith.subf %v, %max : f32
            %e = math.exp %x : f32
            %s2 = arith.addf %s, %e : f32
            scf.yield %s2 : f32
          }
          scf.for %j = %c0 to %cs step %c1 {
            %v = memref.load %arg3[%b, %h, %i, %j] : memref<1x2x5x5xf32>
            %x = arith.subf %v, %max : f32
            %e = math.exp %x : f32
            %y = arith.divf %e, %sum : f32
            memref.store %y, %arg4[%b, %h, %i, %j] : memref<1x2x5x5xf32>
          }
          scf.for %d = %c0 to %cd step %c1 {
            %acc = scf.for %j = %c0 to %cs step %c1 iter_args(%sum0 = %f32_zero) -> (f32) {
              %p = memref.load %arg4[%b, %h, %i, %j] : memref<1x2x5x5xf32>
              %vv = memref.load %arg2[%b, %h, %j, %d] : memref<1x2x5x8xf32>
              %mul = arith.mulf %p, %vv : f32
              %sum1 = arith.addf %sum0, %mul : f32
              scf.yield %sum1 : f32
            }
            memref.store %acc, %arg5[%b, %h, %i, %d] : memref<1x2x5x8xf32>
          }
        }
      }
    }
    return
  }

  func.func @forward(
      %arg0: memref<1x2x5x8xf32>,
      %arg1: memref<1x2x5x8xf32>,
      %arg2: memref<1x2x5x8xf32>,
      %arg3: memref<1x2x5x8xf32>) {
    %view_q = memref.alloc() : memref<1x2x5x8xf32>
    %view_k = memref.alloc() : memref<1x2x5x8xf32>
    %view_v = memref.alloc() : memref<1x2x5x8xf32>
    %view_scores = memref.alloc() : memref<1x2x5x5xf32>
    %view_probs = memref.alloc() : memref<1x2x5x5xf32>
    %view_out = memref.alloc() : memref<1x2x5x8xf32>

    func.call @copy_kernel_0(%arg0, %view_q)
      : (memref<1x2x5x8xf32>, memref<1x2x5x8xf32>) -> ()
    func.call @copy_kernel_1(%arg1, %view_k)
      : (memref<1x2x5x8xf32>, memref<1x2x5x8xf32>) -> ()
    func.call @copy_kernel_2(%arg2, %view_v)
      : (memref<1x2x5x8xf32>, memref<1x2x5x8xf32>) -> ()

    func.call @outlined_group_atomic_kernel_sdpa_0(
      %view_q, %view_k, %view_v, %view_scores, %view_probs, %view_out)
      : (
        memref<1x2x5x8xf32>, memref<1x2x5x8xf32>, memref<1x2x5x8xf32>,
        memref<1x2x5x5xf32>, memref<1x2x5x5xf32>, memref<1x2x5x8xf32>) -> ()

    func.call @copy_kernel_3(%view_out, %arg3)
      : (memref<1x2x5x8xf32>, memref<1x2x5x8xf32>) -> ()

    memref.dealloc %view_q : memref<1x2x5x8xf32>
    memref.dealloc %view_k : memref<1x2x5x8xf32>
    memref.dealloc %view_v : memref<1x2x5x8xf32>
    memref.dealloc %view_scores : memref<1x2x5x5xf32>
    memref.dealloc %view_probs : memref<1x2x5x5xf32>
    memref.dealloc %view_out : memref<1x2x5x8xf32>
    return
  }

  func.func private @copy_kernel_0(%arg0: memref<1x2x5x8xf32>, %arg1: memref<1x2x5x8xf32>) {
    memref.copy %arg0, %arg1 : memref<1x2x5x8xf32> to memref<1x2x5x8xf32>
    return
  }

  func.func private @copy_kernel_1(%arg0: memref<1x2x5x8xf32>, %arg1: memref<1x2x5x8xf32>) {
    memref.copy %arg0, %arg1 : memref<1x2x5x8xf32> to memref<1x2x5x8xf32>
    return
  }

  func.func private @copy_kernel_2(%arg0: memref<1x2x5x8xf32>, %arg1: memref<1x2x5x8xf32>) {
    memref.copy %arg0, %arg1 : memref<1x2x5x8xf32> to memref<1x2x5x8xf32>
    return
  }

  func.func private @copy_kernel_3(%arg0: memref<1x2x5x8xf32>, %arg1: memref<1x2x5x8xf32>) {
    memref.copy %arg0, %arg1 : memref<1x2x5x8xf32> to memref<1x2x5x8xf32>
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

    q = np.ascontiguousarray(np.ones((1, 2, 5, 8), dtype=np.float32))
    k = np.ascontiguousarray(np.ones((1, 2, 5, 8), dtype=np.float32))
    v = np.ascontiguousarray(np.ones((1, 2, 5, 8), dtype=np.float32))
    out = np.ascontiguousarray(np.zeros((1, 2, 5, 8), dtype=np.float32))

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

    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * 0.125
    probs = softmax_last_dim(scores)
    ref = np.matmul(probs, v)

    max_abs = float(np.max(np.abs(out - ref)))
    print("out shape =", out.shape)
    print("out[0,0,0,0:8] =", out[0, 0, 0, 0:8])
    print("ref[0,0,0,0:8] =", ref[0, 0, 0, 0:8])
    print("max_abs_err =", max_abs)
    print("allclose =", bool(np.allclose(out, ref, rtol=1e-5, atol=1e-5)))


if __name__ == "__main__":
    main()
