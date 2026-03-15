from mlir import ir
from mlir.execution_engine import ExecutionEngine
from mlir.passmanager import PassManager
from mlir.runtime.np_to_memref import get_ranked_memref_descriptor

import numpy as np
from pathlib import Path
import struct

from utils import memref


def create_kernel(ctx: ir.Context) -> ir.Module:
    with ctx:
        module = ir.Module.parse(
            r"""
module {
  // compute K^T for each batch/head.
  func.func private @kernel_k_transpose(
      %arg0: memref<2x5x8xf32, strided<[40, 8, 1], offset: 0>>,
      %arg1: memref<2x8x5xf32>) {
    linalg.transpose
      ins(%arg0 : memref<2x5x8xf32, strided<[40, 8, 1], offset: 0>>)
      outs(%arg1 : memref<2x8x5xf32>)
      permutation = [0, 2, 1]
    return
  }

  // compute scores = Q * K^T.
  func.func private @kernel_qk_t(
      %arg0: memref<2x5x8xf32, strided<[40, 8, 1], offset: 0>>,
      %arg1: memref<2x8x5xf32>,
      %arg2: memref<2x5x5xf32, strided<[25, 5, 1], offset: 0>>) {
    %cst0 = arith.constant 0.0 : f32
    linalg.fill ins(%cst0 : f32) outs(%arg2 : memref<2x5x5xf32, strided<[25, 5, 1], offset: 0>>)
    linalg.batch_matmul
      ins(%arg0, %arg1 : memref<2x5x8xf32, strided<[40, 8, 1], offset: 0>>, memref<2x8x5xf32>)
      outs(%arg2 : memref<2x5x5xf32, strided<[25, 5, 1], offset: 0>>)
    return
  }

  // probs = softmax(scores * scale) along last dimension.
  func.func private @kernel_softmax(
      %arg0: memref<2x5x5xf32, strided<[25, 5, 1], offset: 0>>,
      %arg1: memref<2x5x5xf32, strided<[25, 5, 1], offset: 0>>) {
    %cst_scale = arith.constant 1.250000e-01 : f32
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%arg0 : memref<2x5x5xf32, strided<[25, 5, 1], offset: 0>>) outs(%arg1 : memref<2x5x5xf32, strided<[25, 5, 1], offset: 0>>) {
    ^bb0(%in: f32, %_out: f32):
      %scaled = arith.mulf %in, %cst_scale : f32
      linalg.yield %scaled : f32
    }

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cb = arith.constant 2 : index
    %cs = arith.constant 5 : index
    %neg_inf = arith.constant -3.4028235E+38 : f32
    %f32_zero = arith.constant 0.0 : f32
    scf.for %b = %c0 to %cb step %c1 {
      scf.for %i = %c0 to %cs step %c1 {
        %max = scf.for %j = %c0 to %cs step %c1 iter_args(%m = %neg_inf) -> (f32) {
          %v = memref.load %arg1[%b, %i, %j] : memref<2x5x5xf32, strided<[25, 5, 1], offset: 0>>
          %m2 = arith.maximumf %m, %v : f32
          scf.yield %m2 : f32
        }
        %sum = scf.for %j = %c0 to %cs step %c1 iter_args(%s = %f32_zero) -> (f32) {
          %v = memref.load %arg1[%b, %i, %j] : memref<2x5x5xf32, strided<[25, 5, 1], offset: 0>>
          %x = arith.subf %v, %max : f32
          %e = math.exp %x : f32
          %s2 = arith.addf %s, %e : f32
          scf.yield %s2 : f32
        }
        scf.for %j = %c0 to %cs step %c1 {
          %v = memref.load %arg1[%b, %i, %j] : memref<2x5x5xf32, strided<[25, 5, 1], offset: 0>>
          %x = arith.subf %v, %max : f32
          %e = math.exp %x : f32
          %y = arith.divf %e, %sum : f32
          memref.store %y, %arg1[%b, %i, %j] : memref<2x5x5xf32, strided<[25, 5, 1], offset: 0>>
        }
      }
    }
    return
  }

  // Final projection: O = probs * V.
  func.func private @kernel_pv(
      %arg0: memref<2x5x5xf32, strided<[25, 5, 1], offset: 0>>,
      %arg1: memref<2x5x8xf32, strided<[40, 8, 1], offset: 0>>,
      %arg2: memref<2x5x8xf32, strided<[40, 8, 1], offset: 0>>) {
    %cst0 = arith.constant 0.0 : f32
    linalg.fill ins(%cst0 : f32) outs(%arg2 : memref<2x5x8xf32, strided<[40, 8, 1], offset: 0>>)
    linalg.batch_matmul
      ins(%arg0, %arg1 : memref<2x5x5xf32, strided<[25, 5, 1], offset: 0>>, memref<2x5x8xf32, strided<[40, 8, 1], offset: 0>>)
      outs(%arg2 : memref<2x5x8xf32, strided<[40, 8, 1], offset: 0>>)
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
    %k_t = memref.alloc() : memref<2x8x5xf32>
    %q3 = memref.subview %view_q[0, 0, 0, 0] [1, 2, 5, 8] [1, 1, 1, 1]
      : memref<1x2x5x8xf32> to memref<2x5x8xf32, strided<[40, 8, 1], offset: 0>>
    %k3 = memref.subview %view_k[0, 0, 0, 0] [1, 2, 5, 8] [1, 1, 1, 1]
      : memref<1x2x5x8xf32> to memref<2x5x8xf32, strided<[40, 8, 1], offset: 0>>
    %v3 = memref.subview %view_v[0, 0, 0, 0] [1, 2, 5, 8] [1, 1, 1, 1]
      : memref<1x2x5x8xf32> to memref<2x5x8xf32, strided<[40, 8, 1], offset: 0>>
    %scores3 = memref.subview %view_scores[0, 0, 0, 0] [1, 2, 5, 5] [1, 1, 1, 1]
      : memref<1x2x5x5xf32> to memref<2x5x5xf32, strided<[25, 5, 1], offset: 0>>
    %probs3 = memref.subview %view_probs[0, 0, 0, 0] [1, 2, 5, 5] [1, 1, 1, 1]
      : memref<1x2x5x5xf32> to memref<2x5x5xf32, strided<[25, 5, 1], offset: 0>>
    %out3 = memref.subview %view_out[0, 0, 0, 0] [1, 2, 5, 8] [1, 1, 1, 1]
      : memref<1x2x5x8xf32> to memref<2x5x8xf32, strided<[40, 8, 1], offset: 0>>

    func.call @copy_kernel_0(%arg0, %view_q)
      : (memref<1x2x5x8xf32>, memref<1x2x5x8xf32>) -> ()
    func.call @copy_kernel_1(%arg1, %view_k)
      : (memref<1x2x5x8xf32>, memref<1x2x5x8xf32>) -> ()
    func.call @copy_kernel_2(%arg2, %view_v)
      : (memref<1x2x5x8xf32>, memref<1x2x5x8xf32>) -> ()

    func.call @kernel_k_transpose(%k3, %k_t)
      : (memref<2x5x8xf32, strided<[40, 8, 1], offset: 0>>, memref<2x8x5xf32>) -> ()
    func.call @kernel_qk_t(%q3, %k_t, %scores3)
      : (memref<2x5x8xf32, strided<[40, 8, 1], offset: 0>>, memref<2x8x5xf32>, memref<2x5x5xf32, strided<[25, 5, 1], offset: 0>>) -> ()
    func.call @kernel_softmax(%scores3, %probs3)
      : (memref<2x5x5xf32, strided<[25, 5, 1], offset: 0>>, memref<2x5x5xf32, strided<[25, 5, 1], offset: 0>>) -> ()
    func.call @kernel_pv(%probs3, %v3, %out3)
      : (memref<2x5x5xf32, strided<[25, 5, 1], offset: 0>>, memref<2x5x8xf32, strided<[40, 8, 1], offset: 0>>, memref<2x5x8xf32, strided<[40, 8, 1], offset: 0>>) -> ()

    func.call @copy_kernel_3(%view_out, %arg3)
      : (memref<1x2x5x8xf32>, memref<1x2x5x8xf32>) -> ()

    memref.dealloc %view_q : memref<1x2x5x8xf32>
    memref.dealloc %view_k : memref<1x2x5x8xf32>
    memref.dealloc %view_v : memref<1x2x5x8xf32>
    memref.dealloc %view_scores : memref<1x2x5x5xf32>
    memref.dealloc %view_probs : memref<1x2x5x5xf32>
    memref.dealloc %view_out : memref<1x2x5x8xf32>
    memref.dealloc %k_t : memref<2x8x5xf32>
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
        pm.add("expand-strided-metadata")
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


def write_tensor_binary(path: Path, arr: np.ndarray) -> None:
    # Format:
    # [dtype: int32][num_dims: int64][dim_sizes: int64 * rank][raw data]
    # dtype IDs follow the user's script mapping: FP32 -> 1.
    arr = np.ascontiguousarray(arr)
    if arr.dtype != np.float32:
        raise ValueError(f"Expected float32 tensor, got {arr.dtype}")
    dtype_id = 1  # FP32
    rank = arr.ndim
    with path.open("wb") as f:
        f.write(struct.pack("<i", dtype_id))
        f.write(struct.pack("<q", rank))
        for d in arr.shape:
            f.write(struct.pack("<q", int(d)))
        f.write(arr.tobytes(order="C"))


def main():
    ctx = ir.Context()
    kernel = create_kernel(ctx)

    rng = np.random.default_rng(42)
    q = np.ascontiguousarray(rng.normal(loc=0.2, scale=1.1, size=(1, 2, 5, 8)).astype(np.float32))
    k = np.ascontiguousarray(rng.normal(loc=-0.1, scale=0.9, size=(1, 2, 5, 8)).astype(np.float32))
    v = np.ascontiguousarray(rng.normal(loc=0.05, scale=1.3, size=(1, 2, 5, 8)).astype(np.float32))
    out = np.ascontiguousarray(np.zeros((1, 2, 5, 8), dtype=np.float32))

    bin_dir = Path("playground/generated_bins")
    bin_dir.mkdir(parents=True, exist_ok=True)
    q_path = bin_dir / "q.bin"
    k_path = bin_dir / "k.bin"
    v_path = bin_dir / "v.bin"
    out_path = bin_dir / "out.bin"
    write_tensor_binary(q_path, q)
    write_tensor_binary(k_path, k)
    write_tensor_binary(v_path, v)

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
    write_tensor_binary(out_path, out)
    print("q_bin =", q_path)
    print("k_bin =", k_path)
    print("v_bin =", v_path)
    print("out_bin =", out_path)


if __name__ == "__main__":
    main()
