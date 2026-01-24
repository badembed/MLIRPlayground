from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import structured
from mlir.execution_engine import ExecutionEngine
from mlir.passmanager import PassManager
from mlir.runtime.np_to_memref import get_ranked_memref_descriptor

from cuda.bindings import runtime as cudart

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


def create_gpu_pipeline(ctx: ir.Context) -> PassManager:
    with ctx:
        pm = PassManager("builtin.module")
        pm.add("func.func(llvm-request-c-wrappers)")
        pm.add("convert-linalg-to-parallel-loops")
        pm.add("func.func(gpu-map-parallel-loops)")
        pm.add("func.func(convert-parallel-loops-to-gpu)")
        pm.add("gpu-kernel-outlining")
        pm.add("gpu-lower-to-nvvm-pipeline")

        pm.add("reconcile-unrealized-casts")
        pm.add("canonicalize")
        pm.add("cse")
        pm.add("gpu-to-llvm")
    return pm


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


def retarget_memref_to_device(mref, dev_ptr):
    """
    Overwrite mref.allocated / mref.aligned with a device pointer,
    keeping offset/sizes/strides unchanged.
    """
    MemRefType = type(mref)

    alloc_type = None
    aligned_type = None
    for name, ctype_ in MemRefType._fields_:
        if name == "allocated":
            alloc_type = ctype_
        elif name == "aligned":
            aligned_type = ctype_

    if alloc_type is None or aligned_type is None:
        raise RuntimeError("memref descriptor is missing 'allocated' or 'aligned' fields")

    dev_addr = int(dev_ptr)

    def as_field_value(field_type, addr: int):
        if issubclass(field_type, ctypes._Pointer):
            return ctypes.cast(ctypes.c_void_p(addr), field_type)
        if field_type is ctypes.c_void_p:
            return ctypes.c_void_p(addr)
        return field_type(addr)

    mref.allocated = as_field_value(alloc_type, dev_addr)
    mref.aligned = as_field_value(aligned_type, dev_addr)
    return mref


def main():
    ctx = ir.Context()
    kernel = create_kernel(ctx)

    pm = create_gpu_pipeline(ctx)
    pm.run(kernel.operation)

    mlir_libs = [
        "/home/alex/sources/ML/mlirPython2/.venv/lib/python3.12/site-packages/mlir/lib/libmlir_cuda_runtime.so"
    ]
    eng = ExecutionEngine(kernel, shared_libs=mlir_libs)
    eng.initialize()
    add_func = eng.lookup("add")

        # CPU buffers
    a = np.ascontiguousarray(np.array([1,1,1,1], dtype=np.float32))
    b = np.ascontiguousarray(np.array([1,1,1,1], dtype=np.float32))
    out = np.ascontiguousarray(np.zeros(4, dtype=np.float32))

    a_memref = get_ranked_memref_descriptor(a)
    b_memref = get_ranked_memref_descriptor(b)
    out_memref = get_ranked_memref_descriptor(out)

    # GPU buffers
    err, d_a = cudart.cudaMalloc(a.nbytes)
    assert err == cudart.cudaError_t.cudaSuccess

    err, d_b = cudart.cudaMalloc(b.nbytes)
    assert err == cudart.cudaError_t.cudaSuccess

    err, d_out = cudart.cudaMalloc(out.nbytes)
    assert err == cudart.cudaError_t.cudaSuccess

    (err,) = cudart.cudaMemcpy(
        d_a, a.ctypes.data, a.nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
    )
    assert err == cudart.cudaError_t.cudaSuccess

    (err,) = cudart.cudaMemcpy(
        d_b, b.ctypes.data, b.nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
    )
    assert err == cudart.cudaError_t.cudaSuccess

    (err,) = cudart.cudaMemcpy(
        d_out, out.ctypes.data, out.nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
    )
    assert err == cudart.cudaError_t.cudaSuccess

    # Retarget memrefs to device pointers (in-place)
    retarget_memref_to_device(a_memref, d_a)
    retarget_memref_to_device(b_memref, d_b)
    retarget_memref_to_device(out_memref, d_out)

    # Pack args for MLIR JIT
    args = memref.to_packed_args([a_memref, b_memref, out_memref])
    add_func(args)

    # Copy result back to host
    (err,) = cudart.cudaMemcpy(
        out.ctypes.data, d_out, out.nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
    )
    assert err == cudart.cudaError_t.cudaSuccess

    print("out =", out)


if __name__ == "__main__":
    main()
