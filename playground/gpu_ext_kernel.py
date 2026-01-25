from mlir import ir
from mlir.dialects import transform
from mlir.dialects import gpu, builtin
from mlir.dialects.transform import structured
from mlir.execution_engine import ExecutionEngine
from mlir.passmanager import PassManager
from mlir.runtime.np_to_memref import get_ranked_memref_descriptor

from cuda.bindings import runtime as cudart
from cuda.bindings import driver as cudadrv

import argparse
import ctypes
from importlib import resources
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


def extract_cubin_from_module(module: ir.Module) -> bytes:
    # Walk top-level ops in the module body
    for op in module.body.operations:
        if isinstance(op, gpu.BinaryOp):
            # gpu.binary @name [#gpu.object<...>, ...]
            obj0_attr = op.objects[0]        # generic Attribute
            obj0 = gpu.ObjectAttr(obj0_attr) # typed wrapper

            blob = obj0.object  # in your build this is already bytes

            if isinstance(blob, bytes):
                return blob
            elif isinstance(blob, str):
                # In some builds it's a str; encode as raw bytes
                return blob.encode("latin1")
            else:
                raise TypeError(f"Unexpected gpu.ObjectAttr.object type: {type(blob)}")

    raise RuntimeError("No gpu.binary op found in module")


def launch_add_kernel_from_cubin(
    cubin_path: str,
    a_memref,
    b_memref,
    out_memref,
    n_elements: int,
):
    # 1. Read cubin
    with open(cubin_path, "rb") as f:
        cubin_bytes = f.read()

    # 2. Initialize driver
    (err,) = cudadrv.cuInit(0)
    assert err == cudadrv.CUresult.CUDA_SUCCESS, f"cuInit failed: {err}"

    # 3. Load module from cubin
    err, module = cudadrv.cuModuleLoadData(cubin_bytes)
    assert err == cudadrv.CUresult.CUDA_SUCCESS, f"cuModuleLoadData failed: {err}"

    # 4. Get kernel function by name (@add_kernel in your IR)
    err, func = cudadrv.cuModuleGetFunction(module, b"add_kernel")
    assert err == cudadrv.CUresult.CUDA_SUCCESS, f"cuModuleGetFunction failed: {err}"

    # 5. Prepare kernel arguments using HelperKernelParams style.
    #    The GPU kernel expects expanded memref descriptors:
    #    (allocated, aligned, offset, sizes..., strides...) per memref.
    kernel_arg_values = []
    kernel_arg_types = []

    def append_scalar(value):
        kernel_arg_values.append(int(value))
        kernel_arg_types.append(ctypes.c_int64)

    def append_ptr(value):
        if isinstance(value, ctypes._Pointer):
            addr = ctypes.cast(value, ctypes.c_void_p).value
        else:
            addr = int(value)
        kernel_arg_values.append(addr)
        kernel_arg_types.append(ctypes.c_void_p)

    def expand_memref_kernel_args(mref):
        append_ptr(mref.allocated)
        append_ptr(mref.aligned)
        append_scalar(mref.offset)
        for i in range(len(mref.shape)):
            append_scalar(mref.shape[i])
        for i in range(len(mref.strides)):
            append_scalar(mref.strides[i])

    append_scalar(a_memref.strides[0])
    append_scalar(a_memref.offset)
    expand_memref_kernel_args(a_memref)
    expand_memref_kernel_args(b_memref)
    expand_memref_kernel_args(out_memref)

    kernel_args = (
        tuple(kernel_arg_values),
        tuple(kernel_arg_types),
    )

    # 6. Grid / block dims
    block_x = 1
    grid_x = n_elements

    # 7. Launch the kernel
    err, = cudadrv.cuLaunchKernel(
        func,
        grid_x, 1, 1,      # grid dim
        block_x, 1, 1,     # block dim
        0,                 # sharedMemBytes
        0,                 # stream (0 = default stream)
        kernel_args,       # kernelParams (HelperKernelParams)
        0,                 # extra (must be an integer, not None)
    )
    assert err == cudadrv.CUresult.CUDA_SUCCESS, f"cuLaunchKernel failed: {err}"



def create_gpu_pipeline(ctx: ir.Context) -> PassManager:
    with ctx:
        pm = PassManager("builtin.module")
        pm.add("func.func(llvm-request-c-wrappers)")
        pm.add("convert-linalg-to-parallel-loops")
        pm.add("func.func(gpu-map-parallel-loops)")
        pm.add("func.func(convert-parallel-loops-to-gpu)")
        pm.add("gpu-kernel-outlining")
        pm.add("gpu-lower-to-nvvm-pipeline")

        #pm.add("gpu.module(convert-gpu-to-nvvm)")
        #pm.add("gpu-module-to-binary")

        # pm.add("reconcile-unrealized-casts")
        # pm.add("canonicalize")
        # pm.add("cse")
        # pm.add("gpu-to-llvm")
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


def prepare_gpu_buffers(a, b, out):
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

    return d_a, d_b, d_out


def main():
    ctx = ir.Context()
    kernel = create_kernel(ctx)

    # CPU buffers
    a = np.ascontiguousarray(np.array([1,1,1,1], dtype=np.float32))
    b = np.ascontiguousarray(np.array([1,1,1,1], dtype=np.float32))
    out = np.ascontiguousarray(np.zeros(4, dtype=np.float32))

    pm = create_gpu_pipeline(ctx)
    pm.run(kernel.operation)
    print(kernel)

    cubin = extract_cubin_from_module(kernel)
    with open("add_sm80.cubin", "wb") as f:
        f.write(cubin)

    # --- RUNTIME STEP: allocate device buffers, prepare memrefs ---
    d_a, d_b, d_out = prepare_gpu_buffers(a, b, out)

    a_memref = get_ranked_memref_descriptor(a)
    b_memref = get_ranked_memref_descriptor(b)
    out_memref = get_ranked_memref_descriptor(out)

    # Retarget memrefs to device pointers (in-place)
    retarget_memref_to_device(a_memref, d_a)
    retarget_memref_to_device(b_memref, d_b)
    retarget_memref_to_device(out_memref, d_out)

    # --- Launch kernel via CUDA driver using the cubin ---
    cubin_path = "/home/alex/sources/ML/mlirPython2/playground/add_sm80.cubin"
    launch_add_kernel_from_cubin(
        cubin_path,
        a_memref,
        b_memref,
        out_memref,
        n_elements=4,
    )

    # --- Copy back result ---
    (err,) = cudart.cudaMemcpy(
        out.ctypes.data, d_out, out.nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
    )
    assert err == cudart.cudaError_t.cudaSuccess

    print("out =", out)


if __name__ == "__main__":
    main()
