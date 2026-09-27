"""Real Mojo GPU kernels for the ridge-regression math in linear_model.mojo.

This machine has no NVIDIA/AMD/Apple GPU, so `DeviceContext()` cannot run here --
Mojo's `mojo run`/`mojo build` fail outright the moment any code path
constructs one, even one gated behind a runtime `has_accelerator()` check,
because Mojo instantiates every reachable function at compile time regardless
of which branch runs (verified empirically against this Mojo 1.1.0 toolchain).
That is why this file is kept separate from src/train.mojo instead of being a
runtime-selected code path in the same binary.

On a CUDA/ROCm/Metal host, build and run this file directly to train with
these kernels instead of the CPU loop in linear_model.mojo -- see README.md.

Without a physical GPU, the kernels can still be *compile-checked* (but not
executed) by cross-compiling for a specific architecture, which does not
require the hardware to be present:

    pixi run mojo build --target-accelerator sm_80 -o /tmp/gpu_kernels src/gpu_kernels.mojo   # NVIDIA
    pixi run mojo build --target-accelerator gfx942 -o /tmp/gpu_kernels src/gpu_kernels.mojo  # AMD

Both were verified to compile successfully from this repo during development.
"""
from std.memory import Pointer
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from std.testing import assert_almost_equal

comptime dtype = DType.float32


def matvec_kernel(
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    x: Pointer[Scalar[dtype], MutAnyOrigin],
    w: Pointer[Scalar[dtype], MutAnyOrigin],
    n_rows: Int32,
    n_cols: Int32,
):
    """Out[i] = sum_j x[i, j] * w[j] -- one thread per row."""
    var i = block_dim.x * block_idx.x + thread_idx.x
    if i < Int(n_rows):
        var acc: Float32 = 0.0
        for j in range(Int(n_cols)):
            acc += x[unsafe_offset=i * Int(n_cols) + j] * w[unsafe_offset=j]
        out_ptr[unsafe_offset=i] = acc


def transpose_matvec_kernel(
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    x: Pointer[Scalar[dtype], MutAnyOrigin],
    r: Pointer[Scalar[dtype], MutAnyOrigin],
    n_rows: Int32,
    n_cols: Int32,
):
    """Out[j] = sum_i x[i, j] * r[i] -- one thread per feature (the gradient step)."""
    var j = block_dim.x * block_idx.x + thread_idx.x
    if j < Int(n_cols):
        var acc: Float32 = 0.0
        for i in range(Int(n_rows)):
            acc += x[unsafe_offset=i * Int(n_cols) + j] * r[unsafe_offset=i]
        out_ptr[unsafe_offset=j] = acc


def _launch_threads(n: Int) -> Tuple[Int, Int]:
    var threads = min(n, 256)
    var blocks = (n + threads - 1) // threads
    return (blocks, threads)


def main() raises:
    """Self-test: compares both kernels against a plain CPU reference."""
    var n_rows = 37
    var n_cols = 5

    with DeviceContext() as ctx:
        var x_buf = ctx.enqueue_create_buffer[dtype](n_rows * n_cols)
        var w_buf = ctx.enqueue_create_buffer[dtype](n_cols)
        var r_buf = ctx.enqueue_create_buffer[dtype](n_rows)
        var matvec_out = ctx.enqueue_create_buffer[dtype](n_rows)
        var grad_out = ctx.enqueue_create_buffer[dtype](n_cols)

        var expected_matvec = List[Float32](length=n_rows, fill=0.0)
        var expected_grad = List[Float32](length=n_cols, fill=0.0)

        with x_buf.map_to_host() as x_host, w_buf.map_to_host() as w_host, r_buf.map_to_host() as r_host:
            for j in range(n_cols):
                w_host[j] = Scalar[dtype](j + 1)
            for i in range(n_rows):
                r_host[i] = Scalar[dtype](i) * 0.1
                for j in range(n_cols):
                    x_host[i * n_cols + j] = Scalar[dtype]((i + j) % 7)

            for i in range(n_rows):
                var acc: Float32 = 0.0
                for j in range(n_cols):
                    acc += x_host[i * n_cols + j] * w_host[j]
                expected_matvec[i] = acc
            for j in range(n_cols):
                var acc: Float32 = 0.0
                for i in range(n_rows):
                    acc += x_host[i * n_cols + j] * r_host[i]
                expected_grad[j] = acc

        var mv_blocks: Int
        var mv_threads: Int
        mv_blocks, mv_threads = _launch_threads(n_rows)
        ctx.enqueue_function[matvec_kernel](
            matvec_out, x_buf, w_buf, Int32(n_rows), Int32(n_cols),
            grid_dim=mv_blocks, block_dim=mv_threads,
        )

        var g_blocks: Int
        var g_threads: Int
        g_blocks, g_threads = _launch_threads(n_cols)
        ctx.enqueue_function[transpose_matvec_kernel](
            grad_out, x_buf, r_buf, Int32(n_rows), Int32(n_cols),
            grid_dim=g_blocks, block_dim=g_threads,
        )

        ctx.synchronize()

        with matvec_out.map_to_host() as mv_host, grad_out.map_to_host() as g_host:
            for i in range(n_rows):
                assert_almost_equal(mv_host[i], expected_matvec[i])
            for j in range(n_cols):
                assert_almost_equal(g_host[j], expected_grad[j])

        print("gpu_kernels self-test passed:", n_rows, "x", n_cols)
