"""
Triton Conv3d — Implicit GEMM approach.

Layout: NDHWC (channels-last) internally for coalesced memory access.
The public API accepts standard PyTorch NCDHW tensors and converts
automatically.

Algorithm
---------
Conv3d is reformulated as GEMM:

    Output[M, N_out] = InputCol[M, K] @ Weight[K, N_out]

where
    M     = batch * D_out * H_out * W_out   (output spatial positions)
    K     = (C_in / groups) * kD * kH * kW  (filter volume)
    N_out = C_out_per_group                 (output channels per group)

InputCol is never materialised; indices are computed on-the-fly inside the
kernel (implicit im2col).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Autotune configurations
# ---------------------------------------------------------------------------

_AUTOTUNE_CONFIGS = [
    # Small tiles for small problems / low channel counts
    triton.Config({"BLOCK_M": 32,  "BLOCK_N": 32,  "BLOCK_K": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32,  "BLOCK_K": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=4),
    # Medium tiles
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    # Large tiles for high channel counts
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=2),
]


# ---------------------------------------------------------------------------
# Forward kernel  (Implicit GEMM)
# ---------------------------------------------------------------------------

@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["M", "N_out", "K"])
@triton.jit
def _conv3d_fwd_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    kD, kH, kW,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    dil_d, dil_h, dil_w,
    C_in_per_group,
    C_out_per_group,
    M,
    N_out,
    K,
    stride_in_n, stride_in_d, stride_in_h, stride_in_w, stride_in_c,
    stride_wt_o, stride_wt_i,
    stride_out_n, stride_out_d, stride_out_h, stride_out_w, stride_out_c,
    HAS_BIAS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    group_id = tl.program_id(1)

    num_M_tiles = tl.cdiv(M, BLOCK_M)
    num_N_tiles = tl.cdiv(N_out, BLOCK_N)

    GROUP_SIZE: tl.constexpr = 8
    pid_m_group = pid // (GROUP_SIZE * num_N_tiles)
    first_pid_m = pid_m_group * GROUP_SIZE
    group_size_m = tl.minimum(num_M_tiles - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + ((pid % (GROUP_SIZE * num_N_tiles)) % group_size_m)
    pid_n = (pid % (GROUP_SIZE * num_N_tiles)) // group_size_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    HW_out = H_out * W_out
    DHW_out = D_out * HW_out

    n_idx = rm // DHW_out
    rem = rm % DHW_out
    d_out_idx = rem // HW_out
    rem2 = rem % HW_out
    h_out_idx = rem2 // W_out
    w_out_idx = rem2 % W_out

    HW_k = kH * kW
    DHW_k = kD * HW_k

    c_out_idx = rn + group_id * C_out_per_group

    # base addresses for input (per output position, no K component yet)
    base_in_addr = (
        n_idx * stride_in_n
        + group_id * C_in_per_group * stride_in_c
    )
    valid_m = rm < M

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)

        c_in_idx = rk // DHW_k
        k_rem = rk % DHW_k
        kd_idx = k_rem // HW_k
        k_rem2 = k_rem % HW_k
        kh_idx = k_rem2 // kW
        kw_idx = k_rem2 % kW

        d_in = d_out_idx[:, None] * stride_d - pad_d + kd_idx[None, :] * dil_d
        h_in = h_out_idx[:, None] * stride_h - pad_h + kh_idx[None, :] * dil_h
        w_in = w_out_idx[:, None] * stride_w - pad_w + kw_idx[None, :] * dil_w

        mask_a = (
            (d_in >= 0) & (d_in < D_in)
            & (h_in >= 0) & (h_in < H_in)
            & (w_in >= 0) & (w_in < W_in)
            & (rk[None, :] < K)
            & valid_m[:, None]
        )

        addr_a = (
            base_in_addr[:, None]
            + d_in * stride_in_d
            + h_in * stride_in_h
            + w_in * stride_in_w
            + c_in_idx[None, :] * stride_in_c
        )
        a = tl.load(input_ptr + addr_a, mask=mask_a, other=0.0)

        # Weight is contiguous (C_out, K_vol) — simple 2D indexing
        mask_b = (rk[:, None] < K) & (rn[None, :] < N_out)
        addr_b = c_out_idx[None, :] * stride_wt_o + rk[:, None] * stride_wt_i
        b = tl.load(weight_ptr + addr_b, mask=mask_b, other=0.0)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

    if HAS_BIAS:
        bias_idx = rn + group_id * C_out_per_group
        bias_vals = tl.load(bias_ptr + bias_idx, mask=rn < N_out, other=0.0)
        acc += bias_vals[None, :]

    c_out_store = rn + group_id * C_out_per_group
    out_addr = (
        n_idx[:, None] * stride_out_n
        + d_out_idx[:, None] * stride_out_d
        + h_out_idx[:, None] * stride_out_h
        + w_out_idx[:, None] * stride_out_w
        + c_out_store[None, :] * stride_out_c
    )
    mask_out = valid_m[:, None] & (rn[None, :] < N_out)
    tl.store(output_ptr + out_addr, acc.to(output_ptr.dtype.element_ty), mask=mask_out)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def _compute_output_size(in_size, kernel, stride, pad, dilation):
    return (in_size + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1


def _weight_to_gemm(w: torch.Tensor) -> torch.Tensor:
    """(C_out, C_in/g, kD, kH, kW) → (C_out, K_vol) as a view (already contiguous)."""
    C_out = w.shape[0]
    return w.reshape(C_out, -1)


def triton_conv3d_forward(
    input: torch.Tensor,      # NCDHW (contiguous)
    weight: torch.Tensor,     # (C_out, C_in/g, kD, kH, kW)
    bias: torch.Tensor | None,
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
    dilation: tuple[int, int, int],
    groups: int,
    allow_tf32: bool = False,
) -> torch.Tensor:
    batch, C_in, D_in, H_in, W_in = input.shape
    C_out, C_in_per_group, kD, kH, kW = weight.shape
    assert C_in == C_in_per_group * groups
    assert C_out % groups == 0

    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding
    dil_d, dil_h, dil_w = dilation

    D_out = _compute_output_size(D_in, kD, stride_d, pad_d, dil_d)
    H_out = _compute_output_size(H_in, kH, stride_h, pad_h, dil_h)
    W_out = _compute_output_size(W_in, kW, stride_w, pad_w, dil_w)

    inp = input.contiguous()
    w_2d = _weight_to_gemm(weight)

    # Allocate output directly in NCDHW
    output = torch.empty(
        (batch, C_out, D_out, H_out, W_out),
        device=input.device, dtype=input.dtype,
    )

    C_out_per_group = C_out // groups
    M = batch * D_out * H_out * W_out
    K = C_in_per_group * kD * kH * kW
    N_out = C_out_per_group

    # Pass raw strides — kernel uses them for scatter/gather addressing
    # Input is NCDHW contiguous: strides are (C_in*D*H*W, D*H*W, H*W, W, 1)
    # but we pass as (n, c, d, h, w) order matching kernel param names
    si_n = inp.stride(0)
    si_c = inp.stride(1)
    si_d = inp.stride(2)
    si_h = inp.stride(3)
    si_w = inp.stride(4)

    so_n = output.stride(0)
    so_c = output.stride(1)
    so_d = output.stride(2)
    so_h = output.stride(3)
    so_w = output.stride(4)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N_out, meta["BLOCK_N"]),
        groups,
    )
    _conv3d_fwd_kernel[grid](
        inp, w_2d,
        bias if bias is not None else inp,
        output,
        D_in, H_in, W_in,
        D_out, H_out, W_out,
        kD, kH, kW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dil_d, dil_h, dil_w,
        C_in_per_group, C_out_per_group,
        M, N_out, K,
        si_n, si_d, si_h, si_w, si_c,
        w_2d.stride(0), w_2d.stride(1),
        so_n, so_d, so_h, so_w, so_c,
        HAS_BIAS=(bias is not None),
        ALLOW_TF32=allow_tf32,
    )

    return output


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------

class _TritonConv3dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups, allow_tf32):
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        return triton_conv3d_forward(input, weight, bias, stride, padding, dilation, groups, allow_tf32)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # Fallback to PyTorch for backward (Phase 5 will replace with Triton kernels)
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv3d_input(
                input.shape, weight, grad_output,
                ctx.stride, ctx.padding, ctx.dilation, ctx.groups,
            )
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv3d_weight(
                input, weight.shape, grad_output,
                ctx.stride, ctx.padding, ctx.dilation, ctx.groups,
            )
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=[0, 2, 3, 4])

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


def triton_conv3d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] = 0,
    dilation: int | tuple[int, int, int] = 1,
    groups: int = 1,
    allow_tf32: bool = False,
) -> torch.Tensor:
    """Functional interface matching ``torch.nn.functional.conv3d``."""
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)
    return _TritonConv3dFunction.apply(input, weight, bias, stride, padding, dilation, groups, allow_tf32)


# ---------------------------------------------------------------------------
# nn.Module
# ---------------------------------------------------------------------------

class TritonConv3d(nn.Module):
    """Drop-in replacement for ``torch.nn.Conv3d`` using a Triton kernel."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)

        assert in_channels % groups == 0
        assert out_channels % groups == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        factory = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *kernel_size, **factory)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.weight[0].numel()
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return triton_conv3d(
            input, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, groups={self.groups}, "
            f"bias={self.bias is not None}"
        )
