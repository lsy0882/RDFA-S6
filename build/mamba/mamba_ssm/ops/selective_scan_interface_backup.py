# Copyright (c) 2023, Tri Dao, Albert Gu.

import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

from einops import rearrange, repeat

from causal_conv1d import causal_conv1d_fn
import causal_conv1d_cuda
import selective_scan_cuda

class SelectiveScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out if not return_last_state else (out, last_state)
        else:
            ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
            out_z = rest[0]
            return out_z if not return_last_state else (out_z, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        if not ctx.has_z:
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            z = None
            out = None
        else:
            u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        # Here we just pass in None and dz will be allocated in the C++ code.
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus,
            False  # option to recompute out_z, not used here
        )
        dz = rest[0] if ctx.has_z else None
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC,
                dD if D is not None else None,
                dz,
                ddelta_bias if delta_bias is not None else None,
                None,
                None)


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                     return_last_state=False):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    """
    return SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)





# Parallel 3 (must use multi kernel Mamba)
class MambaInnerFnNoOutProj(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, xz, conv1ds, x_proj_weight, delta_proj_weight,
                A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
                C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1, num_iterations=4):
        assert checkpoint_lvl in [0, 1]
        if B is not None:
            B = B.detach().requires_grad_(True)
        if C is not None:
            C = C.detach().requires_grad_(True)
        if D is not None:
            D = D.detach().requires_grad_(True)

        num_iterations = min(num_iterations, len(conv1ds))  # num_iterations 값을 conv1ds 길이로 제한
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_gpu_dtype()
            x_proj_weight = x_proj_weight.to(dtype=dtype)
            delta_proj_weight = delta_proj_weight.to(dtype=dtype)
            if B_proj_bias is not None:
                B_proj_bias = B_proj_bias.to(dtype=dtype)
            if C_proj_bias is not None:
                C_proj_bias = C_proj_bias.to(dtype=dtype)
        if xz.stride(-1) != 1:
            xz = xz.contiguous()
        x, z = xz.chunk(2, dim=1)

        ctx.is_variable_B = B is None
        ctx.is_variable_C = C is None
        ctx.B_proj_bias_is_None = B_proj_bias is None
        ctx.C_proj_bias_is_None = C_proj_bias is None

        # 병렬 처리된 결과를 저장할 리스트
        out_z_list = []
        conv1d_out_list = []
        delta_list = []

        for i in range(num_iterations):
            conv1d = conv1ds[i]
            conv1d_weight = conv1d.weight
            conv1d_bias = conv1d.bias

            # conv1d_weight의 폭이 2에서 4 사이인지 확인
            if (conv1d_weight.shape[2] < 2) or (conv1d_weight.shape[2] > 4):
                raise ValueError(f"conv1d_weight width should be between 2 and 4, but got {conv1d_weight.shape[2]}")

            conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
            conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
            conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
            conv1d_out_list.append(conv1d_out)

            # 병렬 처리 가능한 연산들
            x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)
            delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
            delta_list.append(delta)

            if B is None:
                B = x_dbl[:, delta_rank:delta_rank + d_state]
                if B_proj_bias is not None:
                    B = B + B_proj_bias
                if not A.is_complex():
                    B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
                else:
                    B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
            else:
                if B.stride(-1) != 1:
                    B = B.contiguous()
            if C is None:
                C = x_dbl[:, -d_state:]
                if C_proj_bias is not None:
                    C = C + C_proj_bias
                if not A.is_complex():
                    C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
                else:
                    C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
            else:
                if C.stride(-1) != 1:
                    C = C.contiguous()
            if D is not None:
                D = D.contiguous()

            out, scan_intermediates, out_z = selective_scan_cuda.fwd(
                conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
            )

            out_z_list.append(out_z)

        final_out_z = torch.cat(out_z_list, dim=1)  # concat 연산으로 변경

        ctx.delta_softplus = delta_softplus
        ctx.checkpoint_lvl = checkpoint_lvl
        ctx.out_z_list = out_z_list  # out_z_list를 ctx에 저장
        ctx.conv1ds = conv1ds
        ctx.num_iterations = num_iterations
        ctx.delta_list = delta_list

        # conv1d_out과 delta를 저장하거나, 체크포인트 레벨에 따라 재계산할 수 있도록 설정
        ctx.save_for_backward(xz, x_proj_weight, delta_proj_weight,
                            A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, scan_intermediates, out, final_out_z, *conv1d_out_list)

        if checkpoint_lvl >= 1:
            ctx.conv1d_out_list = None
        else:
            ctx.conv1d_out_list = conv1d_out_list

        return final_out_z

    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        saved_tensors = ctx.saved_tensors
        (xz, x_proj_weight, delta_proj_weight,
        A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, scan_intermediates, out, final_out_z, *conv1d_out_list) = saved_tensors
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        x, z = xz.chunk(2, dim=1)
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if ctx.checkpoint_lvl == 1:
            conv1d_out_list = []
            delta_list = []
            for i in range(ctx.num_iterations):
                conv1d = ctx.conv1ds[i]
                conv1d_weight = conv1d.weight
                conv1d_bias = conv1d.bias
                conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
                conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
                conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
                conv1d_out_list.append(conv1d_out)

                x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)
                delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
                delta_list.append(delta)
        else:
            delta_list = ctx.delta_list

        dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
        dx, dz = dxz.chunk(2, dim=1)

        dconv1d_out_list = []
        ddelta_list = []
        dA_list = []
        dB_list = []
        dC_list = []
        dD_list = []
        ddelta_bias_list = []
        dz_list = []

        # out_z_list는 concat되어 있으므로 분리
        out_z_splits = torch.split(dout, dout.size(1) // len(ctx.out_z_list), dim=1)

        for i, out_z in enumerate(ctx.out_z_list):
            dout_part = out_z_splits[i]
            dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, _ = selective_scan_cuda.bwd(
                conv1d_out_list[i], delta_list[i], A, B, C, D, z, delta_bias, dout_part,
                scan_intermediates, out, dz, ctx.delta_softplus, True  # recompute out_z
            )
            dconv1d_out_list.append(dconv1d_out)
            ddelta_list.append(ddelta)
            dA_list.append(dA)
            dB_list.append(dB)
            dC_list.append(dC)
            dD_list.append(dD)
            ddelta_bias_list.append(ddelta_bias)
            dz_list.append(dz)

        dconv1d_out = sum(dconv1d_out_list)
        ddelta = sum(ddelta_list)
        dA = sum(dA_list)
        dB = sum(dB_list) if ctx.is_variable_B else None
        dC = sum(dC_list) if ctx.is_variable_C else None
        dD = sum(dD_list) if D is not None else None
        ddelta_bias = sum(ddelta_bias_list) if delta_bias is not None else None

        x_dbl = F.linear(rearrange(dconv1d_out, 'b d l -> (b l) d'), x_proj_weight)
        dx_dbl = torch.empty_like(x_dbl)
        dB_proj_bias = None
        if ctx.is_variable_B:
            if not A.is_complex():
                dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
            dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
            dB = None
        dC_proj_bias = None
        if ctx.is_variable_C:
            if not A.is_complex():
                dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
            dx_dbl[:, -d_state:] = dC  # (bl d)
            dC = None
        ddelta = rearrange(ddelta, "b d l -> d (b l)")
        ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
        dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
        dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
        dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out_list[0], "b d l -> (b l) d"))  # conv1d_out_list[0]으로 수정
        dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
        dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        dconv1d_weight, dconv1d_bias = [], []

        for i in range(ctx.num_iterations):
            weight = rearrange(ctx.conv1ds[i].weight, "d 1 w -> d w")
            bias = ctx.conv1ds[i].bias
            dconv1d_out_i = dconv1d_out_list[i]
            dx, dweight, dbias = causal_conv1d_cuda.causal_conv1d_bwd(
                x, weight, bias, dconv1d_out_i, dx, True
            )
            dconv1d_weight.append(rearrange(dweight, "d w -> d 1 w"))
            dconv1d_bias.append(dbias if bias is not None else None)

        # 올바른 반환 형식으로 변경
        return (dxz, None, dx_proj_weight, ddelta_proj_weight,
            dA, 
            dB if ctx.is_variable_B else None, 
            dC if ctx.is_variable_C else None, 
            dD if D is not None else None,
            ddelta_bias if delta_bias is not None else None, 
            dB_proj_bias if not ctx.B_proj_bias_is_None else None, 
            dC_proj_bias if not ctx.C_proj_bias_is_None else None, 
            None, None, None)


# class MambaInnerFnNoOutProj(torch.autograd.Function):

#     @staticmethod
#     @custom_fwd
#     def forward(ctx, xz, conv1ds, x_proj_weight, delta_proj_weight,
#                 A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#                 C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1, num_iterations=4):
#         assert checkpoint_lvl in [0, 1]
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         if torch.is_autocast_enabled():
#             dtype = torch.get_autocast_gpu_dtype()
#             x_proj_weight = x_proj_weight.to(dtype=dtype)
#             delta_proj_weight = delta_proj_weight.to(dtype=dtype)
#             if B_proj_bias is not None:
#                 B_proj_bias = B_proj_bias.to(dtype=dtype)
#             if C_proj_bias is not None:
#                 C_proj_bias = C_proj_bias.to(dtype=dtype)
#         if xz.stride(-1) != 1:
#             xz = xz.contiguous()
        
#         conv1d_out_list = []
#         for conv1d in conv1ds:
#             conv1d_weight = rearrange(conv1d.weight, "d 1 w -> d w")
#             conv1d_bias = conv1d.bias.contiguous() if conv1d.bias is not None else None
#             conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(xz, conv1d_weight, conv1d_bias, True)
#             conv1d_out_list.append(conv1d_out)

#         conv1d_out = sum(conv1d_out_list)

#         ctx.is_variable_B = B is None
#         ctx.is_variable_C = C is None
#         ctx.B_proj_bias_is_None = B_proj_bias is None
#         ctx.C_proj_bias_is_None = C_proj_bias is None
        
#         # 병렬 처리된 결과를 저장할 리스트
#         out_z_list = []
        
#         for _ in range(num_iterations):
#             # 병렬 처리 가능한 연산들
#             x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)
#             delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
            
#             if B is None:
#                 B = x_dbl[:, delta_rank:delta_rank + d_state]
#                 if B_proj_bias is not None:
#                     B = B + B_proj_bias
#                 if not A.is_complex():
#                     B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#                 else:
#                     B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#             else:
#                 if B.stride(-1) != 1:
#                     B = B.contiguous()
#             if C is None:
#                 C = x_dbl[:, -d_state:]
#                 if C_proj_bias is not None:
#                     C = C + C_proj_bias
#                 if not A.is_complex():
#                     C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#                 else:
#                     C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#             else:
#                 if C.stride(-1) != 1:
#                     C = C.contiguous()
#             if D is not None:
#                 D = D.contiguous()

#             out, scan_intermediates, out_z = selective_scan_cuda.fwd(
#                 conv1d_out, delta, A, B, C, D, xz, delta_bias, delta_softplus
#             )

#             # 병렬 처리된 결과를 리스트에 추가
#             out_z_list.append(out_z)

#         # 병렬 처리된 결과들을 합침
#         final_out_z = sum(out_z_list)
        
#         ctx.delta_softplus = delta_softplus
#         ctx.checkpoint_lvl = checkpoint_lvl
#         ctx.out_z_list = out_z_list  # out_z_list를 ctx에 저장

#         # conv1d_out과 delta를 저장하거나, 체크포인트 레벨에 따라 재계산할 수 있도록 설정
#         ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
#                             A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, scan_intermediates, out, final_out_z)

#         if checkpoint_lvl >= 1:
#             ctx.conv1d_out = None
#             ctx.delta = None
#         else:
#             ctx.conv1d_out = conv1d_out
#             ctx.delta = delta

#         return final_out_z


#     @staticmethod
#     @custom_bwd
#     def backward(ctx, dout):
#         (xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
#         A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, scan_intermediates, out, final_out_z) = ctx.saved_tensors
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         x, z = xz.chunk(2, dim=1)
#         if dout.stride(-1) != 1:
#             dout = dout.contiguous()
#         if ctx.checkpoint_lvl == 1:
#             conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
#             x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)
#             delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
#                             "d (b l) -> b d l", l=L)
            
#         # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
#         # backward of selective_scan_cuda with the backward of chunk).
#         dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
#         dx, dz = dxz.chunk(2, dim=1)
#         # dout_y = rearrange(dout, "b l d -> b d l") # because no arrange at end of forward, so dout shape is b d l
        
#         # 병렬 backward 처리된 결과를 저장할 리스트
#         dconv1d_out_list = []
#         ddelta_list = []
#         dA_list = []
#         dB_list = []
#         dC_list = []
#         dD_list = []
#         ddelta_bias_list = []
#         dz_list = []
        
#         for out_z in ctx.out_z_list:
#             dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, _ = selective_scan_cuda.bwd(
#                 conv1d_out, delta, A, B, C, D, xz, delta_bias, dout, scan_intermediates, out_z, dz,
#                 ctx.delta_softplus,
#                 True  # option to recompute out_z
#             )
#             dconv1d_out_list.append(dconv1d_out)
#             ddelta_list.append(ddelta)
#             dA_list.append(dA)
#             dB_list.append(dB)
#             dC_list.append(dC)
#             dD_list.append(dD)
#             ddelta_bias_list.append(ddelta_bias)
#             dz_list.append(dz)
        
#         # 병렬 처리된 결과들을 합침
#         dconv1d_out = sum(dconv1d_out_list)
#         ddelta = sum(ddelta_list)
#         dA = sum(dA_list)
#         dB = sum(dB_list)
#         dC = sum(dC_list)
#         dD = sum(dD_list)
#         ddelta_bias = sum(ddelta_bias_list) if delta_bias is not None else None
        
#         dx_dbl = torch.empty_like(x_dbl)
#         dB_proj_bias = None
#         if ctx.is_variable_B:
#             if not A.is_complex():
#                 dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
#             dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
#             dB = None
#         dC_proj_bias = None
#         if ctx.is_variable_C:
#             if not A.is_complex():
#                 dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
#             dx_dbl[:, -d_state:] = dC  # (bl d)
#             dC = None
#         ddelta = rearrange(ddelta, "b d l -> d (b l)")
#         ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
#         dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
#         dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
#         dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
#         dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
#         dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
#         # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
#         # backward of conv1d with the backward of chunk).
#         dx, dconv1d_weight, dconv1d_bias = causal_conv1d_cuda.causal_conv1d_bwd(
#             x, conv1d_weight, conv1d_bias, dconv1d_out, dx, True
#         )
#         dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
#         dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
#         return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
#                 dA, dB, dC, dD,
#                 ddelta_bias if delta_bias is not None else None,
#                 dB_proj_bias, dC_proj_bias, None, None, None)







# # Parallel 3 (must use multi kernel Mamba)

# class MambaInnerFnNoOutProj(torch.autograd.Function):

#     @staticmethod
#     @custom_fwd
#     def forward(ctx, xz, conv1d_weights, conv1d_biases, x_proj_weight, delta_proj_weight,
#                 A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus, checkpoint_lvl, num_iterations):
#         assert checkpoint_lvl in [0, 1]
#         ctx.delta_softplus = delta_softplus
#         ctx.checkpoint_lvl = checkpoint_lvl
#         ctx.num_iterations = num_iterations

#         ctx.conv1d_biases_requires_grad = [bias.requires_grad if isinstance(bias, torch.Tensor) else False for bias in conv1d_biases]
        
#         # conv1d_biases를 detach하고 원래의 requires_grad 상태를 유지합니다
#         conv1d_biases = [bias.detach().requires_grad_(bias.requires_grad) if isinstance(bias, torch.Tensor) else torch.zeros_like(weight[0]).requires_grad_(False) for bias, weight in zip(conv1d_biases, conv1d_weights)]

#         needs_input_grad = [
#             xz.requires_grad,
#             all(w.requires_grad for w in conv1d_weights),
#             any(bias.requires_grad for bias in conv1d_biases),
#             x_proj_weight.requires_grad,
#             delta_proj_weight.requires_grad,
#             A.requires_grad,
#             B.requires_grad if B is not None else False,
#             C.requires_grad if C is not None else False,
#             D.requires_grad if D is not None else False,
#             delta_bias.requires_grad if delta_bias is not None else False,
#             B_proj_bias.requires_grad if B_proj_bias is not None else False,
#             C_proj_bias.requires_grad if C_proj_bias is not None else False,
#             False,  # delta_softplus
#             False,  # checkpoint_lvl
#             False   # num_iterations
#         ]

#         ctx.saved_needs_input_grad = needs_input_grad

#         ctx.is_variable_B = B is None
#         ctx.is_variable_C = C is None
#         ctx.B_proj_bias_is_None = B_proj_bias is None
#         ctx.C_proj_bias_is_None = C_proj_bias is None

#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)

#         if xz.stride(-1) != 1:
#             xz = xz.contiguous()
#         x, z = xz.chunk(2, dim=1)

#         out_z_list = []
#         conv1d_out_list = []
#         delta_list = []

#         for i in range(num_iterations):
#             conv1d_weight = conv1d_weights[i]
#             conv1d_bias = conv1d_biases[i]

#             if (conv1d_weight.shape[1] < 2) or (conv1d_weight.shape[1] > 4):
#                 raise ValueError(f"conv1d_weight width should be between 2 and 4, but got {conv1d_weight.shape[1]}")

#             conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
#             conv1d_out_list.append(conv1d_out)

#             x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)
#             delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
#             delta_list.append(delta)

#             if B is None:
#                 B = x_dbl[:, delta_rank:delta_rank + d_state]
#                 if B_proj_bias is not None:
#                     B = B + B_proj_bias
#                 if not A.is_complex():
#                     B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#                 else:
#                     B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#             else:
#                 if B.stride(-1) != 1:
#                     B = B.contiguous()
#             if C is None:
#                 C = x_dbl[:, -d_state:]
#                 if C_proj_bias is not None:
#                     C = C + C_proj_bias
#                 if not A.is_complex():
#                     C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#                 else:
#                     C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#             else:
#                 if C.stride(-1) != 1:
#                     C = C.contiguous()
#             if D is not None:
#                 D = D.contiguous()

#             out, scan_intermediates, out_z = selective_scan_cuda.fwd(
#                 conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
#             )

#             out_z_list.append(out_z)

#         final_out_z = torch.cat(out_z_list, dim=1)

#         ctx.save_for_backward(xz, x_proj_weight, delta_proj_weight,
#                               A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, scan_intermediates, out, final_out_z, *conv1d_out_list)

#         ctx.conv1d_weights = conv1d_weights
#         ctx.conv1d_biases = conv1d_biases

#         if checkpoint_lvl >= 1:
#             ctx.conv1d_out_list = None
#             ctx.delta_list = None
#         else:
#             ctx.conv1d_out_list = conv1d_out_list
#             ctx.delta_list = delta_list

#         print("Forward pass:")
#         for i, bias in enumerate(conv1d_biases):
#             print(f"conv1d_bias {i} requires_grad: {bias.requires_grad}")

#         return final_out_z

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, dout):
#         saved_tensors = ctx.saved_tensors
#         (xz, x_proj_weight, delta_proj_weight,
#          A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, scan_intermediates, out, final_out_z, *conv1d_out_list) = saved_tensors

#         needs_input_grad = ctx.saved_needs_input_grad

#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         x, z = xz.chunk(2, dim=1)
#         if dout.stride(-1) != 1:
#             dout = dout.contiguous()

#         if ctx.checkpoint_lvl == 1:
#             conv1d_out_list = []
#             delta_list = []
#             for i in range(ctx.num_iterations):
#                 conv1d_weight = ctx.conv1d_weights[i]
#                 conv1d_bias = ctx.conv1d_biases[i]
#                 conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
#                 conv1d_out_list.append(conv1d_out)

#                 x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)
#                 delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
#                 delta_list.append(delta)
#         else:
#             delta_list = ctx.delta_list

#         dxz = torch.empty_like(xz) if needs_input_grad[0] else None
#         dx, dz = dxz.chunk(2, dim=1) if dxz is not None else (None, None)

#         dconv1d_out_list = []
#         ddelta_list = []
#         dA_list = []
#         dB_list = []
#         dC_list = []
#         dD_list = []
#         ddelta_bias_list = []
#         dz_list = []

#         out_z_splits = torch.split(dout, dout.size(1) // len(conv1d_out_list), dim=1)

#         for i, out_z in enumerate(out_z_splits):
#             dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, _ = selective_scan_cuda.bwd(
#                 conv1d_out_list[i], delta_list[i], A, B, C, D, z, delta_bias, out_z,
#                 scan_intermediates, out, dz, ctx.delta_softplus, True
#             )
#             dconv1d_out_list.append(dconv1d_out)
#             ddelta_list.append(ddelta)
#             dA_list.append(dA)
#             dB_list.append(dB)
#             dC_list.append(dC)
#             dD_list.append(dD)
#             ddelta_bias_list.append(ddelta_bias)
#             dz_list.append(dz)

#         dconv1d_out = sum(dconv1d_out_list)
#         ddelta = sum(ddelta_list)
#         dA = sum(dA_list) if needs_input_grad[5] else None
#         dB = sum(dB_list) if ctx.is_variable_B and needs_input_grad[6] else None
#         dC = sum(dC_list) if ctx.is_variable_C and needs_input_grad[7] else None
#         dD = sum(dD_list) if D is not None and needs_input_grad[8] else None
#         ddelta_bias = sum(ddelta_bias_list) if delta_bias is not None and needs_input_grad[9] else None

#         x_dbl = F.linear(rearrange(dconv1d_out, 'b d l -> (b l) d'), x_proj_weight)
#         dx_dbl = torch.empty_like(x_dbl) if needs_input_grad[3] or needs_input_grad[4] else None
#         dB_proj_bias = None
#         if ctx.is_variable_B and needs_input_grad[10]:
#             if not A.is_complex():
#                 dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
#             if dx_dbl is not None:
#                 dx_dbl[:, delta_rank:delta_rank + d_state] = dB
#             dB = None
#         dC_proj_bias = None
#         if ctx.is_variable_C and needs_input_grad[11]:
#             if not A.is_complex():
#                 dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
#             if dx_dbl is not None:
#                 dx_dbl[:, -d_state:] = dC
#             dC = None
#         ddelta = rearrange(ddelta, "b d l -> d (b l)")
#         ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank]) if needs_input_grad[4] else None
#         if dx_dbl is not None:
#             dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
#         dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
#         dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out_list[0], "b d l -> (b l) d")) if needs_input_grad[3] else None
#         dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
#         dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])

#         dconv1d_weights, dconv1d_biases = [], []

#         for i in range(ctx.num_iterations):
#             weight = ctx.conv1d_weights[i]
#             bias = ctx.conv1d_biases[i]
#             dconv1d_out_i = dconv1d_out_list[i]
#             dx_i, dweight, dbias = causal_conv1d_cuda.causal_conv1d_bwd(
#                 x, weight, bias, dconv1d_out_i, dx, True
#             )
#             dconv1d_weights.append(dweight if needs_input_grad[1] else None)
#             dconv1d_biases.append(dbias if ctx.conv1d_biases_requires_grad[i] else None)

#         print("Backward pass:")
#         for i, dbias in enumerate(dconv1d_biases):
#             print(f"dconv1d_bias {i} is None: {dbias is None}")

#         grads = [
#             dxz if needs_input_grad[0] else None,
#             dconv1d_weights if needs_input_grad[1] else None,
#             dconv1d_biases,  # 항상 리스트를 반환합니다
#             dx_proj_weight if needs_input_grad[3] else None,
#             ddelta_proj_weight if needs_input_grad[4] else None,
#             dA if needs_input_grad[5] else None, 
#             dB if ctx.is_variable_B and needs_input_grad[6] else None, 
#             dC if ctx.is_variable_C and needs_input_grad[7] else None, 
#             dD if D is not None and needs_input_grad[8] else None,
#             ddelta_bias if delta_bias is not None and needs_input_grad[9] else None, 
#             dB_proj_bias if not ctx.B_proj_bias_is_None and needs_input_grad[10] else None, 
#             dC_proj_bias if not ctx.C_proj_bias_is_None and needs_input_grad[11] else None,
#             None,  # delta_softplus
#             None,  # checkpoint_lvl
#         ]




# # Parallel 2
# class MambaInnerFnNoOutProj(torch.autograd.Function):

#     @staticmethod
#     @custom_fwd
#     def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
#                 A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#                 C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1, num_iterations=4):
#         assert checkpoint_lvl in [0, 1]
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         if torch.is_autocast_enabled():
#             dtype = torch.get_autocast_gpu_dtype()
#             x_proj_weight = x_proj_weight.to(dtype=dtype)
#             delta_proj_weight = delta_proj_weight.to(dtype=dtype)
#             if B_proj_bias is not None:
#                 B_proj_bias = B_proj_bias.to(dtype=dtype)
#             if C_proj_bias is not None:
#                 C_proj_bias = C_proj_bias.to(dtype=dtype)
#         if xz.stride(-1) != 1:
#             xz = xz.contiguous()
#         conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
#         x, z = xz.chunk(2, dim=1)
#         conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
#         conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
        
#         ctx.is_variable_B = B is None
#         ctx.is_variable_C = C is None
#         ctx.B_proj_bias_is_None = B_proj_bias is None
#         ctx.C_proj_bias_is_None = C_proj_bias is None
        
#         # 병렬 처리된 결과를 저장할 리스트
#         out_z_list = []
        
#         for _ in range(num_iterations):
#             # 병렬 처리 가능한 연산들
#             x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)
#             delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
            
#             if B is None:
#                 B = x_dbl[:, delta_rank:delta_rank + d_state]
#                 if B_proj_bias is not None:
#                     B = B + B_proj_bias
#                 if not A.is_complex():
#                     B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#                 else:
#                     B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#             else:
#                 if B.stride(-1) != 1:
#                     B = B.contiguous()
#             if C is None:
#                 C = x_dbl[:, -d_state:]
#                 if C_proj_bias is not None:
#                     C = C + C_proj_bias
#                 if not A.is_complex():
#                     C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#                 else:
#                     C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#             else:
#                 if C.stride(-1) != 1:
#                     C = C.contiguous()
#             if D is not None:
#                 D = D.contiguous()

#             out, scan_intermediates, out_z = selective_scan_cuda.fwd(
#                 conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
#             )

#             out_z_list.append(out_z)

#         final_out_z = torch.cat(out_z_list, dim=1)  # concat 연산으로 변경
        
#         ctx.delta_softplus = delta_softplus
#         ctx.checkpoint_lvl = checkpoint_lvl
#         ctx.out_z_list = out_z_list  # out_z_list를 ctx에 저장

#         # conv1d_out과 delta를 저장하거나, 체크포인트 레벨에 따라 재계산할 수 있도록 설정
#         ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
#                             A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, scan_intermediates, out, final_out_z)

#         if checkpoint_lvl >= 1:
#             ctx.conv1d_out = None
#             ctx.delta = None
#         else:
#             ctx.conv1d_out = conv1d_out
#             ctx.delta = delta

#         return final_out_z


#     @staticmethod
#     @custom_bwd
#     def backward(ctx, dout):
#         (xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
#         A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, scan_intermediates, out, out_z) = ctx.saved_tensors
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         x, z = xz.chunk(2, dim=1)
#         if dout.stride(-1) != 1:
#             dout = dout.contiguous()
#         if ctx.checkpoint_lvl == 1:
#             conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
#             x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)
#             delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
#                             "d (b l) -> b d l", l=L)
            
#         # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
#         # backward of selective_scan_cuda with the backward of chunk).
#         dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
#         dx, dz = dxz.chunk(2, dim=1)
#         # dout_y = rearrange(dout, "b l d -> b d l") # because no arrange at end of forward, so dout shape is b d l
        
#         dconv1d_out_list = []
#         ddelta_list = []
#         dA_list = []
#         dB_list = []
#         dC_list = []
#         dD_list = []
#         ddelta_bias_list = []
#         dz_list = []
        
#         # out_z_list는 concat되어 있으므로 분리
#         out_z_splits = torch.split(dout, dout.size(1) // len(ctx.out_z_list), dim=1)
        
#         for i, out_z in enumerate(ctx.out_z_list):
#             dout_part = out_z_splits[i]
#             dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, _ = selective_scan_cuda.bwd(
#                 conv1d_out, delta, A, B, C, D, z, delta_bias, dout_part, scan_intermediates, out, dz,
#                 ctx.delta_softplus,
#                 True  # option to recompute out_z
#             )
#             dconv1d_out_list.append(dconv1d_out)
#             ddelta_list.append(ddelta)
#             dA_list.append(dA)
#             dB_list.append(dB)
#             dC_list.append(dC)
#             dD_list.append(dD)
#             ddelta_bias_list.append(ddelta_bias)
#             dz_list.append(dz)
        
#         dconv1d_out = sum(dconv1d_out_list)
#         ddelta = sum(ddelta_list)
#         dA = sum(dA_list)
#         dB = sum(dB_list)
#         dC = sum(dC_list)
#         dD = sum(dD_list)
#         ddelta_bias = sum(ddelta_bias_list) if delta_bias is not None else None
        
#         dx_dbl = torch.empty_like(x_dbl)
#         dB_proj_bias = None
#         if ctx.is_variable_B:
#             if not A.is_complex():
#                 dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
#             dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
#             dB = None
#         dC_proj_bias = None
#         if ctx.is_variable_C:
#             if not A.is_complex():
#                 dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
#             dx_dbl[:, -d_state:] = dC  # (bl d)
#             dC = None
#         ddelta = rearrange(ddelta, "b d l -> d (b l)")
#         ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
#         dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
#         dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
#         dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
#         dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
#         dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
#         # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
#         # backward of conv1d with the backward of chunk).
#         dx, dconv1d_weight, dconv1d_bias = causal_conv1d_cuda.causal_conv1d_bwd(
#             x, conv1d_weight, conv1d_bias, dconv1d_out, dx, True
#         )
#         dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
#         dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
#         return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
#                 dA, dB, dC, dD,
#                 ddelta_bias if delta_bias is not None else None,
#                 dB_proj_bias, dC_proj_bias, None, None, None)














# # Parallel 1
# class MambaInnerFnNoOutProj(torch.autograd.Function):

#     @staticmethod
#     @custom_fwd
#     def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
#                 A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#                 C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1, num_iterations=4):
#         assert checkpoint_lvl in [0, 1]
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         if torch.is_autocast_enabled():
#             dtype = torch.get_autocast_gpu_dtype()
#             x_proj_weight = x_proj_weight.to(dtype=dtype)
#             delta_proj_weight = delta_proj_weight.to(dtype=dtype)
#             if B_proj_bias is not None:
#                 B_proj_bias = B_proj_bias.to(dtype=dtype)
#             if C_proj_bias is not None:
#                 C_proj_bias = C_proj_bias.to(dtype=dtype)
#         if xz.stride(-1) != 1:
#             xz = xz.contiguous()
#         conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
#         x, z = xz.chunk(2, dim=1)
#         conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
#         conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
        
#         ctx.is_variable_B = B is None
#         ctx.is_variable_C = C is None
#         ctx.B_proj_bias_is_None = B_proj_bias is None
#         ctx.C_proj_bias_is_None = C_proj_bias is None
        
#         # 병렬 처리된 결과를 저장할 리스트
#         out_z_list = []
        
#         for _ in range(num_iterations):
#             # 병렬 처리 가능한 연산들
#             x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)
#             delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
            
#             if B is None:
#                 B = x_dbl[:, delta_rank:delta_rank + d_state]
#                 if B_proj_bias is not None:
#                     B = B + B_proj_bias
#                 if not A.is_complex():
#                     B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#                 else:
#                     B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#             else:
#                 if B.stride(-1) != 1:
#                     B = B.contiguous()
#             if C is None:
#                 C = x_dbl[:, -d_state:]
#                 if C_proj_bias is not None:
#                     C = C + C_proj_bias
#                 if not A.is_complex():
#                     C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#                 else:
#                     C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#             else:
#                 if C.stride(-1) != 1:
#                     C = C.contiguous()
#             if D is not None:
#                 D = D.contiguous()

#             out, scan_intermediates, out_z = selective_scan_cuda.fwd(
#                 conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
#             )

#             # 병렬 처리된 결과를 리스트에 추가
#             out_z_list.append(out_z)

#         # 병렬 처리된 결과들을 합침
#         final_out_z = sum(out_z_list)
        
#         ctx.delta_softplus = delta_softplus
#         ctx.checkpoint_lvl = checkpoint_lvl
#         ctx.out_z_list = out_z_list  # out_z_list를 ctx에 저장

#         # conv1d_out과 delta를 저장하거나, 체크포인트 레벨에 따라 재계산할 수 있도록 설정
#         ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
#                             A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, scan_intermediates, out, final_out_z)

#         if checkpoint_lvl >= 1:
#             ctx.conv1d_out = None
#             ctx.delta = None
#         else:
#             ctx.conv1d_out = conv1d_out
#             ctx.delta = delta

#         return final_out_z


#     @staticmethod
#     @custom_bwd
#     def backward(ctx, dout):
#         (xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
#         A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, scan_intermediates, out, out_z) = ctx.saved_tensors
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         x, z = xz.chunk(2, dim=1)
#         if dout.stride(-1) != 1:
#             dout = dout.contiguous()
#         if ctx.checkpoint_lvl == 1:
#             conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
#             x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)
#             delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
#                             "d (b l) -> b d l", l=L)
            
#         # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
#         # backward of selective_scan_cuda with the backward of chunk).
#         dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
#         dx, dz = dxz.chunk(2, dim=1)
#         # dout_y = rearrange(dout, "b l d -> b d l") # because no arrange at end of forward, so dout shape is b d l
        
#         # 병렬 backward 처리된 결과를 저장할 리스트
#         dconv1d_out_list = []
#         ddelta_list = []
#         dA_list = []
#         dB_list = []
#         dC_list = []
#         dD_list = []
#         ddelta_bias_list = []
#         dz_list = []
        
#         for out_z in ctx.out_z_list:
#             dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, _ = selective_scan_cuda.bwd(
#                 conv1d_out, delta, A, B, C, D, z, delta_bias, dout, scan_intermediates, out, dz,
#                 ctx.delta_softplus,
#                 True  # option to recompute out_z
#             )
#             dconv1d_out_list.append(dconv1d_out)
#             ddelta_list.append(ddelta)
#             dA_list.append(dA)
#             dB_list.append(dB)
#             dC_list.append(dC)
#             dD_list.append(dD)
#             ddelta_bias_list.append(ddelta_bias)
#             dz_list.append(dz)
        
#         # 병렬 처리된 결과들을 합침
#         dconv1d_out = sum(dconv1d_out_list)
#         ddelta = sum(ddelta_list)
#         dA = sum(dA_list)
#         dB = sum(dB_list)
#         dC = sum(dC_list)
#         dD = sum(dD_list)
#         ddelta_bias = sum(ddelta_bias_list) if delta_bias is not None else None
        
#         dx_dbl = torch.empty_like(x_dbl)
#         dB_proj_bias = None
#         if ctx.is_variable_B:
#             if not A.is_complex():
#                 dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
#             dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
#             dB = None
#         dC_proj_bias = None
#         if ctx.is_variable_C:
#             if not A.is_complex():
#                 dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
#             dx_dbl[:, -d_state:] = dC  # (bl d)
#             dC = None
#         ddelta = rearrange(ddelta, "b d l -> d (b l)")
#         ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
#         dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
#         dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
#         dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
#         dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
#         dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
#         # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
#         # backward of conv1d with the backward of chunk).
#         dx, dconv1d_weight, dconv1d_bias = causal_conv1d_cuda.causal_conv1d_bwd(
#             x, conv1d_weight, conv1d_bias, dconv1d_out, dx, True
#         )
#         dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
#         dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
#         return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
#                 dA, dB, dC, dD,
#                 ddelta_bias if delta_bias is not None else None,
#                 dB_proj_bias, dC_proj_bias, None, None, None)





# # Add skip connection & Recurrent mechanism 4
# class MambaInnerFnNoOutProj(torch.autograd.Function):

#     @staticmethod
#     @custom_fwd
#     def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
#                 A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#                 C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1, num_iterations=4):
#         assert checkpoint_lvl in [0, 1]
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         if torch.is_autocast_enabled():
#             dtype = torch.get_autocast_gpu_dtype()
#             x_proj_weight = x_proj_weight.to(dtype=dtype)
#             delta_proj_weight = delta_proj_weight.to(dtype=dtype)
#             if B_proj_bias is not None:
#                 B_proj_bias = B_proj_bias.to(dtype=dtype)
#             if C_proj_bias is not None:
#                 C_proj_bias = C_proj_bias.to(dtype=dtype)
#         if xz.stride(-1) != 1:
#             xz = xz.contiguous()
#         conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
#         x, z = xz.chunk(2, dim=1)
#         conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
#         conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
        
#         ctx.is_variable_B = B is None
#         ctx.is_variable_C = C is None
#         ctx.B_proj_bias_is_None = B_proj_bias is None
#         ctx.C_proj_bias_is_None = C_proj_bias is None
        
#         for _ in range(num_iterations):
#             x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)
#             delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
            
            
#             if B is None:
#                 B = x_dbl[:, delta_rank:delta_rank + d_state]
#                 if B_proj_bias is not None:
#                     B = B + B_proj_bias
#                 if not A.is_complex():
#                     B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#                 else:
#                     B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#             else:
#                 if B.stride(-1) != 1:
#                     B = B.contiguous()
#             if C is None:
#                 C = x_dbl[:, -d_state:]
#                 if C_proj_bias is not None:
#                     C = C + C_proj_bias
#                 if not A.is_complex():
#                     C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#                 else:
#                     C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#             else:
#                 if C.stride(-1) != 1:
#                     C = C.contiguous()
#             if D is not None:
#                 D = D.contiguous()

#             out, scan_intermediates, out_z = selective_scan_cuda.fwd(
#                 conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
#             )

#             # Skip connection
#             out_z = out_z + conv1d_out
#             conv1d_out = out_z.detach()  # Update conv1d_out for the next iteration

#         ctx.delta_softplus = delta_softplus
#         ctx.checkpoint_lvl = checkpoint_lvl
#         if checkpoint_lvl >= 1:
#             conv1d_out, delta = None, None
#         ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
#                               delta_proj_weight, conv1d_out, delta,
#                               A, B, C, D, delta_bias, scan_intermediates, out, out_z)
#         return out_z

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, dout):
#         (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, 
#         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, out, out_z) = ctx.saved_tensors
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         x, z = xz.chunk(2, dim=1)
        
#         if dout.stride(-1) != 1:
#             dout = dout.contiguous()
        
#         if ctx.checkpoint_lvl == 1:
#             conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
#             delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
#                             "d (b l) -> b d l", l=L)
        
#         # Include skip connection in gradient
#         dconv1d_out = dout
        
#         dxz = torch.empty_like(xz)
#         dx, dz = dxz.chunk(2, dim=1)
#         dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, _ = selective_scan_cuda.bwd(
#             conv1d_out, delta, A, B, C, D, z, delta_bias, dconv1d_out, scan_intermediates, out, dz,
#             ctx.delta_softplus,
#             True
#         )
        
#         dconv1d_out += dout  # Add gradient from skip connection
        
#         dD = dD if D is not None else None
#         dx_dbl = torch.empty_like(x_dbl)
#         dB_proj_bias = None
        
#         if ctx.is_variable_B:
#             if not A.is_complex():
#                 dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
#             dx_dbl[:, delta_rank:delta_rank + d_state] = dB
#             dB = None
        
#         dC_proj_bias = None
#         if ctx.is_variable_C:
#             if not A.is_complex():
#                 dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
#             dx_dbl[:, -d_state:] = dC
#             dC = None
        
#         ddelta = rearrange(ddelta, "b d l -> d (b l)")
#         ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
#         dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
#         dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
#         dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
#         dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
#         dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        
#         dx, dconv1d_weight, dconv1d_bias = causal_conv1d_cuda.causal_conv1d_bwd(
#             x, conv1d_weight, conv1d_bias, dconv1d_out, dx, True
#         )
        
#         dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
#         dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        
#         return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
#                 dA, dB if B.requires_grad else None, dC if C.requires_grad else None, dD,
#                 ddelta_bias if delta_bias is not None else None,
#                 dB_proj_bias, dC_proj_bias, None, None, None)




# # Add skip connection & Recurrent mechanism 3
# class MambaInnerFnNoOutProj(torch.autograd.Function):

#     @staticmethod
#     @custom_fwd
#     def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
#                 A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#                 C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1, num_iterations=4):
#         assert checkpoint_lvl in [0, 1]
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         if torch.is_autocast_enabled():
#             dtype = torch.get_autocast_gpu_dtype()
#             x_proj_weight = x_proj_weight.to(dtype=dtype)
#             delta_proj_weight = delta_proj_weight.to(dtype=dtype)
#             if B_proj_bias is not None:
#                 B_proj_bias = B_proj_bias.to(dtype=dtype)
#             if C_proj_bias is not None:
#                 C_proj_bias = C_proj_bias.to(dtype=dtype)
#         if xz.stride(-1) != 1:
#             xz = xz.contiguous()
#         conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
#         x, z = xz.chunk(2, dim=1)
#         conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
#         conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
        
#         ctx.is_variable_B = B is None
#         ctx.is_variable_C = C is None
#         ctx.B_proj_bias_is_None = B_proj_bias is None
#         ctx.C_proj_bias_is_None = C_proj_bias is None
        
#         for _ in range(num_iterations):
#             x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)
#             delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
            
            
#             if B is None:
#                 B = x_dbl[:, delta_rank:delta_rank + d_state]
#                 if B_proj_bias is not None:
#                     B = B + B_proj_bias
#                 if not A.is_complex():
#                     B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#                 else:
#                     B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#             else:
#                 if B.stride(-1) != 1:
#                     B = B.contiguous()
#             if C is None:
#                 C = x_dbl[:, -d_state:]
#                 if C_proj_bias is not None:
#                     C = C + C_proj_bias
#                 if not A.is_complex():
#                     C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#                 else:
#                     C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#             else:
#                 if C.stride(-1) != 1:
#                     C = C.contiguous()
#             if D is not None:
#                 D = D.contiguous()

#             out, scan_intermediates, out_z = selective_scan_cuda.fwd(
#                 conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
#             )

#             # Skip connection
#             out_z = out_z + conv1d_out
#             conv1d_out = out_z  # Update conv1d_out for the next iteration

#         ctx.delta_softplus = delta_softplus
#         ctx.checkpoint_lvl = checkpoint_lvl
#         if checkpoint_lvl >= 1:
#             conv1d_out, delta = None, None
#         ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
#                               delta_proj_weight, conv1d_out, delta,
#                               A, B, C, D, delta_bias, scan_intermediates, out, out_z)
#         return out_z

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, dout):
#         (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, 
#         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, out, out_z) = ctx.saved_tensors
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         x, z = xz.chunk(2, dim=1)
        
#         if dout.stride(-1) != 1:
#             dout = dout.contiguous()
        
#         if ctx.checkpoint_lvl == 1:
#             conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
#             delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
#                             "d (b l) -> b d l", l=L)
        
#         # Include skip connection in gradient
#         dconv1d_out = dout
        
#         dxz = torch.empty_like(xz)
#         dx, dz = dxz.chunk(2, dim=1)
#         dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, _ = selective_scan_cuda.bwd(
#             conv1d_out, delta, A, B, C, D, z, delta_bias, dconv1d_out, scan_intermediates, out, dz,
#             ctx.delta_softplus,
#             True
#         )
        
#         dconv1d_out += dout  # Add gradient from skip connection
        
#         dD = dD if D is not None else None
#         dx_dbl = torch.empty_like(x_dbl)
#         dB_proj_bias = None
        
#         if ctx.is_variable_B:
#             if not A.is_complex():
#                 dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
#             dx_dbl[:, delta_rank:delta_rank + d_state] = dB
#             dB = None
        
#         dC_proj_bias = None
#         if ctx.is_variable_C:
#             if not A.is_complex():
#                 dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
#             dx_dbl[:, -d_state:] = dC
#             dC = None
        
#         ddelta = rearrange(ddelta, "b d l -> d (b l)")
#         ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
#         dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
#         dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
#         dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
#         dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
#         dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        
#         dx, dconv1d_weight, dconv1d_bias = causal_conv1d_cuda.causal_conv1d_bwd(
#             x, conv1d_weight, conv1d_bias, dconv1d_out, dx, True
#         )
        
#         dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
#         dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        
#         return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
#                 dA, dB if B.requires_grad else None, dC if C.requires_grad else None, dD,
#                 ddelta_bias if delta_bias is not None else None,
#                 dB_proj_bias, dC_proj_bias, None, None, None)




# # Add skip connection & Recurrent mechanism2
# class MambaInnerFnNoOutProj(torch.autograd.Function):

#     @staticmethod
#     @custom_fwd
#     def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
#                 A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#                 C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1, num_iterations=4):
#         assert checkpoint_lvl in [0, 1]
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)

#         if torch.is_autocast_enabled():
#             dtype = torch.get_autocast_gpu_dtype()
#             x_proj_weight = x_proj_weight.to(dtype=dtype)
#             delta_proj_weight = delta_proj_weight.to(dtype=dtype)
#             if B_proj_bias is not None:
#                 B_proj_bias = B_proj_bias.to(dtype=dtype)
#             if C_proj_bias is not None:
#                 C_proj_bias = C_proj_bias.to(dtype=dtype)
        
#         batch_norm = torch.nn.BatchNorm1d(xz.size(1)//2).to(xz.device)  # BatchNorm layer
#         dropout = torch.nn.Dropout(0.05).to(xz.device)  # Dropout layer with 10% drop rate

        
#         if xz.stride(-1) != 1:
#             xz = xz.contiguous()
#         conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
#         x, z = xz.chunk(2, dim=1)
#         conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
#         conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)

#         ctx.is_variable_B = B is None
#         ctx.is_variable_C = C is None
#         ctx.B_proj_bias_is_None = B_proj_bias is None
#         ctx.C_proj_bias_is_None = C_proj_bias is None

#         B_init = B is None
#         C_init = C is None

#         for _ in range(num_iterations):
#             # x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)
#             # delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
#             conv1d_out = batch_norm(conv1d_out)  # Apply batch normalization
#             x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)
#             x_dbl = dropout(x_dbl)  # Apply dropout
#             delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)

            
#             if B_init:
#                 B = x_dbl[:, delta_rank:delta_rank + d_state]
#                 if B_proj_bias is not None:
#                     B = B + B_proj_bias
#                 if not A.is_complex():
#                     B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#                 else:
#                     B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#                 B_init = False
#             else:
#                 if B.stride(-1) != 1:
#                     B = B.contiguous()

#             if C_init:
#                 C = x_dbl[:, -d_state:]
#                 if C_proj_bias is not None:
#                     C = C + C_proj_bias
#                 if not A.is_complex():
#                     C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#                 else:
#                     C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#                 C_init = False
#             else:
#                 if C.stride(-1) != 1:
#                     C = C.contiguous()

#             if D is not None:
#                 D = D.contiguous()


#             out, scan_intermediates, out_z = selective_scan_cuda.fwd(
#                 conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
#             )

#             # Skip connection
#             out_z = out_z + conv1d_out
#             out_z = F.silu(out_z)

#             # Check for NaN values and apply gradient clipping
#             if torch.isnan(out_z).any():
#                 raise ValueError("NaN detected in out_z during iteration")

#             # # Gradient Clipping
#             # out_z = torch.clamp(out_z, min=-1e6, max=1e6)

#             conv1d_out = out_z  # Update conv1d_out for the next iteration

#         ctx.delta_softplus = delta_softplus
#         ctx.checkpoint_lvl = checkpoint_lvl
#         if checkpoint_lvl >= 1:
#             conv1d_out, delta = None, None
#         ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
#                               delta_proj_weight, conv1d_out, delta,
#                               A, B, C, D, delta_bias, scan_intermediates, out, out_z)
#         return out_z
    
#     @staticmethod
#     @custom_bwd
#     def backward(ctx, dout):
#         (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, 
#         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, out, out_z) = ctx.saved_tensors
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         x, z = xz.chunk(2, dim=1)
        
#         if dout.stride(-1) != 1:
#             dout = dout.contiguous()
        
#         if ctx.checkpoint_lvl == 1:
#             conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
#             delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
#                             "d (b l) -> b d l", l=L)
        
#         # Backpropagate through SiLU activation
#         dout = dout * torch.sigmoid(out_z) * (1 + out_z * (1 - torch.sigmoid(out_z)))
        
#         # Include skip connection in gradient
#         dconv1d_out = dout
        
#         dxz = torch.empty_like(xz)
#         dx, dz = dxz.chunk(2, dim=1)
#         dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, _ = selective_scan_cuda.bwd(
#             conv1d_out, delta, A, B, C, D, z, delta_bias, dconv1d_out, scan_intermediates, out, dz,
#             ctx.delta_softplus,
#             True
#         )
        
#         dconv1d_out += dout  # Add gradient from skip connection
        
#         dD = dD if D is not None else None
#         dx_dbl = torch.empty_like(x_dbl)
#         dB_proj_bias = None
        
#         if ctx.is_variable_B:
#             if not A.is_complex():
#                 dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
#             dx_dbl[:, delta_rank:delta_rank + d_state] = dB
#             dB = None
        
#         dC_proj_bias = None
#         if ctx.is_variable_C:
#             if not A.is_complex():
#                 dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
#             dx_dbl[:, -d_state:] = dC
#             dC = None
        
#         ddelta = rearrange(ddelta, "b d l -> d (b l)")
#         ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
#         dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
#         dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
#         dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
#         dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
#         dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        
#         dx, dconv1d_weight, dconv1d_bias = causal_conv1d_cuda.causal_conv1d_bwd(
#             x, conv1d_weight, conv1d_bias, dconv1d_out, dx, True
#         )
        
#         dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
#         dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        
#         # Return the correct number of gradients
#         return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
#                 dA, dB if B.requires_grad else None, dC if C.requires_grad else None, dD,
#                 ddelta_bias if delta_bias is not None else None,
#                 dB_proj_bias, dC_proj_bias, None, None, None)  # Add extra Nones to match the number of inputs




# # Add skip connection & Recurrent mechanism
# class MambaInnerFnNoOutProj(torch.autograd.Function):

#     @staticmethod
#     @custom_fwd
#     def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
#                 A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#                 C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1, num_iterations=4):
#         assert checkpoint_lvl in [0, 1]
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)

#         if torch.is_autocast_enabled():
#             dtype = torch.get_autocast_gpu_dtype()
#             x_proj_weight = x_proj_weight.to(dtype=dtype)
#             delta_proj_weight = delta_proj_weight.to(dtype=dtype)
#             if B_proj_bias is not None:
#                 B_proj_bias = B_proj_bias.to(dtype=dtype)
#             if C_proj_bias is not None:
#                 C_proj_bias = C_proj_bias.to(dtype=dtype)
        
#         batch_norm = torch.nn.BatchNorm1d(xz.size(1)//2).to(xz.device)  # BatchNorm layer
#         dropout = torch.nn.Dropout(0.05).to(xz.device)  # Dropout layer with 10% drop rate

        
#         if xz.stride(-1) != 1:
#             xz = xz.contiguous()
#         conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
#         x, z = xz.chunk(2, dim=1)
#         conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
#         conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)

#         ctx.is_variable_B = B is None
#         ctx.is_variable_C = C is None
#         ctx.B_proj_bias_is_None = B_proj_bias is None
#         ctx.C_proj_bias_is_None = C_proj_bias is None

#         B_init = B is None
#         C_init = C is None

#         for _ in range(num_iterations):
#             # x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)
#             # delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
#             conv1d_out = batch_norm(conv1d_out)  # Apply batch normalization
#             x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)
#             x_dbl = dropout(x_dbl)  # Apply dropout
#             delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)

            
#             if B_init:
#                 B = x_dbl[:, delta_rank:delta_rank + d_state]
#                 if B_proj_bias is not None:
#                     B = B + B_proj_bias
#                 if not A.is_complex():
#                     B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#                 else:
#                     B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#                 B_init = False
#             else:
#                 if B.stride(-1) != 1:
#                     B = B.contiguous()

#             if C_init:
#                 C = x_dbl[:, -d_state:]
#                 if C_proj_bias is not None:
#                     C = C + C_proj_bias
#                 if not A.is_complex():
#                     C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#                 else:
#                     C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#                 C_init = False
#             else:
#                 if C.stride(-1) != 1:
#                     C = C.contiguous()

#             if D is not None:
#                 D = D.contiguous()


#             out, scan_intermediates, out_z = selective_scan_cuda.fwd(
#                 conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
#             )

#             # Skip connection
#             out_z = out_z + conv1d_out

#             # Check for NaN values and apply gradient clipping
#             if torch.isnan(out_z).any():
#                 raise ValueError("NaN detected in out_z during iteration")

#             # # Gradient Clipping
#             # out_z = torch.clamp(out_z, min=-1e6, max=1e6)

#             conv1d_out = out_z  # Update conv1d_out for the next iteration

#         ctx.delta_softplus = delta_softplus
#         ctx.checkpoint_lvl = checkpoint_lvl
#         if checkpoint_lvl >= 1:
#             conv1d_out, delta = None, None
#         ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
#                               delta_proj_weight, conv1d_out, delta,
#                               A, B, C, D, delta_bias, scan_intermediates, out, out_z)
#         return out_z
    
#     @staticmethod
#     @custom_bwd
#     def backward(ctx, dout):
#         (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, 
#         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, out, out_z) = ctx.saved_tensors
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         x, z = xz.chunk(2, dim=1)
        
#         if dout.stride(-1) != 1:
#             dout = dout.contiguous()
        
#         if ctx.checkpoint_lvl == 1:
#             conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
#             delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
#                             "d (b l) -> b d l", l=L)
        
#         # Include skip connection in gradient
#         dconv1d_out = dout
        
#         dxz = torch.empty_like(xz)
#         dx, dz = dxz.chunk(2, dim=1)
#         dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, _ = selective_scan_cuda.bwd(
#             conv1d_out, delta, A, B, C, D, z, delta_bias, dconv1d_out, scan_intermediates, out, dz,
#             ctx.delta_softplus,
#             True
#         )
        
#         dconv1d_out += dout  # Add gradient from skip connection
        
#         dD = dD if D is not None else None
#         dx_dbl = torch.empty_like(x_dbl)
#         dB_proj_bias = None
        
#         if ctx.is_variable_B:
#             if not A.is_complex():
#                 dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
#             dx_dbl[:, delta_rank:delta_rank + d_state] = dB
#             dB = None
        
#         dC_proj_bias = None
#         if ctx.is_variable_C:
#             if not A.is_complex():
#                 dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
#             dx_dbl[:, -d_state:] = dC
#             dC = None
        
#         ddelta = rearrange(ddelta, "b d l -> d (b l)")
#         ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
#         dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
#         dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
#         dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
#         dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
#         dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        
#         dx, dconv1d_weight, dconv1d_bias = causal_conv1d_cuda.causal_conv1d_bwd(
#             x, conv1d_weight, conv1d_bias, dconv1d_out, dx, True
#         )
        
#         dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
#         dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        
#         # Return the correct number of gradients
#         return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
#                 dA, dB if B.requires_grad else None, dC if C.requires_grad else None, dD,
#                 ddelta_bias if delta_bias is not None else None,
#                 dB_proj_bias, dC_proj_bias, None, None, None)  # Add extra Nones to match the number of inputs




# # Add skip connection2
# class MambaInnerFnNoOutProj(torch.autograd.Function):

#     @staticmethod
#     @custom_fwd
#     def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
#                 A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#                 C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1):
#         assert checkpoint_lvl in [0, 1]
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         if torch.is_autocast_enabled():
#             dtype = torch.get_autocast_gpu_dtype()
#             x_proj_weight = x_proj_weight.to(dtype=dtype)
#             delta_proj_weight = delta_proj_weight.to(dtype=dtype)
#             if B_proj_bias is not None:
#                 B_proj_bias = B_proj_bias.to(dtype=dtype)
#             if C_proj_bias is not None:
#                 C_proj_bias = C_proj_bias.to(dtype=dtype)
#         if xz.stride(-1) != 1:
#             xz = xz.contiguous()
#         conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
#         x, z = xz.chunk(2, dim=1)
#         conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
#         conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)

#         x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)
#         delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
        
#         ctx.is_variable_B = B is None
#         ctx.is_variable_C = C is None
#         ctx.B_proj_bias_is_None = B_proj_bias is None
#         ctx.C_proj_bias_is_None = C_proj_bias is None
#         if B is None:
#             B = x_dbl[:, delta_rank:delta_rank + d_state]
#             if B_proj_bias is not None:
#                 B = B + B_proj_bias
#             if not A.is_complex():
#                 B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#             else:
#                 B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#         else:
#             if B.stride(-1) != 1:
#                 B = B.contiguous()
#         if C is None:
#             C = x_dbl[:, -d_state:]
#             if C_proj_bias is not None:
#                 C = C + C_proj_bias
#             if not A.is_complex():
#                 C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#             else:
#                 C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#         else:
#             if C.stride(-1) != 1:
#                 C = C.contiguous()
#         if D is not None:
#             D = D.contiguous()
        
#         out, scan_intermediates, out_z = selective_scan_cuda.fwd(
#             conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
#         )
        
#         # Skip connection
#         out_z = out_z + conv1d_out
#         out_z = F.silu(out_z)
        
#         ctx.delta_softplus = delta_softplus
#         ctx.checkpoint_lvl = checkpoint_lvl
#         if checkpoint_lvl >= 1:
#             conv1d_out, delta = None, None
#         ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
#                               delta_proj_weight, conv1d_out, delta,
#                               A, B, C, D, delta_bias, scan_intermediates, out, out_z)
#         return out_z

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, dout):
#         (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, 
#         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, out, out_z) = ctx.saved_tensors
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         x, z = xz.chunk(2, dim=1)
        
#         if dout.stride(-1) != 1:
#             dout = dout.contiguous()
        
#         if ctx.checkpoint_lvl == 1:
#             conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
#             delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
#                             "d (b l) -> b d l", l=L)
        
#         # Backpropagate through SiLU activation
#         dout = dout * torch.sigmoid(out_z) * (1 + out_z * (1 - torch.sigmoid(out_z)))
        
#         # Include skip connection in gradient
#         dconv1d_out = dout
        
#         dxz = torch.empty_like(xz)
#         dx, dz = dxz.chunk(2, dim=1)
#         dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, _ = selective_scan_cuda.bwd(
#             conv1d_out, delta, A, B, C, D, z, delta_bias, dconv1d_out, scan_intermediates, out, dz,
#             ctx.delta_softplus,
#             True
#         )
        
#         dconv1d_out += dout  # Add gradient from skip connection
        
#         dD = dD if D is not None else None
#         dx_dbl = torch.empty_like(x_dbl)
#         dB_proj_bias = None
        
#         if ctx.is_variable_B:
#             if not A.is_complex():
#                 dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
#             dx_dbl[:, delta_rank:delta_rank + d_state] = dB
#             dB = None
        
#         dC_proj_bias = None
#         if ctx.is_variable_C:
#             if not A.is_complex():
#                 dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
#             dx_dbl[:, -d_state:] = dC
#             dC = None
        
#         ddelta = rearrange(ddelta, "b d l -> d (b l)")
#         ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
#         dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
#         dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
#         dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
#         dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
#         dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        
#         dx, dconv1d_weight, dconv1d_bias = causal_conv1d_cuda.causal_conv1d_bwd(
#             x, conv1d_weight, conv1d_bias, dconv1d_out, dx, True
#         )
        
#         dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
#         dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        
#         return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
#                 dA, dB, dC, dD,
#                 ddelta_bias if delta_bias is not None else None,
#                 dB_proj_bias, dC_proj_bias, None)



# # Add skip connection
# class MambaInnerFnNoOutProj(torch.autograd.Function):

#     @staticmethod
#     @custom_fwd
#     def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
#                 A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#                 C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1):
#         assert checkpoint_lvl in [0, 1]
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         if torch.is_autocast_enabled():
#             dtype = torch.get_autocast_gpu_dtype()
#             x_proj_weight = x_proj_weight.to(dtype=dtype)
#             delta_proj_weight = delta_proj_weight.to(dtype=dtype)
#             if B_proj_bias is not None:
#                 B_proj_bias = B_proj_bias.to(dtype=dtype)
#             if C_proj_bias is not None:
#                 C_proj_bias = C_proj_bias.to(dtype=dtype)
#         if xz.stride(-1) != 1:
#             xz = xz.contiguous()
#         conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
#         x, z = xz.chunk(2, dim=1)
#         conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
#         conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)

#         x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)
#         delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
        
#         ctx.is_variable_B = B is None
#         ctx.is_variable_C = C is None
#         ctx.B_proj_bias_is_None = B_proj_bias is None
#         ctx.C_proj_bias_is_None = C_proj_bias is None
#         if B is None:
#             B = x_dbl[:, delta_rank:delta_rank + d_state]
#             if B_proj_bias is not None:
#                 B = B + B_proj_bias
#             if not A.is_complex():
#                 B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#             else:
#                 B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#         else:
#             if B.stride(-1) != 1:
#                 B = B.contiguous()
#         if C is None:
#             C = x_dbl[:, -d_state:]
#             if C_proj_bias is not None:
#                 C = C + C_proj_bias
#             if not A.is_complex():
#                 C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#             else:
#                 C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#         else:
#             if C.stride(-1) != 1:
#                 C = C.contiguous()
#         if D is not None:
#             D = D.contiguous()
        
#         out, scan_intermediates, out_z = selective_scan_cuda.fwd(
#             conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
#         )
        
#         # Skip connection
#         out_z = out_z + conv1d_out
        
#         ctx.delta_softplus = delta_softplus
#         ctx.checkpoint_lvl = checkpoint_lvl
#         if checkpoint_lvl >= 1:
#             conv1d_out, delta = None, None
#         ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
#                               delta_proj_weight, conv1d_out, delta,
#                               A, B, C, D, delta_bias, scan_intermediates, out, out_z)
#         return out_z

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, dout):
#         (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, 
#         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, out, out_z) = ctx.saved_tensors
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         x, z = xz.chunk(2, dim=1)
        
#         if dout.stride(-1) != 1:
#             dout = dout.contiguous()
        
#         if ctx.checkpoint_lvl == 1:
#             conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
#             delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
#                             "d (b l) -> b d l", l=L)
        
#         # Include skip connection in gradient
#         dconv1d_out = dout
        
#         dxz = torch.empty_like(xz)
#         dx, dz = dxz.chunk(2, dim=1)
#         dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, _ = selective_scan_cuda.bwd(
#             conv1d_out, delta, A, B, C, D, z, delta_bias, dconv1d_out, scan_intermediates, out, dz,
#             ctx.delta_softplus,
#             True
#         )
        
#         dconv1d_out += dout  # Add gradient from skip connection
        
#         dD = dD if D is not None else None
#         dx_dbl = torch.empty_like(x_dbl)
#         dB_proj_bias = None
        
#         if ctx.is_variable_B:
#             if not A.is_complex():
#                 dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
#             dx_dbl[:, delta_rank:delta_rank + d_state] = dB
#             dB = None
        
#         dC_proj_bias = None
#         if ctx.is_variable_C:
#             if not A.is_complex():
#                 dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
#             dx_dbl[:, -d_state:] = dC
#             dC = None
        
#         ddelta = rearrange(ddelta, "b d l -> d (b l)")
#         ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
#         dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
#         dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
#         dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
#         dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
#         dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        
#         dx, dconv1d_weight, dconv1d_bias = causal_conv1d_cuda.causal_conv1d_bwd(
#             x, conv1d_weight, conv1d_bias, dconv1d_out, dx, True
#         )
        
#         dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
#         dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        
#         return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
#                 dA, dB, dC, dD,
#                 ddelta_bias if delta_bias is not None else None,
#                 dB_proj_bias, dC_proj_bias, None)







# #  Upgrade version 11
# class MambaInnerFnNoOutProj(torch.autograd.Function):

#     @staticmethod
#     @custom_fwd
#     def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
#                 A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#                 C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1):
#         assert checkpoint_lvl in [0, 1]
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         if torch.is_autocast_enabled():
#             dtype = torch.get_autocast_gpu_dtype()
#             x_proj_weight = x_proj_weight.to(dtype=dtype)
#             delta_proj_weight = delta_proj_weight.to(dtype=dtype)
#             if B_proj_bias is not None:
#                 B_proj_bias = B_proj_bias.to(dtype=dtype)
#             if C_proj_bias is not None:
#                 C_proj_bias = C_proj_bias.to(dtype=dtype)
#         if xz.stride(-1) != 1:
#             xz = xz.contiguous()
#         conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
#         x, z = xz.chunk(2, dim=1)
#         conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
#         conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)

#         x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)
#         delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
        
#         ctx.is_variable_B = B is None
#         ctx.is_variable_C = C is None
#         ctx.B_proj_bias_is_None = B_proj_bias is None
#         ctx.C_proj_bias_is_None = C_proj_bias is None
#         if B is None:
#             B = x_dbl[:, delta_rank:delta_rank + d_state]
#             if B_proj_bias is not None:
#                 B = B + B_proj_bias
#             if not A.is_complex():
#                 B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#             else:
#                 B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#         else:
#             if B.stride(-1) != 1:
#                 B = B.contiguous()
#         if C is None:
#             C = x_dbl[:, -d_state:]
#             if C_proj_bias is not None:
#                 C = C + C_proj_bias
#             if not A.is_complex():
#                 C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#             else:
#                 C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#         else:
#             if C.stride(-1) != 1:
#                 C = C.contiguous()
#         if D is not None:
#             D = D.contiguous()
        
#         out, scan_intermediates, out_z = selective_scan_cuda.fwd(
#             conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
#         )
        
#         # Skip connection
#         out_z = out_z + conv1d_out
        
#         # Apply Layer Normalization
#         # out_z = F.layer_norm(out_z, out_z.shape[1:])
#         # out_z = F.silu(out_z)
        
        
#         ctx.delta_softplus = delta_softplus
#         ctx.checkpoint_lvl = checkpoint_lvl
#         if checkpoint_lvl >= 1:
#             conv1d_out, delta = None, None
#         ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
#                               delta_proj_weight, conv1d_out, delta,
#                               A, B, C, D, delta_bias, scan_intermediates, out, out_z)
#         return out_z

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, dout):
#         (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, 
#         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, out, out_z) = ctx.saved_tensors
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         x, z = xz.chunk(2, dim=1)
        
#         if dout.stride(-1) != 1:
#             dout = dout.contiguous()
        
#         if ctx.checkpoint_lvl == 1:
#             conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
#             delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
#                             "d (b l) -> b d l", l=L)
        
#         # # Backpropagate through SiLU activation
#         # dout = dout * torch.sigmoid(out_z) * (1 + out_z * (1 - torch.sigmoid(out_z)))
        
#         # # Backpropagate through layer normalization
#         # normalized_shape = out_z.shape[1:]
#         # grad_out_z = dout.contiguous()
#         # mean = out_z.mean(dim=-1, keepdim=True)
#         # variance = out_z.var(dim=-1, keepdim=True, unbiased=False)
#         # out_z_centered = out_z - mean
#         # std_inv = 1.0 / torch.sqrt(variance + 1e-5)
#         # grad_out_z_centered = grad_out_z * std_inv
#         # grad_var = (grad_out_z * out_z_centered).sum(dim=-1, keepdim=True) * -0.5 * std_inv.pow(3)
#         # grad_mean = grad_out_z.sum(dim=-1, keepdim=True) * -std_inv + grad_var * out_z_centered.mean(dim=-1, keepdim=True) * -2.0
#         # dout = grad_out_z_centered + grad_var * 2.0 * out_z_centered / normalized_shape[-1] + grad_mean / normalized_shape[-1]
        
#         dxz = torch.empty_like(xz)
#         dx, dz = dxz.chunk(2, dim=1)
#         dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
#             conv1d_out, delta, A, B, C, D, z, delta_bias, dout, scan_intermediates, out, dz,
#             ctx.delta_softplus,
#             True
#         )
        
#         dD = dD if D is not None else None
#         dx_dbl = torch.empty_like(x_dbl)
#         dB_proj_bias = None
        
#         if ctx.is_variable_B:
#             if not A.is_complex():
#                 dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
#             dx_dbl[:, delta_rank:delta_rank + d_state] = dB
#             dB = None
        
#         dC_proj_bias = None
#         if ctx.is_variable_C:
#             if not A.is_complex():
#                 dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
#             dx_dbl[:, -d_state:] = dC
#             dC = None
        
#         ddelta = rearrange(ddelta, "b d l -> d (b l)")
#         ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
#         dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
#         dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
#         dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
#         dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
#         dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        
#         dx, dconv1d_weight, dconv1d_bias = causal_conv1d_cuda.causal_conv1d_bwd(
#             x, conv1d_weight, conv1d_bias, dconv1d_out, dx, True
#         )
        
#         dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
#         dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        
#         return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
#                 dA, dB, dC, dD,
#                 ddelta_bias if delta_bias is not None else None,
#                 dB_proj_bias, dC_proj_bias, None)


# # Upgrade version 10
# class MambaInnerFnNoOutProj(torch.autograd.Function):

#     @staticmethod
#     @custom_fwd
#     def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
#                 A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#                 C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1):
#         assert checkpoint_lvl in [0, 1]
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         if torch.is_autocast_enabled():
#             dtype = torch.get_autocast_gpu_dtype()
#             x_proj_weight = x_proj_weight.to(dtype=dtype)
#             delta_proj_weight = delta_proj_weight.to(dtype=dtype)
#             if B_proj_bias is not None:
#                 B_proj_bias = B_proj_bias.to(dtype=dtype)
#             if C_proj_bias is not None:
#                 C_proj_bias = C_proj_bias.to(dtype=dtype)
#         if xz.stride(-1) != 1:
#             xz = xz.contiguous()
#         conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
#         x, z = xz.chunk(2, dim=1)
#         conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
#         conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)

#         x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)
#         delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
        
#         ctx.is_variable_B = B is None
#         ctx.is_variable_C = C is None
#         ctx.B_proj_bias_is_None = B_proj_bias is None
#         ctx.C_proj_bias_is_None = C_proj_bias is None
#         if B is None:
#             B = x_dbl[:, delta_rank:delta_rank + d_state]
#             if B_proj_bias is not None:
#                 B = B + B_proj_bias
#             if not A.is_complex():
#                 B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#             else:
#                 B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#         else:
#             if B.stride(-1) != 1:
#                 B = B.contiguous()
#         if C is None:
#             C = x_dbl[:, -d_state:]
#             if C_proj_bias is not None:
#                 C = C + C_proj_bias
#             if not A.is_complex():
#                 C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#             else:
#                 C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#         else:
#             if C.stride(-1) != 1:
#                 C = C.contiguous()
#         if D is not None:
#             D = D.contiguous()
        
#         out, scan_intermediates, out_z = selective_scan_cuda.fwd(
#             conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
#         )
        
#         # Skip connection
#         out_z = out_z + conv1d_out
        
#         # Apply Layer Normalization
#         # out_z = F.layer_norm(out_z, out_z.shape[1:])
#         out_z = F.silu(out_z)
        
        
#         ctx.delta_softplus = delta_softplus
#         ctx.checkpoint_lvl = checkpoint_lvl
#         if checkpoint_lvl >= 1:
#             conv1d_out, delta = None, None
#         ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
#                               delta_proj_weight, conv1d_out, delta,
#                               A, B, C, D, delta_bias, scan_intermediates, out, out_z)
#         return out_z

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, dout):
#         (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, 
#         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, out, out_z) = ctx.saved_tensors
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         x, z = xz.chunk(2, dim=1)
        
#         if dout.stride(-1) != 1:
#             dout = dout.contiguous()
        
#         if ctx.checkpoint_lvl == 1:
#             conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
#             delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
#                             "d (b l) -> b d l", l=L)
        
#         # Backpropagate through SiLU activation
#         dout = dout * torch.sigmoid(out_z) * (1 + out_z * (1 - torch.sigmoid(out_z)))
        
#         # # Backpropagate through layer normalization
#         # normalized_shape = out_z.shape[1:]
#         # grad_out_z = dout.contiguous()
#         # mean = out_z.mean(dim=-1, keepdim=True)
#         # variance = out_z.var(dim=-1, keepdim=True, unbiased=False)
#         # out_z_centered = out_z - mean
#         # std_inv = 1.0 / torch.sqrt(variance + 1e-5)
#         # grad_out_z_centered = grad_out_z * std_inv
#         # grad_var = (grad_out_z * out_z_centered).sum(dim=-1, keepdim=True) * -0.5 * std_inv.pow(3)
#         # grad_mean = grad_out_z.sum(dim=-1, keepdim=True) * -std_inv + grad_var * out_z_centered.mean(dim=-1, keepdim=True) * -2.0
#         # dout = grad_out_z_centered + grad_var * 2.0 * out_z_centered / normalized_shape[-1] + grad_mean / normalized_shape[-1]
        
#         dxz = torch.empty_like(xz)
#         dx, dz = dxz.chunk(2, dim=1)
#         dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
#             conv1d_out, delta, A, B, C, D, z, delta_bias, dout, scan_intermediates, out, dz,
#             ctx.delta_softplus,
#             True
#         )
        
#         dD = dD if D is not None else None
#         dx_dbl = torch.empty_like(x_dbl)
#         dB_proj_bias = None
        
#         if ctx.is_variable_B:
#             if not A.is_complex():
#                 dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
#             dx_dbl[:, delta_rank:delta_rank + d_state] = dB
#             dB = None
        
#         dC_proj_bias = None
#         if ctx.is_variable_C:
#             if not A.is_complex():
#                 dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
#             dx_dbl[:, -d_state:] = dC
#             dC = None
        
#         ddelta = rearrange(ddelta, "b d l -> d (b l)")
#         ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
#         dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
#         dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
#         dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
#         dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
#         dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        
#         dx, dconv1d_weight, dconv1d_bias = causal_conv1d_cuda.causal_conv1d_bwd(
#             x, conv1d_weight, conv1d_bias, dconv1d_out, dx, True
#         )
        
#         dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
#         dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        
#         return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
#                 dA, dB, dC, dD,
#                 ddelta_bias if delta_bias is not None else None,
#                 dB_proj_bias, dC_proj_bias, None)



# # Upgrade version 9
# class MambaInnerFnNoOutProj(torch.autograd.Function):

#     @staticmethod
#     @custom_fwd
#     def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
#                 A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#                 C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1):
#         assert checkpoint_lvl in [0, 1]
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         if torch.is_autocast_enabled():
#             dtype = torch.get_autocast_gpu_dtype()
#             x_proj_weight = x_proj_weight.to(dtype=dtype)
#             delta_proj_weight = delta_proj_weight.to(dtype=dtype)
#             if B_proj_bias is not None:
#                 B_proj_bias = B_proj_bias.to(dtype=dtype)
#             if C_proj_bias is not None:
#                 C_proj_bias = C_proj_bias.to(dtype=dtype)
#         if xz.stride(-1) != 1:
#             xz = xz.contiguous()
#         conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
#         x, z = xz.chunk(2, dim=1)
#         conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
#         conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)

#         x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)
#         delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
        
#         ctx.is_variable_B = B is None
#         ctx.is_variable_C = C is None
#         ctx.B_proj_bias_is_None = B_proj_bias is None
#         ctx.C_proj_bias_is_None = C_proj_bias is None
#         if B is None:
#             B = x_dbl[:, delta_rank:delta_rank + d_state]
#             if B_proj_bias is not None:
#                 B = B + B_proj_bias
#             if not A.is_complex():
#                 B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#             else:
#                 B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#         else:
#             if B.stride(-1) != 1:
#                 B = B.contiguous()
#         if C is None:
#             C = x_dbl[:, -d_state:]
#             if C_proj_bias is not None:
#                 C = C + C_proj_bias
#             if not A.is_complex():
#                 C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#             else:
#                 C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#         else:
#             if C.stride(-1) != 1:
#                 C = C.contiguous()
#         if D is not None:
#             D = D.contiguous()
        
#         out, scan_intermediates, out_z = selective_scan_cuda.fwd(
#             conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
#         )
        
#         # Skip connection
#         out_z = out_z + conv1d_out
        
#         # Apply Layer Normalization
#         out_z = F.layer_norm(out_z, out_z.shape[1:])
#         out_z = F.silu(out_z)
        
        
#         ctx.delta_softplus = delta_softplus
#         ctx.checkpoint_lvl = checkpoint_lvl
#         if checkpoint_lvl >= 1:
#             conv1d_out, delta = None, None
#         ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
#                               delta_proj_weight, conv1d_out, delta,
#                               A, B, C, D, delta_bias, scan_intermediates, out, out_z)
#         return out_z

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, dout):
#         (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, 
#         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, out, out_z) = ctx.saved_tensors
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         x, z = xz.chunk(2, dim=1)
        
#         if dout.stride(-1) != 1:
#             dout = dout.contiguous()
        
#         if ctx.checkpoint_lvl == 1:
#             conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
#             delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
#                             "d (b l) -> b d l", l=L)
        
#         # Backpropagate through SiLU activation
#         dout = dout * torch.sigmoid(out_z) * (1 + out_z * (1 - torch.sigmoid(out_z)))
        
#         # Backpropagate through layer normalization
#         normalized_shape = out_z.shape[1:]
#         grad_out_z = dout.contiguous()
#         mean = out_z.mean(dim=-1, keepdim=True)
#         variance = out_z.var(dim=-1, keepdim=True, unbiased=False)
#         out_z_centered = out_z - mean
#         std_inv = 1.0 / torch.sqrt(variance + 1e-5)
#         grad_out_z_centered = grad_out_z * std_inv
#         grad_var = (grad_out_z * out_z_centered).sum(dim=-1, keepdim=True) * -0.5 * std_inv.pow(3)
#         grad_mean = grad_out_z.sum(dim=-1, keepdim=True) * -std_inv + grad_var * out_z_centered.mean(dim=-1, keepdim=True) * -2.0
#         dout = grad_out_z_centered + grad_var * 2.0 * out_z_centered / normalized_shape[-1] + grad_mean / normalized_shape[-1]
        
#         dxz = torch.empty_like(xz)
#         dx, dz = dxz.chunk(2, dim=1)
#         dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
#             conv1d_out, delta, A, B, C, D, z, delta_bias, dout, scan_intermediates, out, dz,
#             ctx.delta_softplus,
#             True
#         )
        
#         dD = dD if D is not None else None
#         dx_dbl = torch.empty_like(x_dbl)
#         dB_proj_bias = None
        
#         if ctx.is_variable_B:
#             if not A.is_complex():
#                 dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
#             dx_dbl[:, delta_rank:delta_rank + d_state] = dB
#             dB = None
        
#         dC_proj_bias = None
#         if ctx.is_variable_C:
#             if not A.is_complex():
#                 dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
#             dx_dbl[:, -d_state:] = dC
#             dC = None
        
#         ddelta = rearrange(ddelta, "b d l -> d (b l)")
#         ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
#         dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
#         dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
#         dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
#         dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
#         dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        
#         dx, dconv1d_weight, dconv1d_bias = causal_conv1d_cuda.causal_conv1d_bwd(
#             x, conv1d_weight, conv1d_bias, dconv1d_out, dx, True
#         )
        
#         dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
#         dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        
#         return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
#                 dA, dB, dC, dD,
#                 ddelta_bias if delta_bias is not None else None,
#                 dB_proj_bias, dC_proj_bias, None)




# # Upgrade version 8
# class MambaInnerFnNoOutProj(torch.autograd.Function):

#     @staticmethod
#     @custom_fwd
#     def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
#                 A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#                 C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1):
#         assert checkpoint_lvl in [0, 1]
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         if torch.is_autocast_enabled():
#             dtype = torch.get_autocast_gpu_dtype()
#             x_proj_weight = x_proj_weight.to(dtype=dtype)
#             delta_proj_weight = delta_proj_weight.to(dtype=dtype)
#             if B_proj_bias is not None:
#                 B_proj_bias = B_proj_bias.to(dtype=dtype)
#             if C_proj_bias is not None:
#                 C_proj_bias = C_proj_bias.to(dtype=dtype)
#         if xz.stride(-1) != 1:
#             xz = xz.contiguous()
#         conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
#         x, z = xz.chunk(2, dim=1)
#         conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
#         conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)

#         x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)
#         delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
        
#         ctx.is_variable_B = B is None
#         ctx.is_variable_C = C is None
#         ctx.B_proj_bias_is_None = B_proj_bias is None
#         ctx.C_proj_bias_is_None = C_proj_bias is None
#         if B is None:
#             B = x_dbl[:, delta_rank:delta_rank + d_state]
#             if B_proj_bias is not None:
#                 B = B + B_proj_bias
#             if not A.is_complex():
#                 B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#             else:
#                 B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#         else:
#             if B.stride(-1) != 1:
#                 B = B.contiguous()
#         if C is None:
#             C = x_dbl[:, -d_state:]
#             if C_proj_bias is not None:
#                 C = C + C_proj_bias
#             if not A.is_complex():
#                 C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#             else:
#                 C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#         else:
#             if C.stride(-1) != 1:
#                 C = C.contiguous()
#         if D is not None:
#             D = D.contiguous()
        
#         out, scan_intermediates, out_z = selective_scan_cuda.fwd(
#             conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
#         )
        
#         # Skip connection
#         out_z = out_z + conv1d_out
        
#         # Apply Layer Normalization
#         # out_z = F.layer_norm(out_z, out_z.shape[1:])
#         # out_z = F.silu(out_z)
        
        
#         ctx.delta_softplus = delta_softplus
#         ctx.checkpoint_lvl = checkpoint_lvl
#         if checkpoint_lvl >= 1:
#             conv1d_out, delta = None, None
#         ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
#                               delta_proj_weight, conv1d_out, delta,
#                               A, B, C, D, delta_bias, scan_intermediates, out, out_z)
#         return out_z

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, dout):
#         (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, 
#         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, out, out_z) = ctx.saved_tensors
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         x, z = xz.chunk(2, dim=1)
        
#         if dout.stride(-1) != 1:
#             dout = dout.contiguous()
        
#         if ctx.checkpoint_lvl == 1:
#             conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
#             delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
#                             "d (b l) -> b d l", l=L)
        
#         # Backpropagate through SiLU activation
#         # dout = dout * (F.silu(out_z) + torch.sigmoid(out_z) * (1 - F.silu(out_z)))
        
#         # Backpropagate through layer normalization
#         # dout = F.layer_norm(out_z, out_z.shape[1:]).backward(dout)
        
#         # Include skip connection in gradient
#         dconv1d_out = dout
        
#         dxz = torch.empty_like(xz)
#         dx, dz = dxz.chunk(2, dim=1)
#         dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
#             conv1d_out, delta, A, B, C, D, z, delta_bias, dconv1d_out, scan_intermediates, out, dz,
#             ctx.delta_softplus,
#             True
#         )
        
#         dD = dD if D is not None else None
#         dx_dbl = torch.empty_like(x_dbl)
#         dB_proj_bias = None
        
#         if ctx.is_variable_B:
#             if not A.is_complex():
#                 dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
#             dx_dbl[:, delta_rank:delta_rank + d_state] = dB
#             dB = None
        
#         dC_proj_bias = None
#         if ctx.is_variable_C:
#             if not A.is_complex():
#                 dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
#             dx_dbl[:, -d_state:] = dC
#             dC = None
        
#         ddelta = rearrange(ddelta, "b d l -> d (b l)")
#         ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
#         dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
#         dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
#         dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
#         dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
#         dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        
#         dx, dconv1d_weight, dconv1d_bias = causal_conv1d_cuda.causal_conv1d_bwd(
#             x, conv1d_weight, conv1d_bias, dconv1d_out, dx, True
#         )
        
#         dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
#         dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        
#         return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
#                 dA, dB, dC, dD,
#                 ddelta_bias if delta_bias is not None else None,
#                 dB_proj_bias, dC_proj_bias, None)





# # Upgrade version 7
# class MambaInnerFnNoOutProj(torch.autograd.Function):

#     @staticmethod
#     @custom_fwd
#     def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
#                 A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#                 C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1):
#         assert checkpoint_lvl in [0, 1]
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         if torch.is_autocast_enabled():
#             dtype = torch.get_autocast_gpu_dtype()
#             x_proj_weight = x_proj_weight.to(dtype=dtype)
#             delta_proj_weight = delta_proj_weight.to(dtype=dtype)
#             if B_proj_bias is not None:
#                 B_proj_bias = B_proj_bias.to(dtype=dtype)
#             if C_proj_bias is not None:
#                 C_proj_bias = C_proj_bias.to(dtype=dtype)
#         if xz.stride(-1) != 1:
#             xz = xz.contiguous()
#         conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
#         x, z = xz.chunk(2, dim=1)
#         conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
#         conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)

#         x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)
#         delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
        
#         ctx.is_variable_B = B is None
#         ctx.is_variable_C = C is None
#         ctx.B_proj_bias_is_None = B_proj_bias is None
#         ctx.C_proj_bias_is_None = C_proj_bias is None
#         if B is None:
#             B = x_dbl[:, delta_rank:delta_rank + d_state]
#             if B_proj_bias is not None:
#                 B = B + B_proj_bias
#             if not A.is_complex():
#                 B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#             else:
#                 B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#         else:
#             if B.stride(-1) != 1:
#                 B = B.contiguous()
#         if C is None:
#             C = x_dbl[:, -d_state:]
#             if C_proj_bias is not None:
#                 C = C + C_proj_bias
#             if not A.is_complex():
#                 C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#             else:
#                 C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#         else:
#             if C.stride(-1) != 1:
#                 C = C.contiguous()
#         if D is not None:
#             D = D.contiguous()
        
#         out, scan_intermediates, out_z = selective_scan_cuda.fwd(
#             conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
#         )
        
#         # Skip connection
#         out_z = out_z + conv1d_out
        
#         # Apply Layer Normalization
#         # out_z = F.layer_norm(out_z, out_z.shape[1:])
#         out_z = F.silu(out_z)
        
        
#         ctx.delta_softplus = delta_softplus
#         ctx.checkpoint_lvl = checkpoint_lvl
#         if checkpoint_lvl >= 1:
#             conv1d_out, delta = None, None
#         ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
#                               delta_proj_weight, conv1d_out, delta,
#                               A, B, C, D, delta_bias, scan_intermediates, out, out_z)
#         return out_z

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, dout):
#         (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, 
#         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, out, out_z) = ctx.saved_tensors
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         x, z = xz.chunk(2, dim=1)
        
#         if dout.stride(-1) != 1:
#             dout = dout.contiguous()
        
#         if ctx.checkpoint_lvl == 1:
#             conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
#             delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
#                             "d (b l) -> b d l", l=L)
        
#         # Backpropagate through SiLU activation
#         dout = dout * (F.silu(out_z) + torch.sigmoid(out_z) * (1 - F.silu(out_z)))
        
#         # Backpropagate through layer normalization
#         # dout = F.layer_norm(out_z, out_z.shape[1:]).backward(dout)
        
#         # Include skip connection in gradient
#         dconv1d_out = dout
        
#         dxz = torch.empty_like(xz)
#         dx, dz = dxz.chunk(2, dim=1)
#         dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
#             conv1d_out, delta, A, B, C, D, z, delta_bias, dconv1d_out, scan_intermediates, out, dz,
#             ctx.delta_softplus,
#             True
#         )
        
#         dD = dD if D is not None else None
#         dx_dbl = torch.empty_like(x_dbl)
#         dB_proj_bias = None
        
#         if ctx.is_variable_B:
#             if not A.is_complex():
#                 dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
#             dx_dbl[:, delta_rank:delta_rank + d_state] = dB
#             dB = None
        
#         dC_proj_bias = None
#         if ctx.is_variable_C:
#             if not A.is_complex():
#                 dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
#             dx_dbl[:, -d_state:] = dC
#             dC = None
        
#         ddelta = rearrange(ddelta, "b d l -> d (b l)")
#         ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
#         dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
#         dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
#         dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
#         dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
#         dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        
#         dx, dconv1d_weight, dconv1d_bias = causal_conv1d_cuda.causal_conv1d_bwd(
#             x, conv1d_weight, conv1d_bias, dconv1d_out, dx, True
#         )
        
#         dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
#         dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        
#         return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
#                 dA, dB, dC, dD,
#                 ddelta_bias if delta_bias is not None else None,
#                 dB_proj_bias, dC_proj_bias, None)






# # Upgrade version 6
# class MambaInnerFnNoOutProj(torch.autograd.Function):

#     @staticmethod
#     @custom_fwd
#     def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
#                 A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#                 C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1):
#         """
#              xz: (batch, dim, seqlen)
#         """
#         assert checkpoint_lvl in [0, 1]
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         if torch.is_autocast_enabled():
#             x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
#             delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
#         if xz.stride(-1) != 1:
#             xz = xz.contiguous()
#         conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
#         x, z = xz.chunk(2, dim=1)
#         conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
#         conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)

#         x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
#         delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
        
#         ctx.is_variable_B = B is None
#         ctx.is_variable_C = C is None
#         ctx.B_proj_bias_is_None = B_proj_bias is None
#         ctx.C_proj_bias_is_None = C_proj_bias is None
#         if B is None:
#             B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
#             if B_proj_bias is not None:
#                 B = B + B_proj_bias.to(dtype=B.dtype)
#             if not A.is_complex():
#                 B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#             else:
#                 B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#         else:
#             if B.stride(-1) != 1:
#                 B = B.contiguous()
#         if C is None:
#             C = x_dbl[:, -d_state:]  # (bl dstate)
#             if C_proj_bias is not None:
#                 C = C + C_proj_bias.to(dtype=C.dtype)
#             if not A.is_complex():
#                 C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#             else:
#                 C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#         else:
#             if C.stride(-1) != 1:
#                 C = C.contiguous()
#         if D is not None:
#             D = D.contiguous()
        
#         out, scan_intermediates, out_z = selective_scan_cuda.fwd(
#             conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
#         )
        
#         # Skip connection
#         out_z = out_z + conv1d_out
        
#         ctx.delta_softplus = delta_softplus
#         ctx.checkpoint_lvl = checkpoint_lvl
#         if checkpoint_lvl >= 1:
#             conv1d_out, delta = None, None
#         ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
#                               delta_proj_weight, conv1d_out, delta,
#                               A, B, C, D, delta_bias, scan_intermediates, out, out_z)
#         return out_z

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, dout):
#         # Unpack saved tensors
#         (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, 
#          conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, out, out_z) = ctx.saved_tensors
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         x, z = xz.chunk(2, dim=1)
        
#         if dout.stride(-1) != 1:
#             dout = dout.contiguous()
        
#         if ctx.checkpoint_lvl == 1:
#             conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
#             delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
#                               "d (b l) -> b d l", l=L)
        
#         # Adjust dout for skip connection
#         dout = dout + conv1d_out
        
#         dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
#         dx, dz = dxz.chunk(2, dim=1)
#         dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
#             conv1d_out, delta, A, B, C, D, z, delta_bias, dout, scan_intermediates, out, dz,
#             ctx.delta_softplus,
#             True  # option to recompute out_z
#         )
        
#         dD = dD if D is not None else None
#         dx_dbl = torch.empty_like(x_dbl)
#         dB_proj_bias = None
        
#         if ctx.is_variable_B:
#             if not A.is_complex():
#                 dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
#             dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
#             dB = None
        
#         dC_proj_bias = None
#         if ctx.is_variable_C:
#             if not A.is_complex():
#                 dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
#             dx_dbl[:, -d_state:] = dC  # (bl d)
#             dC = None
        
#         ddelta = rearrange(ddelta, "b d l -> d (b l)")
#         ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
#         dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
#         dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
#         dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
#         dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
#         dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        
#         dx, dconv1d_weight, dconv1d_bias = causal_conv1d_cuda.causal_conv1d_bwd(
#             x, conv1d_weight, conv1d_bias, dconv1d_out, dx, True
#         )
        
#         dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
#         dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        
#         return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
#                 dA, dB, dC, dD,
#                 ddelta_bias if delta_bias is not None else None,
#                 dB_proj_bias, dC_proj_bias, None)



# # Upgrade version 5
# class MambaInnerFnNoOutProj(torch.autograd.Function):

#     @staticmethod
#     @custom_fwd
#     def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
#                 A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#                 C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1):
#         assert checkpoint_lvl in [0, 1]
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         if torch.is_autocast_enabled():
#             dtype = torch.get_autocast_gpu_dtype()
#             x_proj_weight = x_proj_weight.to(dtype=dtype)
#             delta_proj_weight = delta_proj_weight.to(dtype=dtype)
#             if B_proj_bias is not None:
#                 B_proj_bias = B_proj_bias.to(dtype=dtype)
#             if C_proj_bias is not None:
#                 C_proj_bias = C_proj_bias.to(dtype=dtype)
#         if xz.stride(-1) != 1:
#             xz = xz.contiguous()
#         conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
#         x, z = xz.chunk(2, dim=1)
#         conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
#         conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)

#         x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)
#         delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
        
#         ctx.is_variable_B = B is None
#         ctx.is_variable_C = C is None
#         ctx.B_proj_bias_is_None = B_proj_bias is None
#         ctx.C_proj_bias_is_None = C_proj_bias is None
#         if B is None:
#             B = x_dbl[:, delta_rank:delta_rank + d_state]
#             if B_proj_bias is not None:
#                 B = B + B_proj_bias
#             if not A.is_complex():
#                 B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#             else:
#                 B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#         else:
#             if B.stride(-1) != 1:
#                 B = B.contiguous()
#         if C is None:
#             C = x_dbl[:, -d_state:]
#             if C_proj_bias is not None:
#                 C = C + C_proj_bias
#             if not A.is_complex():
#                 C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#             else:
#                 C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#         else:
#             if C.stride(-1) != 1:
#                 C = C.contiguous()
#         if D is not None:
#             D = D.contiguous()
        
#         out, scan_intermediates, out_z = selective_scan_cuda.fwd(
#             conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
#         )
        
#         # Skip connection
#         out_z = out_z + conv1d_out
        
#         ctx.delta_softplus = delta_softplus
#         ctx.checkpoint_lvl = checkpoint_lvl
#         if checkpoint_lvl >= 1:
#             conv1d_out, delta = None, None
#         ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
#                               delta_proj_weight, conv1d_out, delta,
#                               A, B, C, D, delta_bias, scan_intermediates, out, out_z)
#         return out_z

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, dout):
#         (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, 
#         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, out, out_z) = ctx.saved_tensors
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         x, z = xz.chunk(2, dim=1)
        
#         if dout.stride(-1) != 1:
#             dout = dout.contiguous()
        
#         if ctx.checkpoint_lvl == 1:
#             conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
#             delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
#                             "d (b l) -> b d l", l=L)
        
#         # Include skip connection in gradient
#         dconv1d_out = dout
        
        
#         dxz = torch.empty_like(xz)
#         dx, dz = dxz.chunk(2, dim=1)
#         dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
#             conv1d_out, delta, A, B, C, D, z, delta_bias, dconv1d_out, scan_intermediates, out, dz,
#             ctx.delta_softplus,
#             True
#         )
        
#         # dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
#         #     conv1d_out, delta, A, B, C, D, z, delta_bias, dout, scan_intermediates, out, dz,
#         #     ctx.delta_softplus,
#         #     True  # option to recompute out_z
#         # )
        
#         dD = dD if D is not None else None
#         dx_dbl = torch.empty_like(x_dbl)
#         dB_proj_bias = None
        
#         if ctx.is_variable_B:
#             if not A.is_complex():
#                 dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
#             dx_dbl[:, delta_rank:delta_rank + d_state] = dB
#             dB = None
        
#         dC_proj_bias = None
#         if ctx.is_variable_C:
#             if not A.is_complex():
#                 dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
#             dx_dbl[:, -d_state:] = dC
#             dC = None
        
#         ddelta = rearrange(ddelta, "b d l -> d (b l)")
#         ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
#         dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
#         dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
#         dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
#         dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
#         dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        
#         dx, dconv1d_weight, dconv1d_bias = causal_conv1d_cuda.causal_conv1d_bwd(
#             x, conv1d_weight, conv1d_bias, dconv1d_out, dx, True
#         )
        
#         dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
#         dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        
#         return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
#                 dA, dB, dC, dD,
#                 ddelta_bias if delta_bias is not None else None,
#                 dB_proj_bias, dC_proj_bias, None)


# # Upgrade version4
# class MambaInnerFnNoOutProj(torch.autograd.Function):

#     @staticmethod
#     @custom_fwd
#     def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
#                 A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#                 C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1):
#         """
#              xz: (batch, dim, seqlen)
#         """
#         assert checkpoint_lvl in [0, 1]
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         if torch.is_autocast_enabled():
#             x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
#             delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
#         if xz.stride(-1) != 1:
#             xz = xz.contiguous()
#         conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
#         x, z = xz.chunk(2, dim=1)
#         conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
#         conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
#         # We're being very careful here about the layout, to avoid extra transposes.
#         # We want delta to have d as the slowest moving dimension
#         # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
#         x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
#         delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
#         ctx.is_variable_B = B is None
#         ctx.is_variable_C = C is None
#         ctx.B_proj_bias_is_None = B_proj_bias is None
#         ctx.C_proj_bias_is_None = C_proj_bias is None
#         if B is None:  # variable B
#             B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
#             if B_proj_bias is not None:
#                 B = B + B_proj_bias.to(dtype=B.dtype)
#             if not A.is_complex():
#                 B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#             else:
#                 B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#         else:
#             if B.stride(-1) != 1:
#                 B = B.contiguous()
#         if C is None:  # variable C
#             C = x_dbl[:, -d_state:]  # (bl dstate)
#             if C_proj_bias is not None:
#                 C = C + C_proj_bias.to(dtype=C.dtype)
#             if not A.is_complex():
#                 C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#             else:
#                 C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#         else:
#             if C.stride(-1) != 1:
#                 C = C.contiguous()
#         if D is not None:
#             D = D.contiguous()
#         out, scan_intermediates, out_z = selective_scan_cuda.fwd(
#             conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
#         )
#         out = out + conv1d_out  # Adding skip connection here
#         ctx.delta_softplus = delta_softplus
#         ctx.checkpoint_lvl = checkpoint_lvl
#         if checkpoint_lvl >= 1:  # Will recompute conv1d_out and delta in the backward pass
#             conv1d_out, delta = None, None
#         ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
#                               delta_proj_weight, conv1d_out, delta,
#                               A, B, C, D, delta_bias, scan_intermediates, out)
#         return out_z  # Ensure this return matches the expected output shape

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, dout):
#         # dout: (batch, seqlen, dim)
#         (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, 
#          conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, out) = ctx.saved_tensors
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         x, z = xz.chunk(2, dim=1)
#         if dout.stride(-1) != 1:
#             dout = dout.contiguous()
#         if ctx.checkpoint_lvl == 1:
#             conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
#             delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
#                               "d (b l) -> b d l", l=L)
#         dout = dout + conv1d_out  # Adjust dout for the skip connection
#         dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
#         dx, dz = dxz.chunk(2, dim=1)
#         dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
#             conv1d_out, delta, A, B, C, D, z, delta_bias, dout, scan_intermediates, out, dz,
#             ctx.delta_softplus,
#             True  # option to recompute out_z
#         )
#         dD = dD if D is not None else None
#         dx_dbl = torch.empty_like(x_dbl)
#         dB_proj_bias = None
#         if ctx.is_variable_B:
#             if not A.is_complex():
#                 dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
#             dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
#             dB = None
#         dC_proj_bias = None
#         if ctx.is_variable_C:
#             if not A.is_complex():
#                 dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
#             dx_dbl[:, -d_state:] = dC  # (bl d)
#             dC = None
#         ddelta = rearrange(ddelta, "b d l -> d (b l)")
#         ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
#         dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
#         dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
#         dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
#         dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
#         dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
#         dx, dconv1d_weight, dconv1d_bias = causal_conv1d_cuda.causal_conv1d_bwd(
#             x, conv1d_weight, conv1d_bias, dconv1d_out, dx, True
#         )
#         dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
#         dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
#         return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
#                 dA, dB, dC, dD,
#                 ddelta_bias if delta_bias is not None else None,
#                 dB_proj_bias, dC_proj_bias, None)



# # Upgrade version3
# class MambaInnerFnNoOutProj(torch.autograd.Function):

#     @staticmethod
#     @custom_fwd
#     def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
#                 A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#                 C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1):
#         """
#              xz: (batch, dim, seqlen)
#         """
#         assert checkpoint_lvl in [0, 1]
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         if torch.is_autocast_enabled():
#             x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
#             delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
#         if xz.stride(-1) != 1:
#             xz = xz.contiguous()
#         conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
#         x, z = xz.chunk(2, dim=1)
#         conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
#         conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
#         # We're being very careful here about the layout, to avoid extra transposes.
#         # We want delta to have d as the slowest moving dimension
#         # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
#         x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
#         delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l = L)
#         ctx.is_variable_B = B is None
#         ctx.is_variable_C = C is None
#         ctx.B_proj_bias_is_None = B_proj_bias is None
#         ctx.C_proj_bias_is_None = C_proj_bias is None
#         if B is None:  # variable B
#             B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
#             if B_proj_bias is not None:
#                 B = B + B_proj_bias.to(dtype=B.dtype)
#             if not A.is_complex():
#                 # B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
#                 B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#             else:
#                 B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#         else:
#             if B.stride(-1) != 1:
#                 B = B.contiguous()
#         if C is None:  # variable C
#             C = x_dbl[:, -d_state:]  # (bl dstate)
#             if C_proj_bias is not None:
#                 C = C + C_proj_bias.to(dtype=C.dtype)
#             if not A.is_complex():
#                 # C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
#                 C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#             else:
#                 C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#         else:
#             if C.stride(-1) != 1:
#                 C = C.contiguous()
#         if D is not None:
#             D = D.contiguous()
#         out, scan_intermediates, out_z = selective_scan_cuda.fwd(
#             conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
#         )
        
#         # Skip connection
#         out_z = out_z + conv1d_out
        
#         ctx.delta_softplus = delta_softplus
#         ctx.checkpoint_lvl = checkpoint_lvl
#         if checkpoint_lvl >= 1:  # Will recompute conv1d_out and delta in the backward pass
#             conv1d_out, delta = None, None
#         ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
#                               delta_proj_weight, conv1d_out, delta,
#                               A, B, C, D, delta_bias, scan_intermediates, out)
#         # return rearrange(out_z, "b d l -> b l d")
#         return out_z

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, dout):
#         # dout: (batch, seqlen, dim)
#         (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, 
#          conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, out) = ctx.saved_tensors
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         x, z = xz.chunk(2, dim=1)
#         if dout.stride(-1) != 1:
#             dout = dout.contiguous()
#         if ctx.checkpoint_lvl == 1:
#             conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
#             delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
#                               "d (b l) -> b d l", l = L)
#         # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
#         # backward of selective_scan_cuda with the backward of chunk).
#         dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
#         dx, dz = dxz.chunk(2, dim=1)
#         # dout_y = rearrange(dout, "b l d -> b d l") # because no arrange at end of forward, so dout shape is b d l
#         dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
#             conv1d_out, delta, A, B, C, D, z, delta_bias, dout, scan_intermediates, out, dz,
#             ctx.delta_softplus,
#             True  # option to recompute out_z
#         )
#         dD = dD if D is not None else None
#         dx_dbl = torch.empty_like(x_dbl)
#         dB_proj_bias = None
#         if ctx.is_variable_B:
#             if not A.is_complex():
#                 dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
#             dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
#             dB = None
#         dC_proj_bias = None
#         if ctx.is_variable_C:
#             if not A.is_complex():
#                 dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
#             dx_dbl[:, -d_state:] = dC  # (bl d)
#             dC = None
#         ddelta = rearrange(ddelta, "b d l -> d (b l)")
#         ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
#         dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
#         dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
#         dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
#         dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
#         dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
#         # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
#         # backward of conv1d with the backward of chunk).
#         dx, dconv1d_weight, dconv1d_bias = causal_conv1d_cuda.causal_conv1d_bwd(
#             x, conv1d_weight, conv1d_bias, dconv1d_out, dx, True
#         )
#         dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
#         dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
#         return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
#                 dA, dB, dC, dD,
#                 ddelta_bias if delta_bias is not None else None,
#                 dB_proj_bias, dC_proj_bias, None)



# # Upgrade version2
# class MambaInnerFnNoOutProj(torch.autograd.Function):

#     @staticmethod
#     @custom_fwd
#     def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
#                 A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#                 C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1):
#         """
#              xz: (batch, dim, seqlen)
#         """
#         assert checkpoint_lvl in [0, 1]
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         if torch.is_autocast_enabled():
#             x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
#             delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
#         if xz.stride(-1) != 1:
#             xz = xz.contiguous()
#         conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
#         x, z = xz.chunk(2, dim=1)
#         conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
#         conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)

#         x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
#         delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
        
#         ctx.is_variable_B = B is None
#         ctx.is_variable_C = C is None
#         ctx.B_proj_bias_is_None = B_proj_bias is None
#         ctx.C_proj_bias_is_None = C_proj_bias is None
#         if B is None:
#             B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
#             if B_proj_bias is not None:
#                 B = B + B_proj_bias.to(dtype=B.dtype)
#             if not A.is_complex():
#                 B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#             else:
#                 B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#         else:
#             if B.stride(-1) != 1:
#                 B = B.contiguous()
#         if C is None:
#             C = x_dbl[:, -d_state:]  # (bl dstate)
#             if C_proj_bias is not None:
#                 C = C + C_proj_bias.to(dtype=C.dtype)
#             if not A.is_complex():
#                 C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#             else:
#                 C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#         else:
#             if C.stride(-1) != 1:
#                 C = C.contiguous()
#         if D is not None:
#             D = D.contiguous()
        
#         out, scan_intermediates, out_z = selective_scan_cuda.fwd(
#             conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
#         )
        
#         # Skip connection
#         out_z = out_z + conv1d_out
        
#         ctx.delta_softplus = delta_softplus
#         ctx.checkpoint_lvl = checkpoint_lvl
#         if checkpoint_lvl >= 1:
#             conv1d_out, delta = None, None
#         ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
#                               delta_proj_weight, conv1d_out, delta,
#                               A, B, C, D, delta_bias, scan_intermediates, out, out_z)
#         return out_z

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, dout):
#         # Unpack saved tensors
#         (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, 
#          conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, out, out_z) = ctx.saved_tensors
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         x, z = xz.chunk(2, dim=1)
        
#         if dout.stride(-1) != 1:
#             dout = dout.contiguous()
        
#         if ctx.checkpoint_lvl == 1:
#             conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
#             delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
#                               "d (b l) -> b d l", l=L)
        
#         # Adjust dout for skip connection
#         dout = dout + conv1d_out
        
#         dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
#         dx, dz = dxz.chunk(2, dim=1)
#         dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
#             conv1d_out, delta, A, B, C, D, z, delta_bias, dout, scan_intermediates, out, dz,
#             ctx.delta_softplus,
#             True  # option to recompute out_z
#         )
        
#         dD = dD if D is not None else None
#         dx_dbl = torch.empty_like(x_dbl)
#         dB_proj_bias = None
        
#         if ctx.is_variable_B:
#             if not A.is_complex():
#                 dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
#             dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
#             dB = None
        
#         dC_proj_bias = None
#         if ctx.is_variable_C:
#             if not A.is_complex():
#                 dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
#             dx_dbl[:, -d_state:] = dC  # (bl d)
#             dC = None
        
#         ddelta = rearrange(ddelta, "b d l -> d (b l)")
#         ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
#         dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
#         dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
#         dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
#         dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
#         dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        
#         dx, dconv1d_weight, dconv1d_bias = causal_conv1d_cuda.causal_conv1d_bwd(
#             x, conv1d_weight, conv1d_bias, dconv1d_out, dx, True
#         )
        
#         dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
#         dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        
#         return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
#                 dA, dB, dC, dD,
#                 ddelta_bias if delta_bias is not None else None,
#                 dB_proj_bias, dC_proj_bias, None)



# # Upgrade version
# class MambaInnerFnNoOutProj(torch.autograd.Function):

#     @staticmethod
#     @custom_fwd
#     def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
#                 A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#                 C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1):
#         assert checkpoint_lvl in [0, 1]
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         if torch.is_autocast_enabled():
#             dtype = torch.get_autocast_gpu_dtype()
#             x_proj_weight = x_proj_weight.to(dtype=dtype)
#             delta_proj_weight = delta_proj_weight.to(dtype=dtype)
#             if B_proj_bias is not None:
#                 B_proj_bias = B_proj_bias.to(dtype=dtype)
#             if C_proj_bias is not None:
#                 C_proj_bias = C_proj_bias.to(dtype=dtype)
#         if xz.stride(-1) != 1:
#             xz = xz.contiguous()
#         conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
#         x, z = xz.chunk(2, dim=1)
#         conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
#         conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)

#         x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)
#         delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
        
#         ctx.is_variable_B = B is None
#         ctx.is_variable_C = C is None
#         ctx.B_proj_bias_is_None = B_proj_bias is None
#         ctx.C_proj_bias_is_None = C_proj_bias is None
#         if B is None:
#             B = x_dbl[:, delta_rank:delta_rank + d_state]
#             if B_proj_bias is not None:
#                 B = B + B_proj_bias
#             if not A.is_complex():
#                 B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#             else:
#                 B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#         else:
#             if B.stride(-1) != 1:
#                 B = B.contiguous()
#         if C is None:
#             C = x_dbl[:, -d_state:]
#             if C_proj_bias is not None:
#                 C = C + C_proj_bias
#             if not A.is_complex():
#                 C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#             else:
#                 C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#         else:
#             if C.stride(-1) != 1:
#                 C = C.contiguous()
#         if D is not None:
#             D = D.contiguous()
        
#         out, scan_intermediates, out_z = selective_scan_cuda.fwd(
#             conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
#         )
        
#         # Skip connection
#         out_z = out_z + conv1d_out
        
#         ctx.delta_softplus = delta_softplus
#         ctx.checkpoint_lvl = checkpoint_lvl
#         if checkpoint_lvl >= 1:
#             conv1d_out, delta = None, None
#         ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
#                               delta_proj_weight, conv1d_out, delta,
#                               A, B, C, D, delta_bias, scan_intermediates, out, out_z)
#         return out_z

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, dout):
#         (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, 
#         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, out, out_z) = ctx.saved_tensors
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         x, z = xz.chunk(2, dim=1)
        
#         if dout.stride(-1) != 1:
#             dout = dout.contiguous()
        
#         if ctx.checkpoint_lvl == 1:
#             conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
#             delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
#                             "d (b l) -> b d l", l=L)
        
#         # # Include skip connection in gradient
#         # dconv1d_out = dout
        
        
#         dxz = torch.empty_like(xz)
#         dx, dz = dxz.chunk(2, dim=1)
#         # dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
#         #     conv1d_out, delta, A, B, C, D, z, delta_bias, dconv1d_out, scan_intermediates, out, dz,
#         #     ctx.delta_softplus,
#         #     True
#         # )
        
#         dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
#             conv1d_out, delta, A, B, C, D, z, delta_bias, dout, scan_intermediates, out, dz,
#             ctx.delta_softplus,
#             True  # option to recompute out_z
#         )
        
#         dD = dD if D is not None else None
#         dx_dbl = torch.empty_like(x_dbl)
#         dB_proj_bias = None
        
#         if ctx.is_variable_B:
#             if not A.is_complex():
#                 dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
#             dx_dbl[:, delta_rank:delta_rank + d_state] = dB
#             dB = None
        
#         dC_proj_bias = None
#         if ctx.is_variable_C:
#             if not A.is_complex():
#                 dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
#             dx_dbl[:, -d_state:] = dC
#             dC = None
        
#         ddelta = rearrange(ddelta, "b d l -> d (b l)")
#         ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
#         dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
#         dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
#         dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
#         dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
#         dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        
#         dx, dconv1d_weight, dconv1d_bias = causal_conv1d_cuda.causal_conv1d_bwd(
#             x, conv1d_weight, conv1d_bias, dconv1d_out, dx, True
#         )
        
#         dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
#         dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        
#         return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
#                 dA, dB, dC, dD,
#                 ddelta_bias if delta_bias is not None else None,
#                 dB_proj_bias, dC_proj_bias, None)



# class MambaInnerFnNoOutProj(torch.autograd.Function):

#     @staticmethod
#     @custom_fwd
#     def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
#                 A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#                 C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1):
#         """
#              xz: (batch, dim, seqlen)
#         """
#         assert checkpoint_lvl in [0, 1]
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         if torch.is_autocast_enabled():
#             x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
#             delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
#         if xz.stride(-1) != 1:
#             xz = xz.contiguous()
#         conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
#         x, z = xz.chunk(2, dim=1)
#         conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
#         conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
#         # We're being very careful here about the layout, to avoid extra transposes.
#         # We want delta to have d as the slowest moving dimension
#         # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
#         x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
#         delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l = L)
#         ctx.is_variable_B = B is None
#         ctx.is_variable_C = C is None
#         ctx.B_proj_bias_is_None = B_proj_bias is None
#         ctx.C_proj_bias_is_None = C_proj_bias is None
#         if B is None:  # variable B
#             B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
#             if B_proj_bias is not None:
#                 B = B + B_proj_bias.to(dtype=B.dtype)
#             if not A.is_complex():
#                 # B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
#                 B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#             else:
#                 B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#         else:
#             if B.stride(-1) != 1:
#                 B = B.contiguous()
#         if C is None:  # variable C
#             C = x_dbl[:, -d_state:]  # (bl dstate)
#             if C_proj_bias is not None:
#                 C = C + C_proj_bias.to(dtype=C.dtype)
#             if not A.is_complex():
#                 # C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
#                 C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#             else:
#                 C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#         else:
#             if C.stride(-1) != 1:
#                 C = C.contiguous()
#         if D is not None:
#             D = D.contiguous()
#         out, scan_intermediates, out_z = selective_scan_cuda.fwd(
#             conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
#         )
#         ctx.delta_softplus = delta_softplus
#         ctx.checkpoint_lvl = checkpoint_lvl
#         if checkpoint_lvl >= 1:  # Will recompute conv1d_out and delta in the backward pass
#             conv1d_out, delta = None, None
#         ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
#                               delta_proj_weight, conv1d_out, delta,
#                               A, B, C, D, delta_bias, scan_intermediates, out)
#         # return rearrange(out_z, "b d l -> b l d")
#         return out_z

#     @staticmethod
#     @custom_bwd
#     def backward(ctx, dout):
#         # dout: (batch, seqlen, dim)
#         (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, 
#          conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, out) = ctx.saved_tensors
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         x, z = xz.chunk(2, dim=1)
#         if dout.stride(-1) != 1:
#             dout = dout.contiguous()
#         if ctx.checkpoint_lvl == 1:
#             conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
#             delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
#                               "d (b l) -> b d l", l = L)
#         # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
#         # backward of selective_scan_cuda with the backward of chunk).
#         dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
#         dx, dz = dxz.chunk(2, dim=1)
#         # dout_y = rearrange(dout, "b l d -> b d l") # because no arrange at end of forward, so dout shape is b d l
#         dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
#             conv1d_out, delta, A, B, C, D, z, delta_bias, dout, scan_intermediates, out, dz,
#             ctx.delta_softplus,
#             True  # option to recompute out_z
#         )
#         dD = dD if D is not None else None
#         dx_dbl = torch.empty_like(x_dbl)
#         dB_proj_bias = None
#         if ctx.is_variable_B:
#             if not A.is_complex():
#                 dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
#             dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
#             dB = None
#         dC_proj_bias = None
#         if ctx.is_variable_C:
#             if not A.is_complex():
#                 dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
#             dx_dbl[:, -d_state:] = dC  # (bl d)
#             dC = None
#         ddelta = rearrange(ddelta, "b d l -> d (b l)")
#         ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
#         dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
#         dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
#         dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
#         dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
#         dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
#         # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
#         # backward of conv1d with the backward of chunk).
#         dx, dconv1d_weight, dconv1d_bias = causal_conv1d_cuda.causal_conv1d_bwd(
#             x, conv1d_weight, conv1d_bias, dconv1d_out, dx, True
#         )
#         dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
#         dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
#         return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
#                 dA, dB, dC, dD,
#                 ddelta_bias if delta_bias is not None else None,
#                 dB_proj_bias, dC_proj_bias, None)



# class MambaInnerFnNoOutProj(torch.autograd.Function):

#     @staticmethod
#     @custom_fwd
#     def forward(ctx, xz, conv1ds, x_proj_weight, delta_proj_weight,
#                 A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#                 C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1, num_iterations=4):
#         assert checkpoint_lvl in [0, 1]
#         num_iterations = min(num_iterations, len(conv1ds))  # num_iterations 값을 conv1ds 길이로 제한
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         if torch.is_autocast_enabled():
#             dtype = torch.get_autocast_gpu_dtype()
#             x_proj_weight = x_proj_weight.to(dtype=dtype)
#             delta_proj_weight = delta_proj_weight.to(dtype=dtype)
#             if B_proj_bias is not None:
#                 B_proj_bias = B_proj_bias.to(dtype=dtype)
#             if C_proj_bias is not None:
#                 C_proj_bias = C_proj_bias.to(dtype=dtype)
#         if xz.stride(-1) != 1:
#             xz = xz.contiguous()
#         x, z = xz.chunk(2, dim=1)
        
#         ctx.is_variable_B = B is None
#         ctx.is_variable_C = C is None
#         ctx.B_proj_bias_is_None = B_proj_bias is None
#         ctx.C_proj_bias_is_None = C_proj_bias is None
        
#         # None인 경우 Zero Tensor로 대체
#         B = torch.zeros_like(A) if B is None else B
#         C = torch.zeros_like(A) if C is None else C
#         D = torch.zeros_like(A) if D is None else D
#         delta_bias = torch.zeros(1, device=xz.device) if delta_bias is None else delta_bias
#         B_proj_bias = torch.zeros_like(A) if B_proj_bias is None else B_proj_bias
#         C_proj_bias = torch.zeros_like(A) if C_proj_bias is None else C_proj_bias
        
#         # 병렬 처리된 결과를 저장할 리스트
#         out_z_list = []
#         conv1d_out_list = []
#         delta_list = []

#         for i in range(num_iterations):
#             conv1d = conv1ds[i]
#             conv1d_weight = conv1d.weight
#             conv1d_bias = conv1d.bias
            
#             # conv1d_weight의 폭이 2에서 4 사이인지 확인
#             if (conv1d_weight.shape[2] < 2) or (conv1d_weight.shape[2] > 4):
#                 raise ValueError(f"conv1d_weight width should be between 2 and 4, but got {conv1d_weight.shape[2]}")

#             conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
#             conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
#             conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
#             conv1d_out_list.append(conv1d_out)

#             # 병렬 처리 가능한 연산들
#             x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)
#             delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
#             delta_list.append(delta)
            
#             if B is None:
#                 B = x_dbl[:, delta_rank:delta_rank + d_state]
#                 if B_proj_bias is not None:
#                     B = B + B_proj_bias
#                 if not A.is_complex():
#                     B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#                 else:
#                     B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#             else:
#                 if B.stride(-1) != 1:
#                     B = B.contiguous()
#             if C is None:
#                 C = x_dbl[:, -d_state:]
#                 if C_proj_bias is not None:
#                     C = C + C_proj_bias
#                 if not A.is_complex():
#                     C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#                 else:
#                     C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#             else:
#                 if C.stride(-1) != 1:
#                     C = C.contiguous()
#             if D is not None:
#                 D = D.contiguous()

#             out, scan_intermediates, out_z = selective_scan_cuda.fwd(
#                 conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
#             )

#             out_z_list.append(out_z)

#         final_out_z = torch.cat(out_z_list, dim=1)  # concat 연산으로 변경
        
#         ctx.delta_softplus = delta_softplus
#         ctx.checkpoint_lvl = checkpoint_lvl
#         ctx.out_z_list = out_z_list  # out_z_list를 ctx에 저장
#         ctx.conv1ds = conv1ds
#         ctx.num_iterations = num_iterations
#         ctx.delta_list = delta_list

#         # conv1d_out과 delta를 저장하거나, 체크포인트 레벨에 따라 재계산할 수 있도록 설정
#         ctx.save_for_backward(xz, x_proj_weight, delta_proj_weight,
#                             A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, scan_intermediates, out, final_out_z, *conv1d_out_list)

#         if checkpoint_lvl >= 1:
#             ctx.conv1d_out_list = None
#         else:
#             ctx.conv1d_out_list = conv1d_out_list

#         return final_out_z


#     @staticmethod
#     @custom_bwd
#     def backward(ctx, dout):
#         saved_tensors = ctx.saved_tensors
#         (xz, x_proj_weight, delta_proj_weight,
#         A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, scan_intermediates, out, final_out_z, *conv1d_out_list) = saved_tensors
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         x, z = xz.chunk(2, dim=1)
#         if dout.stride(-1) != 1:
#             dout = dout.contiguous()
#         if ctx.checkpoint_lvl == 1:
#             conv1d_out_list = []
#             delta_list = []
#             for i in range(ctx.num_iterations):
#                 conv1d = ctx.conv1ds[i]
#                 conv1d_weight = conv1d.weight
#                 conv1d_bias = conv1d.bias
#                 conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
#                 conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
#                 conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
#                 conv1d_out_list.append(conv1d_out)

#                 x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)
#                 delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
#                 delta_list.append(delta)
#         else:
#             delta_list = ctx.delta_list
        
#         dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
#         dx, dz = dxz.chunk(2, dim=1)
        
#         dconv1d_out_list = []
#         ddelta_list = []
#         dA_list = []
#         dB_list = []
#         dC_list = []
#         dD_list = []
#         ddelta_bias_list = []
#         dz_list = []
        
#         # out_z_list는 concat되어 있으므로 분리
#         out_z_splits = torch.split(dout, dout.size(1) // len(ctx.out_z_list), dim=1)
        
#         for i, out_z in enumerate(ctx.out_z_list):
#             dout_part = out_z_splits[i]
#             dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, _ = selective_scan_cuda.bwd(
#                 conv1d_out_list[i], delta_list[i], A, B, C, D, z, delta_bias, dout_part,
#                 scan_intermediates, out, dz, ctx.delta_softplus, True  # recompute out_z
#             )
#             dconv1d_out_list.append(dconv1d_out)
#             ddelta_list.append(ddelta)
#             dA_list.append(dA)
#             dB_list.append(dB)
#             dC_list.append(dC)
#             dD_list.append(dD)
#             ddelta_bias_list.append(ddelta_bias)
#             dz_list.append(dz)
        
#         dconv1d_out = sum(dconv1d_out_list)
#         ddelta = sum(ddelta_list)
#         dA = sum(dA_list)
#         dB = sum(dB_list) if ctx.is_variable_B else None
#         dC = sum(dC_list) if ctx.is_variable_C else None
#         dD = sum(dD_list) if D is not None else None
#         ddelta_bias = sum(ddelta_bias_list) if delta_bias is not None else None
        
#         x_dbl = F.linear(rearrange(dconv1d_out, 'b d l -> (b l) d'), x_proj_weight)
#         dx_dbl = torch.empty_like(x_dbl)
#         dB_proj_bias = None
#         if ctx.is_variable_B:
#             if dB.dim() == 2:
#                 dB = dB.unsqueeze(1).unsqueeze(1)  # Ensure dB has 4 dimensions
#             if not A.is_complex():
#                 dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
#             dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
#             dB = None
#         dC_proj_bias = None
#         if ctx.is_variable_C:
#             if dC.dim() == 2:
#                 dC = dC.unsqueeze(1).unsqueeze(1)  # Ensure dC has 4 dimensions
#             if not A.is_complex():
#                 dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
#             dx_dbl[:, -d_state:] = dC  # (bl d)
#             dC = None
#         ddelta = rearrange(ddelta, "b d l -> d (b l)")
#         ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
#         dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
#         dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
#         dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out_list[0], "b d l -> (b l) d"))  # conv1d_out_list[0]으로 수정
#         dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
#         dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
#         dconv1d_weight, dconv1d_bias = [], []

#         for i in range(ctx.num_iterations):
#             weight = rearrange(ctx.conv1ds[i].weight, "d 1 w -> d w")
#             bias = ctx.conv1ds[i].bias
#             dconv1d_out_i = dconv1d_out_list[i]
#             dx, dweight, dbias = causal_conv1d_cuda.causal_conv1d_bwd(
#                 x, weight, bias, dconv1d_out_i, dx, True
#             )
#             dconv1d_weight.append(rearrange(dweight, "d w -> d 1 w"))
#             dconv1d_bias.append(dbias if bias is not None else None)

#         # 올바른 반환 형식으로 변경
#         return (dxz, None, None, dx_proj_weight, ddelta_proj_weight,
#                 dA, dB if ctx.is_variable_B else None, dC if ctx.is_variable_C else None, dD if D is not None else None,
#                 ddelta_bias if delta_bias is not None else None, dB_proj_bias, dC_proj_bias, None, None, None)

# class MambaInnerFnNoOutProj(torch.autograd.Function):

#     @staticmethod
#     @custom_fwd
#     def forward(ctx, xz, conv1ds, x_proj_weight, delta_proj_weight,
#                 A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#                 C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1, num_iterations=4):
#         assert checkpoint_lvl in [0, 1]
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         if torch.is_autocast_enabled():
#             dtype = torch.get_autocast_gpu_dtype()
#             x_proj_weight = x_proj_weight.to(dtype=dtype)
#             delta_proj_weight = delta_proj_weight.to(dtype=dtype)
#             if B_proj_bias is not None:
#                 B_proj_bias = B_proj_bias.to(dtype=dtype)
#             if C_proj_bias is not None:
#                 C_proj_bias = C_proj_bias.to(dtype=dtype)
#         if xz.stride(-1) != 1:
#             xz = xz.contiguous()
#         x, z = xz.chunk(2, dim=1)
        
#         ctx.is_variable_B = B is None
#         ctx.is_variable_C = C is None
#         ctx.B_proj_bias_is_None = B_proj_bias is None
#         ctx.C_proj_bias_is_None = C_proj_bias is None
        
#         # 병렬 처리된 결과를 저장할 리스트
#         out_z_list = []
#         conv1d_out_list = []

#         for i in range(num_iterations):
#             conv1d_weight = conv1ds[i].weight
#             conv1d_bias = conv1ds[i].bias
#             conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
#             conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
#             conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
#             conv1d_out_list.append(conv1d_out)

#             # 병렬 처리 가능한 연산들
#             x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)
#             delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=L)
            
#             if B is None:
#                 B = x_dbl[:, delta_rank:delta_rank + d_state]
#                 if B_proj_bias is not None:
#                     B = B + B_proj_bias
#                 if not A.is_complex():
#                     B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#                 else:
#                     B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#             else:
#                 if B.stride(-1) != 1:
#                     B = B.contiguous()
#             if C is None:
#                 C = x_dbl[:, -d_state:]
#                 if C_proj_bias is not None:
#                     C = C + C_proj_bias
#                 if not A.is_complex():
#                     C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
#                 else:
#                     C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
#             else:
#                 if C.stride(-1) != 1:
#                     C = C.contiguous()
#             if D is not None:
#                 D = D.contiguous()

#             out, scan_intermediates, out_z = selective_scan_cuda.fwd(
#                 conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
#             )

#             out_z_list.append(out_z)

#         final_out_z = torch.cat(out_z_list, dim=1)  # concat 연산으로 변경
        
#         ctx.delta_softplus = delta_softplus
#         ctx.checkpoint_lvl = checkpoint_lvl
#         ctx.out_z_list = out_z_list  # out_z_list를 ctx에 저장

#         # conv1d_out과 delta를 저장하거나, 체크포인트 레벨에 따라 재계산할 수 있도록 설정
#         ctx.save_for_backward(xz, conv1ds, x_proj_weight, delta_proj_weight,
#                             A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, scan_intermediates, out, final_out_z, *conv1d_out_list)

#         if checkpoint_lvl >= 1:
#             ctx.conv1d_out_list = None
#             ctx.delta = None
#         else:
#             ctx.conv1d_out_list = conv1d_out_list
#             ctx.delta = delta

#         return final_out_z


#     @staticmethod
#     @custom_bwd
#     def backward(ctx, dout):
#         saved_tensors = ctx.saved_tensors
#         (xz, conv1ds, x_proj_weight, delta_proj_weight,
#         A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, scan_intermediates, out, out_z, *conv1d_out_list) = saved_tensors
#         L = xz.shape[-1]
#         delta_rank = delta_proj_weight.shape[1]
#         d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
#         x, z = xz.chunk(2, dim=1)
#         if dout.stride(-1) != 1:
#             dout = dout.contiguous()
#         if ctx.checkpoint_lvl == 1:
#             conv1d_out_list = []
#             for i in range(ctx.num_iterations):
#                 conv1d_weight = conv1ds[i].weight
#                 conv1d_bias = conv1ds[i].bias
#                 conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
#                 conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
#                 conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
#                 conv1d_out_list.append(conv1d_out)
        
#         dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
#         dx, dz = dxz.chunk(2, dim=1)
        
#         dconv1d_out_list = []
#         ddelta_list = []
#         dA_list = []
#         dB_list = []
#         dC_list = []
#         dD_list = []
#         ddelta_bias_list = []
#         dz_list = []
        
#         # out_z_list는 concat되어 있으므로 분리
#         out_z_splits = torch.split(dout, dout.size(1) // len(ctx.out_z_list), dim=1)
        
#         for i, out_z in enumerate(ctx.out_z_list):
#             dout_part = out_z_splits[i]
#             dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, _ = selective_scan_cuda.bwd(
#                 conv1d_out_list[i], delta, A, B, C, D, z, delta_bias, dout_part, scan_intermediates, out, dz,
#                 ctx.delta_softplus,
#                 True  # option to recompute out_z
#             )
#             dconv1d_out_list.append(dconv1d_out)
#             ddelta_list.append(ddelta)
#             dA_list.append(dA)
#             dB_list.append(dB)
#             dC_list.append(dC)
#             dD_list.append(dD)
#             ddelta_bias_list.append(ddelta_bias)
#             dz_list.append(dz)
        
#         dconv1d_out = sum(dconv1d_out_list)
#         ddelta = sum(ddelta_list)
#         dA = sum(dA_list)
#         dB = sum(dB_list)
#         dC = sum(dC_list)
#         dD = sum(dD_list)
#         ddelta_bias = sum(ddelta_bias_list) if delta_bias is not None else None
        
#         x_dbl = F.linear(rearrange(dconv1d_out, 'b d l -> (b l) d'), x_proj_weight)
#         dx_dbl = torch.empty_like(x_dbl)
#         dB_proj_bias = None
#         if ctx.is_variable_B:
#             if not A.is_complex():
#                 dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
#             dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
#             dB = None
#         dC_proj_bias = None
#         if ctx.is_variable_C:
#             if not A.is_complex():
#                 dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
#             else:
#                 dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
#             dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
#             dx_dbl[:, -d_state:] = dC  # (bl d)
#             dC = None
#         ddelta = rearrange(ddelta, "b d l -> d (b l)")
#         ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
#         dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
#         dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
#         dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
#         dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
#         dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
#         dx, dconv1d_weight, dconv1d_bias = causal_conv1d_cuda.causal_conv1d_bwd(
#             x, conv1d_weight, conv1d_bias, dconv1d_out, dx, True
#         )
#         dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
#         dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
#         return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
#                 dA, dB, dC, dD,
#                 ddelta_bias if delta_bias is not None else None,
#                 dB_proj_bias, dC_proj_bias, None, None, None)





class MambaInnerFn(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                out_proj_weight, out_proj_bias,
                A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
                C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1):
        """
             xz: (batch, dim, seqlen)
        """
        assert checkpoint_lvl in [0, 1]
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        if torch.is_autocast_enabled():
            x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_weight = out_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_bias = (out_proj_bias.to(dtype=torch.get_autocast_gpu_dtype())
                             if out_proj_bias is not None else None)
        if xz.stride(-1) != 1:
            xz = xz.contiguous()
        conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
        x, z = xz.chunk(2, dim=1)
        conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
        conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
        # We're being very careful here about the layout, to avoid extra transposes.
        # We want delta to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
        delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l = L)
        ctx.is_variable_B = B is None
        ctx.is_variable_C = C is None
        ctx.B_proj_bias_is_None = B_proj_bias is None
        ctx.C_proj_bias_is_None = C_proj_bias is None
        if B is None:  # variable B
            B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
            if B_proj_bias is not None:
                B = B + B_proj_bias.to(dtype=B.dtype)
            if not A.is_complex():
                # B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if B.stride(-1) != 1:
                B = B.contiguous()
        if C is None:  # variable C
            C = x_dbl[:, -d_state:]  # (bl dstate)
            if C_proj_bias is not None:
                C = C + C_proj_bias.to(dtype=C.dtype)
            if not A.is_complex():
                # C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if C.stride(-1) != 1:
                C = C.contiguous()
        if D is not None:
            D = D.contiguous()
        out, scan_intermediates, out_z = selective_scan_cuda.fwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
        )
        ctx.delta_softplus = delta_softplus
        ctx.out_proj_bias_is_None = out_proj_bias is None
        ctx.checkpoint_lvl = checkpoint_lvl
        if checkpoint_lvl >= 1:  # Will recompute conv1d_out and delta in the backward pass
            conv1d_out, delta = None, None
        ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
                              delta_proj_weight, out_proj_weight, conv1d_out, delta,
                              A, B, C, D, delta_bias, scan_intermediates, out)
        return F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)

    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        # dout: (batch, seqlen, dim)
        (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, out_proj_weight,
         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, out) = ctx.saved_tensors
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        x, z = xz.chunk(2, dim=1)
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if ctx.checkpoint_lvl == 1:
            conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
            delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
                              "d (b l) -> b d l", l = L)
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
        dx, dz = dxz.chunk(2, dim=1)
        dout = rearrange(dout, "b l e -> e (b l)")
        dout_y = rearrange(out_proj_weight.t() @ dout, "d (b l) -> b d l", l=L)
        dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, dout_y, scan_intermediates, out, dz,
            ctx.delta_softplus,
            True  # option to recompute out_z
        )
        dout_proj_weight = torch.einsum("eB,dB->ed", dout, rearrange(out_z, "b d l -> d (b l)"))
        dout_proj_bias = dout.sum(dim=(0, 1)) if not ctx.out_proj_bias_is_None else None
        dD = dD if D is not None else None
        dx_dbl = torch.empty_like(x_dbl)
        dB_proj_bias = None
        if ctx.is_variable_B:
            if not A.is_complex():
                dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
            dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
            dB = None
        dC_proj_bias = None
        if ctx.is_variable_C:
            if not A.is_complex():
                dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
            dx_dbl[:, -d_state:] = dC  # (bl d)
            dC = None
        ddelta = rearrange(ddelta, "b d l -> d (b l)")
        ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
        dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
        dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
        dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
        dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
        dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
        # backward of conv1d with the backward of chunk).
        dx, dconv1d_weight, dconv1d_bias = causal_conv1d_cuda.causal_conv1d_bwd(
            x, conv1d_weight, conv1d_bias, dconv1d_out, dx, True
        )
        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
                dout_proj_weight, dout_proj_bias,
                dA, dB, dC, dD,
                ddelta_bias if delta_bias is not None else None,
                dB_proj_bias, dC_proj_bias, None)


class BiMambaInnerFn(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                out_proj_weight, out_proj_bias,
                A, A_b, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
                C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1):
        """
             xz: (batch, dim, seqlen)
        """
        assert checkpoint_lvl in [0, 1]
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        if torch.is_autocast_enabled():
            x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_weight = out_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_bias = (out_proj_bias.to(dtype=torch.get_autocast_gpu_dtype())
                             if out_proj_bias is not None else None)
        if xz.stride(-1) != 1:
            xz = xz.contiguous()
        conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
        x, z = xz.chunk(2, dim=1)
        conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
        conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
        # We're being very careful here about the layout, to avoid extra transposes.
        # We want delta to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
        delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l = L)
        ctx.is_variable_B = B is None
        ctx.is_variable_C = C is None
        ctx.B_proj_bias_is_None = B_proj_bias is None
        ctx.C_proj_bias_is_None = C_proj_bias is None
        if B is None:  # variable B
            B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
            if B_proj_bias is not None:
                B = B + B_proj_bias.to(dtype=B.dtype)
            if not A.is_complex():
                # B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if B.stride(-1) != 1:
                B = B.contiguous()
        if C is None:  # variable C
            C = x_dbl[:, -d_state:]  # (bl dstate)
            if C_proj_bias is not None:
                C = C + C_proj_bias.to(dtype=C.dtype)
            if not A.is_complex():
                # C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if C.stride(-1) != 1:
                C = C.contiguous()
        if D is not None:
            D = D.contiguous()
        out_f, scan_intermediates_f, out_z_f = selective_scan_cuda.fwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
        )
        assert not A_b.is_complex(), "A should not be complex!!"
        out_b, scan_intermediates_b, out_z_b = selective_scan_cuda.fwd(
            conv1d_out.flip([-1]), delta.flip([-1]), A_b, B.flip([-1]), C.flip([-1]), D, z.flip([-1]), delta_bias, delta_softplus,
        )

        out_z = out_z_f + out_z_b.flip([-1])

        ctx.delta_softplus = delta_softplus
        ctx.out_proj_bias_is_None = out_proj_bias is None
        ctx.checkpoint_lvl = checkpoint_lvl
        if checkpoint_lvl >= 1:  # Will recompute conv1d_out and delta in the backward pass
            conv1d_out, delta = None, None
        ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
                              delta_proj_weight, out_proj_weight, conv1d_out, delta,
                              A, A_b, B, C, D, delta_bias, scan_intermediates_f, scan_intermediates_b, out_f, out_b)
        return F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)

    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        # dout: (batch, seqlen, dim)
        (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, out_proj_weight,
         conv1d_out, delta, A, A_b, B, C, D, delta_bias, scan_intermediates_f, scan_intermediates_b, out_f, out_b) = ctx.saved_tensors
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        x, z = xz.chunk(2, dim=1)
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if ctx.checkpoint_lvl == 1:
            conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(x, conv1d_weight, conv1d_bias, True)
            delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
                              "d (b l) -> b d l", l = L)
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
        dx, dz = dxz.chunk(2, dim=1)
        dout = rearrange(dout, "b l e -> e (b l)")
        dout_y = rearrange(out_proj_weight.t() @ dout, "d (b l) -> b d l", l=L)
        dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z_f = selective_scan_cuda.bwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, dout_y, scan_intermediates_f, out_f, dz,
            ctx.delta_softplus,
            True  # option to recompute out_z
        )
        # flip one
        dz_b = torch.empty_like(dz)
        dconv1d_out_f_b, ddelta_f_b, dA_b, dB_f_b, dC_f_b, dD_b, ddelta_bias_b, dz_b, out_z_b = selective_scan_cuda.bwd(
            conv1d_out.flip([-1]), delta.flip([-1]), A_b, B.flip([-1]), C.flip([-1]), D, z.flip([-1]), delta_bias, dout_y.flip([-1]), scan_intermediates_b, out_b, dz_b,
            ctx.delta_softplus,
            True  # option to recompute out_z
        )

        dconv1d_out = dconv1d_out + dconv1d_out_f_b.flip([-1])
        ddelta = ddelta + ddelta_f_b.flip([-1])
        dB = dB + dB_f_b.flip([-1])
        dC = dC + dC_f_b.flip([-1])
        dD = dD + dD_b
        ddelta_bias = ddelta_bias + ddelta_bias_b
        dz = dz + dz_b.flip([-1])
        out_z = out_z_f + out_z_b.flip([-1])
        
        dout_proj_weight = torch.einsum("eB,dB->ed", dout, rearrange(out_z, "b d l -> d (b l)"))
        dout_proj_bias = dout.sum(dim=(0, 1)) if not ctx.out_proj_bias_is_None else None
        dD = dD if D is not None else None
        dx_dbl = torch.empty_like(x_dbl)
        dB_proj_bias = None
        if ctx.is_variable_B:
            if not A.is_complex():
                dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
            dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
            dB = None
        dC_proj_bias = None
        if ctx.is_variable_C:
            if not A.is_complex():
                dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
            dx_dbl[:, -d_state:] = dC  # (bl d)
            dC = None
        ddelta = rearrange(ddelta, "b d l -> d (b l)")
        ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
        dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
        dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
        dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
        dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
        dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
        # backward of conv1d with the backward of chunk).
        dx, dconv1d_weight, dconv1d_bias = causal_conv1d_cuda.causal_conv1d_bwd(
            x, conv1d_weight, conv1d_bias, dconv1d_out, dx, True
        )
        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
                dout_proj_weight, dout_proj_bias,
                dA, dA_b, dB, dC, dD,
                ddelta_bias if delta_bias is not None else None,
                dB_proj_bias, dC_proj_bias, None)
    

def mamba_inner_fn(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True
):
    return MambaInnerFn.apply(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                              out_proj_weight, out_proj_bias,
                              A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus)

def bimamba_inner_fn(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, A_b, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True
):
    return BiMambaInnerFn.apply(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                              out_proj_weight, out_proj_bias,
                              A, A_b, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus)


# def mamba_inner_fn_no_out_proj(
#     xz, conv1d_weights, conv1d_biases, x_proj_weight, delta_proj_weight,
#     A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#     C_proj_bias=None, delta_softplus=True, num_iterations=1
# ):
#     return MambaInnerFnNoOutProj.apply(xz, conv1d_weights, conv1d_biases, x_proj_weight, delta_proj_weight,
#                               A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus, 1, num_iterations)


# def mamba_inner_fn_no_out_proj(
#     xz, conv1ds, x_proj_weight, delta_proj_weight,
#     A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#     C_proj_bias=None, delta_softplus=True, num_iterations=1
# ):
#     conv1d_weights = [conv1d.weight.squeeze(1) for conv1d in conv1ds]
#     conv1d_biases = [conv1d.bias if conv1d.bias is not None else torch.zeros_like(conv1d.weight[0]) for conv1d in conv1ds]
#     return MambaInnerFnNoOutProj.apply(xz, conv1d_weights, conv1d_biases, x_proj_weight, delta_proj_weight,
#                               A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus, 1, num_iterations)

# def mamba_inner_fn_no_out_proj(
#     xz, conv1ds, x_proj_weight, delta_proj_weight,
#     A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#     C_proj_bias=None, delta_softplus=True, num_iterations=1
# ):
#     conv1d_weights = [conv1d.weight.squeeze(1) for conv1d in conv1ds]
#     conv1d_biases = [conv1d.bias if conv1d.bias is not None else None for conv1d in conv1ds]
#     return MambaInnerFnNoOutProj.apply(xz, conv1d_weights, conv1d_biases, x_proj_weight, delta_proj_weight,
#                               A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus, 1, num_iterations)


# def mamba_inner_fn_no_out_proj(
#     xz, conv1ds, x_proj_weight, delta_proj_weight,
#     A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#     C_proj_bias=None, delta_softplus=True, num_iterations=1
# ):
#     return MambaInnerFnNoOutProj.apply(xz, conv1ds, x_proj_weight, delta_proj_weight,
#                               A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus, 1, num_iterations)


# def mamba_inner_fn_no_out_proj(
#     xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
#     A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#     C_proj_bias=None, delta_softplus=True, num_iterations=1
# ):
#     return MambaInnerFnNoOutProj.apply(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
#                               A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus, 1, num_iterations)

# def mamba_inner_fn_no_out_proj(
#     xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
#     A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#     C_proj_bias=None, delta_softplus=True, num_iterations=1
# ):
#     return MambaInnerFnNoOutProj.apply(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
#                               A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus)

# def mamba_inner_fn_no_out_proj(
#     xz, conv1ds, x_proj_weight, delta_proj_weight,
#     A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#     C_proj_bias=None, delta_softplus=True, num_iterations=1
# ):
#     return MambaInnerFnNoOutProj.apply(xz, conv1ds, x_proj_weight, delta_proj_weight,
#                               A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus, 1, num_iterations)


# def mamba_inner_fn_no_out_proj(
#     xz, conv1ds, x_proj_weight, delta_proj_weight,
#     A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#     C_proj_bias=None, delta_softplus=True, num_iterations=1
# ):
#     return MambaInnerFnNoOutProj.apply(xz, conv1ds, x_proj_weight, delta_proj_weight,
#                               A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus, 1, num_iterations)


# def mamba_inner_fn_no_out_proj(
#     xz, conv1ds, x_proj_weight, delta_proj_weight,
#     A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#     C_proj_bias=None, delta_softplus=True, num_iterations=1
# ):
#     return MambaInnerFnNoOutProj.apply(xz, conv1ds, x_proj_weight, delta_proj_weight,
#                               A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus, 1, num_iterations)


# def mamba_inner_fn_no_out_proj(
#     xz, conv1ds, x_proj_weight, delta_proj_weight,
#     A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
#     C_proj_bias=None, delta_softplus=True, num_iterations=1
# ):
#     return MambaInnerFnNoOutProj.apply(xz, conv1ds, x_proj_weight, delta_proj_weight,
#                               A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus, 1, num_iterations)




def mamba_inner_fn_no_out_proj(
    xz, conv1ds, x_proj_weight, delta_proj_weight,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True, num_iterations=1
):
    return MambaInnerFnNoOutProj.apply(xz, conv1ds, x_proj_weight, delta_proj_weight,
                              A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus, 1, num_iterations)



def mamba_inner_ref(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True
):
    L = xz.shape[-1]
    delta_rank = delta_proj_weight.shape[1]
    d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
    x, z = xz.chunk(2, dim=1)
    x = causal_conv1d_fn(x, rearrange(conv1d_weight, "d 1 w -> d w"), conv1d_bias, "silu")
    # We're being very careful here about the layout, to avoid extra transposes.
    # We want delta to have d as the slowest moving dimension
    # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
    x_dbl = F.linear(rearrange(x, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
    delta = delta_proj_weight @ x_dbl[:, :delta_rank].t()
    delta = rearrange(delta, "d (b l) -> b d l", l=L)
    if B is None:  # variable B
        B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl d)
        if B_proj_bias is not None:
            B = B + B_proj_bias.to(dtype=B.dtype)
        if not A.is_complex():
            B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            B = rearrange(B, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
    if C is None:  # variable B
        C = x_dbl[:, -d_state:]  # (bl d)
        if C_proj_bias is not None:
            C = C + C_proj_bias.to(dtype=C.dtype)
        if not A.is_complex():
            C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            C = rearrange(C, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
    y = selective_scan_fn(x, delta, A, B, C, D, z=z, delta_bias=delta_bias, delta_softplus=True)
    return F.linear(rearrange(y, "b d l -> b l d"), out_proj_weight, out_proj_bias)


def bimamba_inner_ref(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, A_b, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True
):
    L = xz.shape[-1]
    delta_rank = delta_proj_weight.shape[1]
    d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
    x, z = xz.chunk(2, dim=1)
    x = causal_conv1d_fn(x, rearrange(conv1d_weight, "d 1 w -> d w"), conv1d_bias, "silu")
    # We're being very careful here about the layout, to avoid extra transposes.
    # We want delta to have d as the slowest moving dimension
    # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
    x_dbl = F.linear(rearrange(x, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
    delta = delta_proj_weight @ x_dbl[:, :delta_rank].t()
    delta = rearrange(delta, "d (b l) -> b d l", l=L)
    if B is None:  # variable B
        B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl d)
        if B_proj_bias is not None:
            B = B + B_proj_bias.to(dtype=B.dtype)
        if not A.is_complex():
            B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            B = rearrange(B, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
    if C is None:  # variable B
        C = x_dbl[:, -d_state:]  # (bl d)
        if C_proj_bias is not None:
            C = C + C_proj_bias.to(dtype=C.dtype)
        if not A.is_complex():
            C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            C = rearrange(C, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
    y = selective_scan_fn(x, delta, A, B, C, D, z=z, delta_bias=delta_bias, delta_softplus=True)
    y_b = selective_scan_fn(x.flip([-1]), delta.flip([-1]), A_b, B.flip([-1]), C.flip([-1]), D, z.flip([-1]), delta_bias, delta_softplus=True)
    y = y + y_b.flip([-1])
    return F.linear(rearrange(y, "b d l -> b l d"), out_proj_weight, out_proj_bias)