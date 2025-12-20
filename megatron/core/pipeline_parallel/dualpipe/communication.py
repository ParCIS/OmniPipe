from typing import List, Tuple

import torch
import torch.distributed as dist


TENSOR_SHAPES: List[Tuple[int]] = None
TENSOR_DTYPE: torch.dtype = None


def set_p2p_tensor_shapes(shapes: List[Tuple[int]]):
    global TENSOR_SHAPES
    TENSOR_SHAPES = shapes


def set_p2p_tensor_dtype(dtype: torch.dtype):
    global TENSOR_DTYPE
    TENSOR_DTYPE = dtype


def build_from_tensor_shapes():
    return [torch.empty(s, dtype=TENSOR_DTYPE, device="cuda", requires_grad=True) for s in TENSOR_SHAPES]


def append_irecv(ops: List[dist.P2POp], src: int) -> List[torch.Tensor]:
    rank = dist.get_rank()
    print(f"[rank{rank}]: recv ({src}, {rank})")
    tensors = build_from_tensor_shapes()
    for tensor in tensors:
        if tensor is not None:
            ops.append(dist.P2POp(dist.irecv, tensor, src))
    return tensors



def append_isend(ops: List[dist.P2POp], tensors: List[torch.Tensor], dst: int) -> None:
    rank = dist.get_rank()
    print(f"[rank{rank}]: send ({rank}, {dst})")
    for tensor in tensors:
        if tensor is not None:
            ops.append(dist.P2POp(dist.isend, tensor, dst))
