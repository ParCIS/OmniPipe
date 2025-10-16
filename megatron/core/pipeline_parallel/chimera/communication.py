from typing import Callable, List
import torch.distributed as dist
import torch
from megatron.core import parallel_state

def get_pipeline_model_parallel_prev_rank():
    vdp = parallel_state.get_virtual_data_parallel_world_size()
    vdp_rank = parallel_state.get_virtual_data_parallel_rank()
    if vdp is None or vdp_rank < vdp // 2:
        return parallel_state.get_pipeline_model_parallel_prev_rank()
    else:
        return parallel_state.get_pipeline_model_parallel_next_rank()


def get_pipeline_model_parallel_next_rank():
    vdp = parallel_state.get_virtual_data_parallel_world_size()
    vdp_rank = parallel_state.get_virtual_data_parallel_rank()
    if vdp is None or vdp_rank < vdp // 2:
        return parallel_state.get_pipeline_model_parallel_next_rank()
    else:
        return parallel_state.get_pipeline_model_parallel_prev_rank()


class CommunicationManager:
    def __init__(self, num_iteration: int, tensor_shape, tensor_dtype, deallocate_fn: Callable, 
                 virtual_data_parallel_world_size: int = 1):
        self.num_iteration = num_iteration
        self.tensor_shape = tensor_shape
        self.tensor_dtype = tensor_dtype
        self.virtual_data_parallel_world_size = virtual_data_parallel_world_size
        self.deallocate_fn = deallocate_fn

        self.p2p_ops: List[dist.P2POp] = []
        self.to_free: List[torch.Tensor] = []
        self.rank = dist.get_rank()

    def send_forward(self, tensor_to_send: torch.Tensor):
        dst = get_pipeline_model_parallel_next_rank()
        self.p2p_ops.append(dist.P2POp(dist.isend, tensor_to_send, dst))
        self.to_free.append(tensor_to_send)
    
    def recv_forward(self):
        src = get_pipeline_model_parallel_prev_rank()
        buffer = torch.zeros(self.tensor_shape, device=torch.cuda.current_device(), requires_grad=True, dtype=self.tensor_dtype)
        self.p2p_ops.append(dist.P2POp(dist.irecv, buffer, src))
        return buffer

    def send_backward(self, tensor_to_send: torch.Tensor):
        dst = get_pipeline_model_parallel_prev_rank()
        self.p2p_ops.append(dist.P2POp(dist.isend, tensor_to_send, dst))
        tensor_to_send = None
    
    def recv_backward(self):
        src = get_pipeline_model_parallel_next_rank()
        buffer = torch.zeros(self.tensor_shape, device=torch.cuda.current_device(), requires_grad=True, dtype=self.tensor_dtype)
        self.p2p_ops.append(dist.P2POp(dist.irecv, buffer, src))
        return buffer

    def commit_and_wait(self):
        if len(self.p2p_ops) == 0:
            return
        reqs = dist.batch_isend_irecv(self.p2p_ops)
        for req in reqs:
            req.wait()
        self.p2p_ops: List[dist.P2POp] = []

        for t in self.to_free:
            self.deallocate_fn(t)
        self.to_free: List[torch.Tensor] = []
    
    def reset(self):
        self.p2p_ops: List[dist.P2POp] = []
        self.to_free: List[torch.Tensor] = []
