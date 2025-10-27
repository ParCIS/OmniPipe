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
    def __init__(self, num_microbatches: int, tensor_shape, tensor_dtype, deallocate_fn: Callable, 
                 virtual_data_parallel_world_size: int = 1):
        self.num_microbatches = num_microbatches
        self.tensor_shape = tensor_shape
        self.tensor_dtype = tensor_dtype
        self.virtual_data_parallel_world_size = virtual_data_parallel_world_size
        self.deallocate_fn = deallocate_fn
        self.num_microbatches_per_vdp_rank = num_microbatches // virtual_data_parallel_world_size

        self.p2p_ops: List[dist.P2POp] = []
        self.comm_info_for_debug = []
        self.to_free: List[torch.Tensor] = []
        self.rank = dist.get_rank()
        self.send_forward_microbatch_id: List[int] = [0 for _ in range(virtual_data_parallel_world_size)]
        self.send_backward_microbatch_id: List[int] = [0 for _ in range(virtual_data_parallel_world_size)]
        self.recv_forward_microbatch_id: List[int] = [0 for _ in range(virtual_data_parallel_world_size)]
        self.recv_backward_microbatch_id: List[int] = [0 for _ in range(virtual_data_parallel_world_size)]
    
    def _generate_unique_tag(self, src_global_rank: int, direction: int, vdp_rank: int, microbatch_id: int):
        """
        Generate a unique p2p tag. 
        Args:
            src_global_rank: int, src rank
            direction: int, 1 if forward else 0
            vdp_rank: int
            microbatch_id: int
        tag: | src_global_rank 15bit | direction 1bit |  vdp_rank 7bit | microbatch_id 9bit|
        """
        microbatch_id = vdp_rank * self.num_microbatches_per_vdp_rank + microbatch_id
        if direction == 0:
            # backward microbatch id starts with self.num_microbatches
            microbatch_id = microbatch_id + self.num_microbatches
        return (src_global_rank << 17) | (direction << 16) | (vdp_rank << 9) | microbatch_id


    def send_forward(self, tensor_to_send: torch.Tensor):
        dst = get_pipeline_model_parallel_next_rank()
        vdp_rank = parallel_state.get_virtual_data_parallel_rank()
        tag = self._generate_unique_tag(src_global_rank=self.rank, direction=1, vdp_rank=vdp_rank, microbatch_id=self.send_forward_microbatch_id[vdp_rank])
        # self.comm_info_for_debug.append(("send_forward", self.rank, dst, vdp_rank * 10 + self.send_forward_microbatch_id[vdp_rank]))
        self.send_forward_microbatch_id[vdp_rank] += 1
        self.p2p_ops.append(dist.P2POp(dist.isend, tensor_to_send, dst, tag=tag))
        self.to_free.append(tensor_to_send)
    
    def recv_forward(self):
        src = get_pipeline_model_parallel_prev_rank()
        vdp_rank = parallel_state.get_virtual_data_parallel_rank()
        tag = self._generate_unique_tag(src_global_rank=src, direction=1, vdp_rank=vdp_rank, microbatch_id=self.recv_forward_microbatch_id[vdp_rank])
        # self.comm_info_for_debug.append(("recv_forward", src, self.rank, vdp_rank * 10 + self.recv_forward_microbatch_id[vdp_rank]))
        self.recv_forward_microbatch_id[vdp_rank] += 1
        buffer = torch.zeros(self.tensor_shape, device=torch.cuda.current_device(), requires_grad=True, dtype=self.tensor_dtype)
        self.p2p_ops.append(dist.P2POp(dist.irecv, buffer, src, tag=tag))
        return buffer

    def send_backward(self, tensor_to_send: torch.Tensor):
        dst = get_pipeline_model_parallel_prev_rank()
        vdp_rank = parallel_state.get_virtual_data_parallel_rank()
        tag = self._generate_unique_tag(src_global_rank=self.rank, direction=0, vdp_rank=vdp_rank, microbatch_id=self.send_backward_microbatch_id[vdp_rank])
        # self.comm_info_for_debug.append(("send_backward", self.rank, dst, vdp_rank * 10 + self.send_backward_microbatch_id[vdp_rank]))
        self.send_backward_microbatch_id[vdp_rank] += 1
        self.p2p_ops.append(dist.P2POp(dist.isend, tensor_to_send, dst, tag=tag))
        tensor_to_send = None
    
    def recv_backward(self):
        src = get_pipeline_model_parallel_next_rank()
        vdp_rank = parallel_state.get_virtual_data_parallel_rank()
        tag = self._generate_unique_tag(src_global_rank=src, direction=0, vdp_rank=vdp_rank, microbatch_id=self.recv_backward_microbatch_id[vdp_rank])
        # self.comm_info_for_debug.append(("recv_backward", src, self.rank, vdp_rank * 10 + self.recv_backward_microbatch_id[vdp_rank]))
        self.recv_backward_microbatch_id[vdp_rank] += 1
        buffer = torch.zeros(self.tensor_shape, device=torch.cuda.current_device(), requires_grad=True, dtype=self.tensor_dtype)
        self.p2p_ops.append(dist.P2POp(dist.irecv, buffer, src, tag=tag))
        return buffer

    def commit(self, wait: bool = True):
        if len(self.p2p_ops) == 0:
            return
        # print(f"[rank {self.rank}]: commit p2p ops: {self.comm_info_for_debug}")
        self.reqs = dist.batch_isend_irecv(self.p2p_ops)
        self.comm_info_for_debug = []
        if wait:
            self.wait()
    
    def wait(self):
        for req in self.reqs:
            req.wait()
        self.p2p_ops: List[dist.P2POp] = []
        for t in self.to_free:
            self.deallocate_fn(t)
        self.to_free: List[torch.Tensor] = []
    
    def reset(self):
        self.p2p_ops: List[dist.P2POp] = []
        self.to_free: List[torch.Tensor] = []
