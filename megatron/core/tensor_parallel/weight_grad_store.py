import queue
from typing import List, Callable, Tuple
import torch
from megatron.core import parallel_state
from megatron.core.utils import prepare_input_tensors_for_wgrad_compute
import torch.distributed as dist
import nvtx
_grad_accum_fusion_available = True
try:
    import fused_weight_gradient_mlp_cuda
except ImportError:
    _grad_accum_fusion_available = False

def gather(input_slice: torch.Tensor, stream: torch.cuda.Stream):
    tp = parallel_state.get_tensor_model_parallel_world_size()
    dim_size = list(input_slice.size())
    dim_size[0] = dim_size[0] * tp
    all_gather_buffer = parallel_state.get_global_memory_buffer().get_tensor(dim_size, input_slice.dtype, "mpu")
    forward_event = torch.cuda.Event()
    forward_event.record(stream)
    with torch.no_grad():
        with torch.cuda.stream(stream):
            stream.wait_event(forward_event)
            handle = dist.all_gather_into_tensor(
                all_gather_buffer,
                input_slice,
                group=parallel_state.get_tensor_model_parallel_group(),
                async_op=True
            )
    return all_gather_buffer, handle


class WeightGradStore:

    enabled_decouple_dw: bool = False
    cache: List[Callable] = []
    gather_stream: torch.cuda.Stream = None
    current_compute_store: Tuple = None
    gradient_accumulation_fusion: bool = False

    @classmethod
    def put(cls, total_input, grad_output, weight, sequence_parallel) -> None:
        cls.cache.append((total_input, grad_output, weight, sequence_parallel))

    @classmethod
    def overlap_all_gather(cls):
        # used for input all gather in ColumnParallel.
        if len(cls.cache) > 0:
            (input_slice, grad_output, weight, sequence_parallel) = cls.cache.pop(0)
            if not sequence_parallel:
                return [input_slice, grad_output, weight, sequence_parallel], None
            total_input, handle = gather(input_slice, cls.gather_stream)
            return [total_input, grad_output, weight, sequence_parallel], handle
        else:
            raise Exception("All Gather empty queue.")


    @classmethod
    def overlap_matmul(cls, compute_store):
        total_input, grad_output, weight, sequence_parallel = compute_store
        grad_output, total_input = prepare_input_tensors_for_wgrad_compute(
            grad_output,
            total_input
        )
        if cls.gradient_accumulation_fusion:
            if weight.main_grad.dtype == torch.float32:
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                    total_input, grad_output, weight.main_grad
                )
            elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                    total_input, grad_output, weight.main_grad
                )
            else:
                raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")
        else:
            # grad_weight = grad_output.t().matmul(total_input)
            # weight.main_grad.data.add_(grad_weight)
            pass
        # total_input.untyped_storage().resize_(0)
        # grad_output.untyped_storage().resize_(0)


    @classmethod
    def pop(cls):
        if len(cls.cache) == 0:
            return
        if cls.gather_stream is None:
            cls.gather_stream = torch.cuda.Stream(device=torch.cuda.current_device())
        handle = None
        # (total_input, grad_output, weight, sequence_parallel), handle = cls.overlap_all_gather()
        current_compute_store, handle = cls.overlap_all_gather()
        while len(cls.cache) > 0:
            if handle is not None:
                handle.wait()
            next_compute_store, handle = cls.overlap_all_gather()
            cls.overlap_matmul(current_compute_store)
            current_compute_store = next_compute_store
        if handle is not None:
            handle.wait()
        cls.overlap_matmul(current_compute_store)

        current_compute_store = None

    @classmethod
    def reset(cls) -> None:
        cls.cache = []
    
    @classmethod
    def set_enabled(cls, enable: bool) -> None:
        cls.enabled_decouple_dw = enable
    
