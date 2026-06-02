import functools
import logging
import os
import queue
from contextlib import contextmanager

from megatron.training import get_args, get_timers
from megatron.core import parallel_state
from megatron.core.utils import get_model_config, get_attr_wrapped_model


def get_offload_time(hidden_size: int, seq_length: int):
    tflops, d2h_bandwidth_gbps = get_gpu_flops_and_offload_bandwidth()
    k = 10.0 / (3 * (6 * hidden_size + seq_length)) * tflops * 1000.0 / d2h_bandwidth_gbps
    return k / 2.0


def get_gpu_flops_and_offload_bandwidth():
    """Get GPU model name and other properties."""
    import torch
    assert torch.cuda.is_available(), "cuda not available"

    gpu_specs = {
        "A100": {
            "tflops": 220,  # Profiled TFLOPS during training
            "d2h_bandwidth_gbps": 15,  # Profiled H2D bandwidth
        },
        # Now we don't support 2-streams per GPU for H100 for bidirectional transfer.
        # Or it could be 2 times faster in H100.
        "H100": {
            "tflops": 220 * 2,
            "d2h_bandwidth_gbps": 21,
        },
    }
    gpu_name = torch.cuda.get_device_name(0)
    for name, specs in gpu_specs.items():
        if name in gpu_name:
            return specs["tflops"], specs["d2h_bandwidth_gbps"]

    default_spec = gpu_specs["A100"]
    return default_spec["tflops"], default_spec["d2h_bandwidth_gbps"]


class WeightGradStore:

    should_split_bw = True
    cache = []
    weight_grad_queue = None  # lazy init
    use_te: bool = True

    @classmethod
    def lazy_init(cls):
        if cls.weight_grad_queue is not None:
            return
        # Lazy init to make sure parallel_state and get_args() have been initialized.
        num_chunks = parallel_state.get_virtual_pipeline_model_parallel_world_size() or 1
        # chunk id => seq id => Queue
        cls.weight_grad_queue = [[queue.Queue() for _ in range(1)] for _ in range(num_chunks)]

    @classmethod
    def is_supported(cls):
        """If not supported, fallback to original schedule."""
        args = get_args()
        if args.pipeline_model_parallel_size <= 1:
            return False
        # if args.virtual_pipeline_model_parallel_size is not None:
        #     return False
        if args.overlap_grad_reduce:
            # the logic of overlapping grad reduce should be changed
            return False
        # if not args.gradient_accumulation_fusion:
        #     return False
        # if args.transformer_impl == 'transformer_engine':
        #     # hard to capture weight gradient computation for transformer_engine
        #     return False
        return True

    @classmethod
    def split_bw(cls):
        if not cls.is_supported():
            return False
        return cls.should_split_bw

    @classmethod
    def enable_split_bw(cls):
        cls.should_split_bw = True

    @classmethod
    def disable_split_bw(cls):
        cls.should_split_bw = False

    @classmethod
    @contextmanager
    def set_split_bw(cls, enabled: bool):
        prev = cls.should_split_bw
        cls.should_split_bw = enabled
        try:
            yield
        finally:
            cls.should_split_bw = prev

    @classmethod
    def put(cls, weight=None,  pre_func=None, func=None, model_chunk=None):
        assert cls.split_bw()
        # func(*pre_func(async_op=False))
        if not cls.use_te:
            cls.cache.append((weight, pre_func, func))
        else:
            def no_op(*args, **kwargs):
                pass
            def te_weight_grad_compute(model_chunk):
                decoder = get_attr_wrapped_model(model_chunk, "decoder")
                transformer_layers = decoder.layers
                for layer in transformer_layers:
                    layer.mlp.backward_dw()
                    layer.self_attention.backward_dw()
            te_func = functools.partial(te_weight_grad_compute, model_chunk=model_chunk)
            cls.cache.append((None, no_op, te_func))
        return
    

    @classmethod
    def queue_size(cls, chunk=0, seq_split_idx=0):
        cls.lazy_init()
        return WeightGradStore.weight_grad_queue[chunk][seq_split_idx].qsize()

    @classmethod
    def flush(cls, chunk=0, seq_split_idx=0):
        cls.lazy_init()
        # Or W later will consume empty computation and leak the non-empty computation.
        if not cls.split_bw():
            assert len(cls.cache) == 0
            return
        cls.weight_grad_queue[chunk][seq_split_idx].put(cls.cache)
        cls.cache = []

    @classmethod
    def pop(cls, chunk=0, seq_split_idx=0):
        cls.lazy_init()
        if cls.weight_grad_queue[chunk][seq_split_idx].qsize() > 0:
            stored_grads = cls.weight_grad_queue[chunk][seq_split_idx].get()
            for weight, pre_func, func in stored_grads:
                if cls.use_te:
                    func()
                else:
                    func(*pre_func(async_op=False))
        else:
            rank = parallel_state.get_pipeline_model_parallel_rank()
            raise Exception(f"Pop empty queue. rank {rank}")

    @classmethod
    def assert_empty(cls):
        rank = parallel_state.get_pipeline_model_parallel_rank()
        assert len(cls.cache) == 0, f"cache is not empty. rank {rank}"
        if cls.weight_grad_queue is None:
            return
        for chunk, chunk_q in enumerate(cls.weight_grad_queue):
            for seq, seq_q in enumerate(chunk_q):
                assert seq_q.empty(), f"Queue is not empty chunk {chunk} seq {seq} rank {rank}. len {seq_q.qsize()}"

    @classmethod
    def clear(cls, model, chunk=0, seq_split_idx=0):
        cls.lazy_init()
        weight_grad_tasks = []
        while cls.weight_grad_queue[chunk][seq_split_idx].qsize() > 0:
            stored_grads = cls.weight_grad_queue[chunk][seq_split_idx].get()
            if len(weight_grad_tasks) == 0:
                for _ in stored_grads:
                    weight_grad_tasks.append([])
            else:
                assert len(weight_grad_tasks) == len(stored_grads)
            for i, task in enumerate(stored_grads):
                weight_grad_tasks[i].append(task)
        # timers = get_timers()
        # weight_params = []
        # handles = []
        # if get_args().overlap_grad_reduce:
        #     handles += model.async_reduce_grad()

        # config = get_model_config(model)
        # # Do async all-reduce for embedding grads firstly, so that the rank 0 won't
        # # be blocked
        # embedding_handles = _allreduce_embedding_grads([model], config, async_op=True)
        # handles += embedding_handles

        for i in range(len(weight_grad_tasks)):
            tasks = weight_grad_tasks[i]
            param = None
            for j in range(len(tasks)):
                weight, pre_func, func = tasks[j]
                if param is None:
                    param = weight
                # assert param.storage().data_ptr() == weight.storage().data_ptr()
                if cls.use_te:
                    func()
                else:
                    func(*pre_func(async_op=False))
                tasks[j] = None  # release memory
            # weight_params.append(param)
            # if get_args().overlap_grad_reduce:
            #     # All-reduce param grad here
            #     handles += model.async_reduce_grad(param)
            weight_grad_tasks[i] = None  # release memory

        # timers('wait_all_reduce', log_level=1).start(barrier=False)
        # for handle in handles:
        #     if handle is not None:
        #         handle.wait()
        # timers('wait_all_reduce').stop()

class RecomputeStore:

    cache = []
    recompute_queue = None
    recompute_flag = False

    @classmethod
    def lazy_init(cls):
        if cls.recompute_queue is not None:
            return
        # Lazy init to make sure parallel_state and get_args() have been initialized.
        num_chunks = parallel_state.get_virtual_pipeline_model_parallel_world_size() or 1
        # chunk id => Queue
        cls.recompute_queue = [queue.Queue() for _ in range(num_chunks)]

    @classmethod
    @contextmanager
    def set_recompute_flag(cls, enabled: bool):
        prev = cls.recompute_flag
        cls.recompute_flag = enabled
        try:
            yield
        finally:
            cls.recompute_flag = prev

    @classmethod
    def should_recompute(cls):
        return cls.recompute_flag

    @classmethod
    def put(cls, func):
        assert cls.recompute_flag
        cls.lazy_init()
        cls.cache.append(func)

    @classmethod
    def flush(cls):
        if not cls.recompute_flag:
            assert len(cls.cache) == 0
            return
        cls.lazy_init()
        chunk = parallel_state.get_virtual_pipeline_model_parallel_rank()
        cls.recompute_queue[chunk].put(cls.cache)
        cls.cache = []

    @classmethod
    def pop(cls):
        cls.lazy_init()
        chunk = parallel_state.get_virtual_pipeline_model_parallel_rank()
        recompute_funcs = cls.recompute_queue[chunk].get()
        for func in recompute_funcs:
            func()
