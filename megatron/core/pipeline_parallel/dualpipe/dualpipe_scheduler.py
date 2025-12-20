import contextlib
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union
from functools import partial
import torch
import torch.distributed as dist
import logging
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.utils import set_streams
from megatron.core.utils import get_attr_wrapped_model, get_model_config, get_model_type
from megatron.core.tensor_parallel.weight_grad_store import WeightGradStore
from .communication import set_p2p_tensor_dtype, set_p2p_tensor_shapes, append_isend, append_irecv
from ..chimera.communication import get_pipeline_model_parallel_next_rank, get_pipeline_model_parallel_prev_rank, CommunicationManager
logger = logging.getLogger(__name__)

class DualPipeScheduler():
    def __init__(self):
        self.first_run = True
    def prepare(
        self,
        *,
        forward_step_func,
        data_iterator: Union[Iterator, List[Iterator]],
        model: Union[torch.nn.Module, List[torch.nn.Module]],
        num_microbatches: int,
        seq_length: int,
        micro_batch_size: int,
        decoder_seq_length: Optional[int] = None,
        forward_only: bool = False,
        collect_non_loss_data: bool = False,
        first_val_step: Optional[bool] = None,
        adjust_tensor_shapes_fn: Optional[Callable] = None,  # unused
    ):
        """Run chimera schedule (multiple pipeline per GPU).
        Returns dictionary with losses if the last stage, empty dict otherwise.
        """
        # import here to avoid circle import
        from megatron.core.pipeline_parallel.schedules import clear_embedding_activation_buffer, deallocate_output_tensor
        self.model = model
        self.config = get_model_config(model[0])
        self.ddp_config = model[0].ddp_config
        self.forward_step_func = forward_step_func
        self.data_iterator = data_iterator
        self.num_microbatches = num_microbatches
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.decoder_seq_length = decoder_seq_length
        self.forward_only = forward_only
        self.collect_non_loss_data = collect_non_loss_data
        self.first_val_step = first_val_step
        self.adjust_tensor_shapes_fn = adjust_tensor_shapes_fn

        if self.config.overlap_p2p_comm and self.config.batch_p2p_comm:
            raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")
        self.num_model_chunks = len(self.model)
        self.tensor_shape = [seq_length, micro_batch_size, self.config.hidden_size]
        self.tensor_shape[0] //= parallel_state.get_context_parallel_world_size()
        if self.config.sequence_parallel:
            self.tensor_shape[0] //= parallel_state.get_tensor_model_parallel_world_size()
        set_p2p_tensor_shapes(self.tensor_shape)
        set_p2p_tensor_dtype(self.config.pipeline_dtype)

        # Needed only when gradients are finalized in M-Core
        if self.config.finalize_model_grads_func is not None and not forward_only:
            # no use yet, set is_last_stage to False so that make it no sense
            embedding_module = clear_embedding_activation_buffer(self.config, model, is_last_stage=False)

        if self.config.timers is not None:
            self.config.timers('forward-backward', log_level=1).start(barrier=self.config.barrier_with_L1_time)

        # Disable async grad reductions
        self.no_sync_func = self.config.no_sync_func
        if isinstance(self.no_sync_func, list):

            def multi_no_sync():
                stack = contextlib.ExitStack()
                for model_chunk_no_sync_func in self.config.no_sync_func:
                    stack.enter_context(model_chunk_no_sync_func())
                return stack

            self.no_sync_func = multi_no_sync
        if self.no_sync_func is None:
            self.no_sync_func = contextlib.nullcontext
        self.no_sync_context = None

        if self.config.grad_sync_func is not None and not isinstance(self.config.grad_sync_func, list):
            self.config.grad_sync_func = [self.config.grad_sync_func for _ in model]
        # Disable config.grad_sync_func and config.param_sync_func if only running forward passes.
        # They will be re-enabled at the end of this function.
        self.grad_sync_func, self.param_sync_func = None, None
        if forward_only:
            self.grad_sync_func, self.param_sync_func = self.config.grad_sync_func, self.config.param_sync_func
            self.config.grad_sync_func, self.config.param_sync_func = None, None
        self.model_type = get_model_type(model[0])

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.virtual_data_parallel_world_size = 2
        self.reset()
        self.param_sync_model_chunk_id_queue: List[int] = None
        self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        self.num_pp_ranks = parallel_state.get_pipeline_model_parallel_world_size()
        self.is_in_second_half = self.pp_rank >= self.num_pp_ranks // 2
        self.is_middle_rank = (self.pp_rank == self.num_pp_ranks // 2 - 1) or (self.pp_rank == self.num_pp_ranks // 2)
        self.comm_mgr = CommunicationManager(
            num_microbatches=self.num_microbatches,
            tensor_shape=self.tensor_shape,
            tensor_dtype=self.config.pipeline_dtype,
            deallocate_fn=partial(deallocate_output_tensor, deallocate_pipeline_outputs=self.config.deallocate_pipeline_outputs),
            virtual_data_parallel_world_size=self.virtual_data_parallel_world_size
        )
        assert num_microbatches >= parallel_state.get_pipeline_model_parallel_world_size() * 2 - 2, f"DualPipe only support num_microbatches >= 2 * PP - 2, num_microbatches = {num_microbatches}"
        if self.rank == 0:
            logger.info(f">>>DualPipeScheduler finishes initialization.")

    def reset(self) -> None:
        # Model chunk IDs with synchronized grads
        self.synchronized_model_chunks = set()
        self.sync_model_queue = []
        self.param_sync_idx = 0
        self.last_backward_vdp_rank = None

        self.total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()
        self.forward_data_store = []
        if not self.forward_only:
            self.output_tensor_grads = [[] for _ in range(len(self.model))]
            self.input_tensor_grads = [[] for _ in range(len(self.model))]
        else:
            self.input_tensor_grads = None
            self.output_tensor_grads = None
        self.input_tensors = [[] for _ in range(len(self.model))]
        self.output_tensors = [[] for _ in range(len(self.model))]
        self.input_tensor_grads = [[] for _ in range(len(self.model))]

        self.current_f_microbatch_id = [0 for _ in range(len(self.model))]
        self.current_b_microbatch_id = [0 for _ in range(len(self.model))]
        self.current_send_f_microbatch_id = [0 for _ in range(len(self.model))]
        self.current_send_b_microbatch_id = [0 for _ in range(len(self.model))]
        self.current_recv_f_microbatch_id = [0 for _ in range(len(self.model))]
        self.current_recv_b_microbatch_id = [0 for _ in range(len(self.model))]
        self.current_w_microbatch_id = [0 for _ in range(len(self.model))]
        self.comm_ops = []
        self.weight_compute_queue_for_te = []
        WeightGradStore.reset()
    
    def disable_grad_sync(self):
        """Disable asynchronous grad reductions"""
        if self.no_sync_context is None:
            self.no_sync_context = self.no_sync_func()
            self.no_sync_context.__enter__()

    def enable_grad_sync(self):
        """Enable asynchronous grad reductions"""
        if self.no_sync_context is not None:
            self.no_sync_context.__exit__(None, None, None)
            self.no_sync_context = None
    
    def _forward_step_helper_preprocess(self, vdp_rank):
        """Preprocess for forward_step_helper"""
        if self.config.param_sync_func is not None and self.param_sync_idx < len(self.model):
            param_sync_model_chunk_id = self.param_sync_model_chunk_id_queue[self.param_sync_idx]
            self.config.param_sync_func[param_sync_model_chunk_id](self.model[param_sync_model_chunk_id].parameters())
            self.param_sync_idx += 1

        # forward step
        microbatch_id = self.current_f_microbatch_id[vdp_rank]
        parallel_state.set_virtual_data_parallel_rank(vdp_rank)
        if len(self.input_tensors[vdp_rank]) == self.current_f_microbatch_id[vdp_rank]:
            self.input_tensors[vdp_rank].append(None)
        input_tensor = self.input_tensors[vdp_rank][microbatch_id]
        return input_tensor

    def _forward_step_helper_postprocess(self, vdp_rank, output_tensor, num_tokens):
        """Postprocess for forward_step_helper"""
        self.output_tensors[vdp_rank].append(output_tensor)
        self.total_num_tokens += num_tokens

        # If forward-only, no need to save tensors for a backward pass.
        microbatch_id = self.current_f_microbatch_id[vdp_rank]
        if self.forward_only:
            # Release the tensor that have completed forward step.
            self.input_tensors[vdp_rank][microbatch_id] = None
            self.output_tensors[vdp_rank][microbatch_id] = None
        self.current_f_microbatch_id[vdp_rank] += 1
        return
    
    def _backward_step_helper_preprocess(self, vdp_rank):
        """
        Preprocess for backward_step_helper.
        Because some user may don't use the transformer-engine, enable async grad sync will cause a mistake,
        we prefer to manually call the grad_sync_func, like the config.overlap_grad_reduce with config.align_grad_reduce.
        """

        parallel_state.set_virtual_data_parallel_rank(vdp_rank)
        microbatch_id = self.current_b_microbatch_id[vdp_rank]
        if parallel_state.is_pipeline_last_stage(ignore_virtual=False):
            if len(self.output_tensor_grads[vdp_rank]) == microbatch_id:
                self.output_tensor_grads[vdp_rank].append(None)
        input_tensor = self.input_tensors[vdp_rank][microbatch_id]
        output_tensor = self.output_tensors[vdp_rank][microbatch_id]
        output_tensor_grad = self.output_tensor_grads[vdp_rank][microbatch_id]

        self.input_tensors[vdp_rank][microbatch_id] = None
        self.output_tensors[vdp_rank][microbatch_id] = None
        self.output_tensor_grads[vdp_rank][microbatch_id] = None

        return input_tensor, output_tensor, output_tensor_grad

    def _backward_step_helper_postprocess(self, vdp_rank, intput_tensor_grad):
        """Postprocess for backward_step_helper"""
        self.current_b_microbatch_id[vdp_rank] += 1
        self.input_tensor_grads[vdp_rank].append(intput_tensor_grad)
    
    def _forward_compute_chunk(self, phase: int) -> None:
        parallel_state.set_virtual_data_parallel_rank(phase)
        from megatron.core.pipeline_parallel.schedules import forward_step
        input_tensor = self._forward_step_helper_preprocess(phase)
        microbatch_id = self.current_f_microbatch_id[phase]

        # doesn't support checkpoint_activations_microbatch yet
        output_tensor, num_tokens = forward_step(
            self.forward_step_func, 
            self.data_iterator[phase],
            self.model[phase],
            self.num_microbatches,
            input_tensor,
            self.forward_data_store,
            self.config,
            self.collect_non_loss_data,
            checkpoint_activations_microbatch=None,
            is_first_microbatch=(microbatch_id == 0),
            current_microbatch=microbatch_id,
            vp_stage=phase,
            is_last_stage=parallel_state.is_pipeline_last_stage(ignore_virtual=False)
        )
        self._forward_step_helper_postprocess(phase, output_tensor, num_tokens)

    def _backward_compute_chunk(self, phase: int, enable_zb: bool = False) -> None:
        if self.forward_only:
            return
        parallel_state.set_virtual_data_parallel_rank(phase)
        from megatron.core.pipeline_parallel.schedules import backward_step
        if self.forward_only:
            return
        input_tensor, output_tensor, output_tensor_grad = self._backward_step_helper_preprocess(phase)
        self.set_decouple_dw(phase, enable_zb)
        input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad, self.model_type, self.config)
        self._backward_step_helper_postprocess(phase, input_tensor_grad)
        self.set_decouple_dw(phase, False)
        if not enable_zb:
            self.current_w_microbatch_id[phase] += 1
            self._maybe_start_sync(phase)
        return input_tensor_grad

    def _forward_backward_compute_chunk(self, phase0: int, phase1: int, enable_zb: bool = False) -> None:
        # import here to avoid circle import
        from megatron.core.pipeline_parallel.schedules import check_first_val_step
        from megatron.core.pipeline_parallel.utils import set_streams
        from megatron.core.pipeline_parallel.combined_1f1b import combined_forward_backward_step

        set_streams()
        # forward prepare
        input_tensor = None
        f_microbatch_id = None
        if phase0 is not None:
            phase0 ^= self.is_in_second_half
            parallel_state.set_virtual_data_parallel_rank(phase0)
            input_tensor = self._forward_step_helper_preprocess(phase0)
            f_microbatch_id = self.current_f_microbatch_id[phase0]
        # backward prepare
        b_input_tensor = None
        b_output_tensor = None
        b_output_tensor_grad = None
        if phase1 is not None:
            phase1 ^= self.is_in_second_half
            parallel_state.set_virtual_data_parallel_rank(phase1)
            self.set_decouple_dw(phase1, enable_zb)
            b_input_tensor, b_output_tensor, b_output_tensor_grad = self._backward_step_helper_preprocess(phase1)
        # Call combined forward and backward step to overlap the communication and computation
        output_tensor, num_tokens, input_tensor_grad = combined_forward_backward_step(
            self.forward_step_func, 
            self.data_iterator[phase0] if phase0 is not None else None,
            self.model[phase0] if phase0 is not None else None,
            self.num_microbatches,
            input_tensor,
            self.forward_data_store,
            self.model[phase1] if phase1 is not None else None,
            b_input_tensor,
            b_output_tensor,
            b_output_tensor_grad,
            self.config,
            f_model_chunk_id=phase0,
            b_model_chunk_id=phase1,
            collect_non_loss_data=self.collect_non_loss_data,
            checkpoint_activations_microbatch=None,
            is_first_microbatch=check_first_val_step(
                first_val_step=self.first_val_step, 
                forward_only=self.forward_only,
                cond=(self.current_f_microbatch_id[phase0] == 0) if phase0 is not None else None
            ),
            current_microbatch=f_microbatch_id,
        )
        # forward post process
        if phase0 is not None:
            self._forward_step_helper_postprocess(phase0, output_tensor, num_tokens)
        # backward post process
        if phase1 is not None:
            # The same as the backward_step_helper
            self._backward_step_helper_postprocess(phase1, input_tensor_grad)
            if not enable_zb:
                self.current_w_microbatch_id[phase1] += 1
                self._maybe_start_sync(phase1)
        return output_tensor, num_tokens, input_tensor_grad

    def _forward_chunk(self, phase: int, recv: bool = True, send: bool = True) -> None:
        if recv:
            self._recv_forward(phase)
        self.comm_mgr.commit()
        if self.config.overlap_moe_expert_parallel_comm:
            self._forward_backward_compute_chunk(phase0=phase, phase1=None)
        else:
            self._forward_compute_chunk(phase)

        if send:
            self._send_forward(phase)

    def _backward_chunk(self, phase: int, enable_zb: bool = False, recv: bool = True, send: bool = True) -> None:
        if recv:
            self._recv_backward(phase)
        self.comm_mgr.commit()
        if self.config.overlap_moe_expert_parallel_comm:
            self._forward_backward_compute_chunk(phase0=None, phase1=phase, enable_zb=enable_zb)
        else:
            self._backward_compute_chunk(phase, enable_zb)

        if send:
            self._send_backward(phase)

    def _forward_backward_chunk(self, phase0: int, phase1: int, recv0: bool = True) -> None:
        if recv0:
            self._recv_forward(phase0)
        self._recv_backward(phase1)
        self.comm_mgr.commit()
        if self.config.overlap_moe_expert_parallel_comm and not self.forward_only:
            self._forward_backward_compute_chunk(phase0, phase1)
        else:
            self._forward_compute_chunk(phase0)
            self._backward_compute_chunk(phase1)

        self._send_forward(phase0)
        self._send_backward(phase1)
    
    def set_decouple_dw(self, phase: int, enable: bool = True):
        def te_set(enable):
            model_wrapped = self.model[phase]
            decoder = get_attr_wrapped_model(model_wrapped, "decoder")
            transformer_layers = decoder.layers
            for layer in transformer_layers:
                if enable:
                    layer.mlp.experts.linear_fc1.wgrad_store.enable_delay_wgrad_compute()
                    layer.mlp.experts.linear_fc2.wgrad_store.enable_delay_wgrad_compute()
                    layer.self_attention.linear_qkv.wgrad_store.enable_delay_wgrad_compute()
                    layer.self_attention.linear_proj.wgrad_store.enable_delay_wgrad_compute()
                else:
                    layer.mlp.experts.linear_fc1.wgrad_store.disable_delay_wgrad_compute()
                    layer.mlp.experts.linear_fc2.wgrad_store.disable_delay_wgrad_compute()
                    layer.self_attention.linear_qkv.wgrad_store.disable_delay_wgrad_compute()
                    layer.self_attention.linear_proj.wgrad_store.disable_delay_wgrad_compute()
        def local_set(enable):
            WeightGradStore.set_enabled(enable)
        
        if self.config.transformer_impl == "transformer_engine":
            te_set(enable)
        else:
            local_set(enable)

    def _weight_chunk(self, phase: int) -> None:
        def te_backward_dw(phase: int):
            model_wrapped = self.model[phase]
            decoder = get_attr_wrapped_model(model_wrapped, "decoder")
            transformer_layers = decoder.layers
            for layer in transformer_layers:
                layer.mlp.backward_dw()
                layer.self_attention.backward_dw()

        def local_backward_dw():
            WeightGradStore.pop()
        
        phase ^= self.is_in_second_half
        self.current_w_microbatch_id[phase] += 1
        logger.debug(f"[rank{self.rank}]: chunk {phase} weight chunk")
        if self.forward_only:
            return

        self.comm_mgr.commit()

        # Assume FIFO
        if self.config.transformer_impl == 'transformer_engine':
            te_backward_dw(phase)
        else:
            local_backward_dw()
        self._maybe_start_sync(vdp_rank=phase)


    def _recv_forward(self, phase: int) -> None:
        phase ^= self.is_in_second_half
        parallel_state.set_virtual_data_parallel_rank(phase)
        is_first_stage = parallel_state.is_pipeline_first_stage()
        if is_first_stage:
            return

        self.current_recv_f_microbatch_id[phase] += 1
        tensor_recv = self.comm_mgr.recv_forward()
        self.input_tensors[phase].append(tensor_recv)

    def _send_forward(self, phase: int) -> None:
        phase ^= self.is_in_second_half
        parallel_state.set_virtual_data_parallel_rank(phase)
        is_last_stage = parallel_state.is_pipeline_last_stage()
        if is_last_stage:
            return

        microbatch_id = self.current_send_f_microbatch_id[phase]
        self.current_send_f_microbatch_id[phase] += 1
        tensor_to_send = self.output_tensors[phase][microbatch_id]
        self.comm_mgr.send_forward(tensor_to_send)

    def _recv_backward(self, phase: int) -> None:
        if self.forward_only:
            return

        phase ^= self.is_in_second_half
        parallel_state.set_virtual_data_parallel_rank(phase)
        is_last_stage = parallel_state.is_pipeline_last_stage()
        if is_last_stage:
            return

        self.current_recv_b_microbatch_id[phase] += 1
        tensor_recv = self.comm_mgr.recv_backward()
        self.output_tensor_grads[phase].append(tensor_recv)

    def _send_backward(self, phase: int) -> None:
        if self.forward_only:
            return

        phase ^= self.is_in_second_half
        parallel_state.set_virtual_data_parallel_rank(phase)
        is_first_stage = parallel_state.is_pipeline_first_stage()
        if is_first_stage:
            return

        microbatch_id = self.current_send_b_microbatch_id[phase]
        self.current_send_b_microbatch_id[phase] += 1
        tensor_to_send = self.input_tensor_grads[phase][microbatch_id]
        self.input_tensor_grads[phase][microbatch_id] = None

        self.comm_mgr.send_backward(tensor_to_send)

    def _is_last_microbatch(self, vdp_rank):
        microbatch_id = self.current_w_microbatch_id[vdp_rank]
        return microbatch_id == self.num_microbatches // self.virtual_data_parallel_world_size
    
    def _maybe_start_sync(self, vdp_rank):
        if self._is_last_microbatch(vdp_rank):
            self.model[vdp_rank].start_grad_sync()
            self.synchronized_model_chunks.add(vdp_rank)
            self.sync_model_queue.append(vdp_rank)
    
    def _finish_sync(self):
        if len(self.synchronized_model_chunks) != len(self.model):
            raise RuntimeError(f"Number of synchronized model is not equal to the models, synchronized_model_chunks is {self.synchronized_model_chunks}")
        if len(self.synchronized_model_chunks) != len(self.sync_model_queue):
            raise RuntimeError(f"Invalid sync model sequence: {self.sync_model_queue}")
        for i in self.sync_model_queue:
            self.config.finalize_model_grads_func(
                [self.model[i]], self.total_num_tokens if self.config.calculate_per_token_loss else None
            )
    
    def step(
        self,
    ) -> Dict[str, int]:
        """
        Execute a training or inference step.
        """
        self.disable_grad_sync()
        self.forward_only = not torch.is_grad_enabled()
        pp_rank = self.pp_rank
        num_ranks = self.num_pp_ranks
        assert num_ranks % 2 == 0
        assert self.num_microbatches > 0 and self.num_microbatches % 2 == 0 and self.num_microbatches >= num_ranks * 2, f"{self.num_microbatches=}, {num_ranks=}"
        num_half_ranks = num_ranks // 2
        half_rank = min(pp_rank, num_ranks - 1 - pp_rank)
        half_num_microbatches = self.num_microbatches // 2
        self.num_half_ranks = num_half_ranks
        self.half_rank = half_rank
        # For the first half of the ranks: phase 0 means forward direction, phase 1 means reverse direction.
        # For the second half of the ranks: phase 0 means reverse direction, phase 1 means forward direction.

        # Step 1: nF0
        # num_half_ranks 是进程总数的一半，half_rank 是当前进程在其所在半部分的排名。
        logger.debug(f"[rank{self.rank}]: Step 1")
        step_1 = (num_half_ranks - half_rank - 1) * 2
        for i in range(step_1):
            self._forward_chunk(0)

        # Step 2: nF0F1
        # a) 在循环中，首先执行前向传播块 0，recv=False 表示不接收数据，
        # send=self.is_middle_rank 表示是否发送数据取决于当前进程是否是中间进程。
        # b) 再次接收前向传播数据。
        # c) 执行前向传播块 1，并根据条件决定是否发送数据。
        # 如果不是中间进程，发送前向传播块 0 的数据。
        step_2 = half_rank + 1
        logger.debug(f"[rank{self.rank}]: Step 2")
        self._recv_forward(0)
        for i in range(step_2):
            self._forward_chunk(0, recv=False, send=self.is_middle_rank)
            self._recv_forward(0)
            self._forward_chunk(1, send=(not self.is_middle_rank) or (i < step_2 - 1))
            if not self.is_middle_rank:
                self._send_forward(0)

        # Step 3: nB1W1F1 (Use zero bubble)
        # a) 循环中，首先执行反向传播块 1，enable_zb=True 表示启用零气泡优化。
                # b) 接收前向传播块 1 的数据。
                # c) 执行权重更新步骤。
                # d) 执行前向传播块 1，不接收数据。
        logger.debug(f"[rank{self.rank}]: Step 3")
        step_3 = num_half_ranks - half_rank - 1
        for i in range(step_3):
            self._backward_chunk(1, enable_zb=True)
            self._recv_forward(1)
            self._weight_chunk(1)
            self._forward_chunk(1, recv=False)

        # Step 4 (Main step): nF0B1F1B0
        # 如果是第一次迭代且当前进程是中间进程，不重叠执行前向传播块 0 和反向传播块 1，
        # 以减少气泡大小。
        # 否则，调用 self._forward_backward_chunk 方法同时执行前向和反向传播。
        # 每次迭代还会执行 self._forward_backward_chunk(1, 0)。
        logger.debug(f"[rank{self.rank}]: Step 4")
        step_4 = half_num_microbatches - num_ranks + half_rank + 1
        for i in range(step_4):
            if i == 0:
                if self.is_middle_rank:
                    # NOTE: We don't overlap these two chunks to further reduce bubble size.
                    self._forward_chunk(0, recv=False, send=False)
                    self._send_forward(1)
                    self._backward_chunk(1, send=False)
                    self._send_forward(0)
                    self._send_backward(1)
                else:
                    self._forward_backward_chunk(0, 1, recv0=False)
            else:
                self._forward_backward_chunk(0, 1)
            self._forward_backward_chunk(1, 0)

        # Step 5: nB1F1B0
        logger.debug(f"[rank{self.rank}]: Step 5")
        step_5 = num_half_ranks - half_rank - 1
        for i in range(step_5):
            self._backward_chunk(1)
            self._forward_backward_chunk(1, 0)

        # Step 6: nB1B0 (The second half of the chunks use zero bubble)
        logger.debug(f"[rank{self.rank}]: Step 6")
        step_6 = half_rank + 1
        enable_zb = False
        for i in range(step_6):
            if i == step_6 // 2 and half_rank % 2 == 1:
                enable_zb = True
            self._backward_chunk(1, enable_zb=enable_zb)
            if enable_zb:
                self.weight_compute_queue_for_te.append(1)
            if i == step_6 // 2 and half_rank % 2 == 0:
                enable_zb = True
            self._backward_chunk(0, enable_zb=enable_zb)
            if enable_zb:
                self.weight_compute_queue_for_te.append(0)

        # Step 7: nWB0 (Use zero bubble)
        logger.debug(f"[rank{self.rank}]: Step 7")
        step_7 = num_half_ranks - half_rank - 1
        for i in range(step_7):
            weight_compute_chunk = self.weight_compute_queue_for_te.pop(0)
            self._weight_chunk(weight_compute_chunk)
            self._backward_chunk(0, enable_zb=True)
            self.weight_compute_queue_for_te.append(0)

        # Step 8: nW
        logger.debug(f"[rank{self.rank}]: Step 8")
        step_8 = half_rank + 1
        for i in range(step_8):
            weight_compute_chunk = self.weight_compute_queue_for_te.pop(0)
            self._weight_chunk(weight_compute_chunk)

        self.comm_mgr.commit()
        self._finish_sync()
        logger.debug(f"[rank{self.rank}]: Pipeline finish")
        if self.config.transformer_impl == "transformer_engine" and self.config.delay_wgrad_compute:
            assert len(self.weight_compute_queue_for_te) == 0, f"Weight grad compute doesnot finish, {self.weight_compute_queue_for_te} remains."
        last_stage_vdp_rank = 0
        for i in range(self.virtual_data_parallel_world_size):
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True, vp_stage=i):
                last_stage_vdp_rank = i
                break
        parallel_state.set_virtual_data_parallel_rank(last_stage_vdp_rank)
        self.enable_grad_sync()
        if self.forward_only:
            self.config.grad_sync_func, self.config.param_sync_func = self.grad_sync_func, self.param_sync_func
        if self.first_run:
            self.param_sync_model_chunk_id_queue = list(reversed(self.sync_model_queue))
            self.first_run = False
        self.enable_grad_sync()
        return self.forward_data_store
    
    def __call__(self, *args, **kwds):
        if self.first_run:
            self.prepare(*args, **kwds)
        else:
            self.reset()
        return self.step()
