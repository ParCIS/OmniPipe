import contextlib
from functools import partial
from megatron.core import parallel_state
from typing import Callable, Iterator, List, Optional, Tuple, Union
import torch
from megatron.core.utils import (
    get_model_config,
    get_model_type,
    get_attr_wrapped_model
)
from megatron.core.tensor_parallel.weight_grad_store import WeightGradStore
import torch.distributed as dist
from .auto_schedule import ChimeraAutoSchedule, ChimeraNode
from .communication import CommunicationManager

def no_op(*unused_args, **unused_kwargs):
    pass

class ChimeraScheduler():
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
        self.virtual_data_parallel_world_size = parallel_state.get_virtual_data_parallel_world_size()
        self.schedule = ChimeraAutoSchedule(self.rank, self.world_size, parallel_state.get_pipeline_model_parallel_world_size(), parallel_state.get_virtual_data_parallel_world_size(), num_microbatches).get_schedule()
        self.comm_mgr = CommunicationManager(
            num_microbatches=self.num_microbatches,
            tensor_shape=self.tensor_shape,
            tensor_dtype=self.config.pipeline_dtype,
            deallocate_fn=partial(deallocate_output_tensor, deallocate_pipeline_outputs=self.config.deallocate_pipeline_outputs),
            virtual_data_parallel_world_size=self.virtual_data_parallel_world_size
        )
        self.reset()
        self.first_run = False

    def reset(self) -> None:
        # Model chunk IDs with synchronized grads
        self.synchronized_model_chunks = set()
        self.sync_model_queue = []
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

        self.comm_mgr.reset()
        WeightGradStore.reset()
        # Because flash attn only support with transformer engine, chimera scheduler disable decouple dw according to the transformer_impl, 
        # looking forward support with local transformer impl.
        self.enable_decouple_dw = self.config.transformer_impl != 'transformer_engine'
    
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
        if self.config.param_sync_func is not None and vdp_rank > 1 and vdp_rank < self.num_model_chunks:
            self.config.param_sync_func[vdp_rank](self.model[vdp_rank].parameters())

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

    def _backward_step_helper_postprocess(self, vdp_rank):
        """Postprocess for backward_step_helper"""
        self.current_b_microbatch_id[vdp_rank] += 1
        self._maybe_start_sync(vdp_rank=vdp_rank)
    
    def _combined_1f1b_schedule_for_chimera(self, f_vdp_rank: Optional[int] = None, b_vdp_rank: Optional[int] = None, pre_forward: Callable = None, post_forward: Callable = None, pre_backward: Callable = None, post_backward: Callable = None) -> None:
        # import here to avoid circle import
        from megatron.core.pipeline_parallel.schedules import check_first_val_step
        from megatron.core.pipeline_parallel.utils import set_streams
        from megatron.core.pipeline_parallel.combined_1f1b import combined_forward_backward_step


        set_streams()
        # forward prepare
        input_tensor = None
        f_microbatch_id = None
        if f_vdp_rank is not None:
            input_tensor = self._forward_step_helper_preprocess(f_vdp_rank)
            f_microbatch_id = self.current_f_microbatch_id[f_vdp_rank]
        # backward prepare
        b_input_tensor = None
        b_output_tensor = None
        b_output_tensor_grad = None
        if b_vdp_rank is not None:
            b_input_tensor, b_output_tensor, b_output_tensor_grad = self._backward_step_helper_preprocess(b_vdp_rank)
        # Call combined forward and backward step to overlap the communication and computation
        output_tensor, num_tokens, input_tensor_grad = combined_forward_backward_step(
            self.forward_step_func, 
            self.data_iterator[f_vdp_rank] if f_vdp_rank is not None else None,
            self.model[f_vdp_rank] if f_vdp_rank is not None else None,
            self.num_microbatches,
            input_tensor,
            self.forward_data_store,
            self.model[b_vdp_rank] if b_vdp_rank is not None else None,
            b_input_tensor,
            b_output_tensor,
            b_output_tensor_grad,
            self.config,
            f_model_chunk_id=f_vdp_rank,
            collect_non_loss_data=self.collect_non_loss_data,
            checkpoint_activations_microbatch=None,
            is_first_microbatch=check_first_val_step(
                first_val_step=self.first_val_step, 
                forward_only=self.forward_only,
                cond=(self.current_f_microbatch_id[f_vdp_rank] == 0) if f_vdp_rank is not None else None
            ),
            current_microbatch=f_microbatch_id,
            pre_forward=pre_forward,
            post_forward=post_forward,
            pre_backward=pre_backward,
            post_backward=post_backward
        )
        # forward post process
        if f_vdp_rank is not None:
            self._forward_step_helper_postprocess(f_vdp_rank, output_tensor, num_tokens)
        # backward post process
        if b_vdp_rank is not None:
            # The same as the backward_step_helper
            self._backward_step_helper_postprocess(b_vdp_rank)
        return output_tensor, num_tokens, input_tensor_grad
        

    def _forward_step_helper(self, vdp_rank: int) -> None:
        # import here to avoid circle import
        from megatron.core.pipeline_parallel.schedules import forward_step
        input_tensor = self._forward_step_helper_preprocess(vdp_rank)
        microbatch_id = self.current_f_microbatch_id[vdp_rank]

        # doesn't support checkpoint_activations_microbatch yet
        output_tensor, num_tokens = forward_step(
            self.forward_step_func, 
            self.data_iterator[vdp_rank],
            self.model[vdp_rank],
            self.num_microbatches,
            input_tensor,
            self.forward_data_store,
            self.config,
            self.collect_non_loss_data,
            checkpoint_activations_microbatch=None,
            is_first_microbatch=(microbatch_id == 0),
            current_microbatch=microbatch_id,
            vp_stage=vdp_rank,
            is_last_stage=parallel_state.is_pipeline_last_stage(ignore_virtual=False)
        )
        self._forward_step_helper_postprocess(vdp_rank, output_tensor, num_tokens)

    def _backward_step_helper(self, vdp_rank: int) -> None:
        from megatron.core.pipeline_parallel.schedules import backward_step
        if self.forward_only:
            return
        input_tensor, output_tensor, output_tensor_grad = self._backward_step_helper_preprocess(vdp_rank)
        input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad, self.model_type, self.config)
        self._backward_step_helper_postprocess(vdp_rank)
        return input_tensor_grad

    def _forward_with_comm(self, vdp_rank: int, pre_forward: Optional[Callable] = None, post_forward: Optional[Callable] = None, recv: bool=True, send: bool=True) -> None:
        parallel_state.set_virtual_data_parallel_rank(vdp_rank)
        # if recv:
        #     self._recv_forward(vdp_rank)
        # self.comm_mgr.commit()
        torch.cuda.nvtx.range_push(f"model_{vdp_rank}_mbc_{self.current_f_microbatch_id[vdp_rank] + vdp_rank * 10}_f")
        if self.config.overlap_moe_expert_parallel_comm:
            self._combined_1f1b_schedule_for_chimera(f_vdp_rank=vdp_rank, pre_forward=pre_forward, post_forward=post_forward)
        else:
            if pre_forward is not None:
                pre_forward()
            self._forward_step_helper(vdp_rank)
            if post_forward is not None:
                post_forward()
        torch.cuda.nvtx.range_pop()
        # if send:
        #     self._send_forward(vdp_rank)

    def _backward_with_comm(self, vdp_rank: int, pre_backward: Optional[Callable] = None, post_backward: Optional[Callable] = None, recv: bool=True, send: bool=True) -> None:
        parallel_state.set_virtual_data_parallel_rank(vdp_rank)
        # if recv:
        #     self._recv_backward(vdp_rank)
        # if pre_backward is not None:
        #     pre_backward()
        # print(f"[rank {self.rank}]: checkpoint 3, pre_backward = {pre_backward}, post_backward = {post_backward}, comm = {self.comm_mgr.comm_info_for_debug}")
        # self.comm_mgr.commit()
        torch.cuda.nvtx.range_push(f"model_{vdp_rank}_mbc_{self.current_b_microbatch_id[vdp_rank] + vdp_rank * 10}_b")
        if self.config.overlap_moe_expert_parallel_comm:
            self._combined_1f1b_schedule_for_chimera(b_vdp_rank=vdp_rank, pre_backward=pre_backward, post_backward=post_backward)
        else:
            if pre_backward is not None:
                pre_backward()
            input_tensor_grad = self._backward_step_helper(vdp_rank)
            if post_backward is not None:
                post_backward(input_tensor_grad, vdp_rank)
        torch.cuda.nvtx.range_pop()
        # if send:
        #     self._send_backward(vdp_rank)
    
    def _forward_backward_with_comm(self, f_vdp_rank: int, b_vdp_rank: int, recv_0: bool = True, recv_1: bool = True, pre_forward: Optional[Callable] = None, post_forward: Optional[Callable] = None, pre_backward: Callable = None, post_backward: Callable = None) -> None:
        """
            pre_forward: function to be called before forward, most likely is recv_forward and recv_backward
            post_backward: function to be called after backward, most likely is send_forward and send_backward
        """
        # if recv_0:
        #     self._recv_forward(vdp_rank_0)
        # if recv_1:
        #     self._recv_backward(vdp_rank_1)
        # if pre_forward is not None:
        #     pre_forward()
        # self.comm_mgr.commit()
        torch.cuda.nvtx.range_push(f"model_{f_vdp_rank}_mbc_{self.current_f_microbatch_id[f_vdp_rank] + f_vdp_rank * 10}_f_model_{b_vdp_rank}_mbc_{self.current_b_microbatch_id[b_vdp_rank] + b_vdp_rank * 10}_b")
        if self.config.overlap_moe_expert_parallel_comm and not self.forward_only:
            output_tensor, num_tokens, input_tensor_grad = self._combined_1f1b_schedule_for_chimera(f_vdp_rank, b_vdp_rank, pre_forward=pre_forward, post_forward=post_forward, pre_backward=pre_backward, post_backward=post_backward)
        else:
            if pre_forward is not None:
                pre_forward()
            self._forward_step_helper(f_vdp_rank)
            if post_forward is not None:
                post_forward(output_tensor, f_vdp_rank)
            if pre_backward is not None:
                pre_backward()
            input_tensor_grad = self._backward_step_helper(b_vdp_rank)
            if post_backward is not None:
                post_backward(input_tensor_grad, b_vdp_rank)
        torch.cuda.nvtx.range_pop()
        # self._send_forward(f_vdp_rank)
        # self._send_backward(b_vdp_rank)

    def _wgrad_compute_and_p2p_overlap(self, vdp_rank: int = None) -> None:
        """Weight backward and wait communication, if doesn't enable decouple dw, only wait communication"""
        def te_backward_dw(vdp_rank: int):
            model_wrapped = self.model[vdp_rank]
            decoder = get_attr_wrapped_model(model_wrapped, "decoder")
            transformer_layers = decoder.layers
            self_attn_blocks = []
            mlp_blocks = []
            for layer in transformer_layers:
                self_attn_blocks.append(layer.self_attention)
                mlp_blocks.append(layer.mlp)
            for mlp, attn in zip(mlp_blocks, self_attn_blocks):
                mlp.backward_dw()
                attn.backward_dw()

        def local_backward_dw():
            WeightGradStore()

        if self.forward_only:
            return
        # start async p2p communication to overlap
        self.comm_mgr.commit(wait=False)
        if self.config.transformer_impl == 'transformer_engine':
            te_backward_dw(vdp_rank)
        else:
            local_backward_dw()
        self.comm_mgr.wait()
        self._maybe_start_sync(vdp_rank=vdp_rank)

    def _recv_forward(self, tensor_to_comm: torch.Tensor = None, vdp_rank: int = None) -> None:
        parallel_state.set_virtual_data_parallel_rank(vdp_rank)
        is_first_stage = parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vdp_rank)
        if is_first_stage:
            return

        self.current_recv_f_microbatch_id[vdp_rank] += 1
        tensors = self.comm_mgr.recv_forward()
        self.input_tensors[vdp_rank].append(tensors)

    def _send_forward(self, tensor_to_comm: torch.Tensor = None, vdp_rank: int = None) -> None:
        parallel_state.set_virtual_data_parallel_rank(vdp_rank)
        is_last_stage = parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vdp_rank)
        if is_last_stage:
            return

        microbatch_id = self.current_send_f_microbatch_id[vdp_rank]
        # print(f"[rank {self.rank}]: checkpoint 1, len of output_tensors[{vdp_rank}] = {len(self.output_tensors[vdp_rank])}, microbatch_id = {microbatch_id}")
        self.current_send_f_microbatch_id[vdp_rank] += 1
        if tensor_to_comm is None:
            tensor_to_comm = self.output_tensors[vdp_rank][microbatch_id]
        self.comm_mgr.send_forward(tensor_to_comm)

    def _recv_backward(self, tensor_to_comm: torch.Tensor = None, vdp_rank: int = None) -> None:
        if self.forward_only:
            return
        parallel_state.set_virtual_data_parallel_rank(vdp_rank)
        is_last_stage = parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vdp_rank)
        if is_last_stage:
            return

        self.current_recv_b_microbatch_id[vdp_rank] += 1
        tensors = self.comm_mgr.recv_backward()
        self.output_tensor_grads[vdp_rank].append(tensors)

    def _send_backward(self, tensor_to_comm: torch.Tensor = None, vdp_rank: int = None) -> None:
        if self.forward_only:
            return
        parallel_state.set_virtual_data_parallel_rank(vdp_rank)
        is_first_stage = parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vdp_rank)
        if is_first_stage:
            return

        microbatch_id = self.current_send_b_microbatch_id[vdp_rank]
        self.current_send_b_microbatch_id[vdp_rank] += 1
        # print(f"[rank {self.rank}]: checkpoint 2, len of input_tensor_grads[{vdp_rank}] = {len(self.input_tensor_grads[vdp_rank])}, microbatch_id = {microbatch_id}")
        # if tensor_to_comm is None:
        #     tensor_to_comm = self.input_tensor_grads[vdp_rank][microbatch_id]
        # self.input_tensor_grads[vdp_rank][microbatch_id] = None
        self.comm_mgr.send_backward(tensor_to_comm)
    
    def _cat_comm_for_combine_chimera(self, comm_func_list: List[Callable], commit_func: Callable = no_op) -> Callable:
        def combined_comm_func(tensor_to_comm: torch.Tensor = None, *unused_args, **unused_kwargs):
            for comm_func in comm_func_list:
                comm_func(tensor_to_comm=tensor_to_comm)
            commit_func()
        return combined_comm_func

    def wgrad_compute_and_p2p_commit_helper(self, vdp_rank: int, commit: bool = True)->Callable:
        if self.config.delay_wgrad_compute and commit:
            # When overlap_moe_expert_parallel_comm and delay_wgrad_compute are both True, the p2p communication overlap with wgrad compute will be handled in combined_1f1b
            if self.config.overlap_moe_expert_parallel_comm:
                wgrad_and_p2p_overlap = self.comm_mgr.commit
            else:
                wgrad_and_p2p_overlap = partial(self._wgrad_compute_and_p2p_overlap, vdp_rank=vdp_rank)
        else:
            wgrad_and_p2p_overlap = no_op
        return wgrad_and_p2p_overlap

    def commit_wrapper(self, *unused_args, **unused_kwargs):
        self.comm_mgr.commit()
    
    def _is_last_microbatch(self, vdp_rank):
        microbatch_id = self.current_b_microbatch_id[vdp_rank]
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

    def run(self):
        """
        Because the communication of multiple pipeline is complex, 
        reader can refence to the dualpipe
        """
        self.disable_grad_sync()
        is_middle_rank = ChimeraNode.is_middle_rank
        # Synchronize params for first two model chunks
        if self.config.param_sync_func is not None:
            self.config.param_sync_func[0](self.model[0].parameters())
            self.config.param_sync_func[1](self.model[1].parameters())
        # Step 1: nF0
        F0_nodes = self.schedule["F0"]
        for node in F0_nodes:
            vdp_rank = node.virtual_data_parallel_rank
            self._recv_forward(vdp_rank=vdp_rank)
            pre_forward = self.commit_wrapper
            post_forward = self._cat_comm_for_combine_chimera([partial(self._send_forward, vdp_rank=vdp_rank)], no_op)
            self._forward_with_comm(vdp_rank, pre_forward=pre_forward, post_forward=post_forward)
        # Step 2: nF0F1
        F0F1_nodes = self.schedule["F0F1"]
        self._recv_forward(vdp_rank=F0F1_nodes[0].virtual_data_parallel_rank)    # dualpipe: recv_forward(0)
        for i in range(0, len(F0F1_nodes), 2):
            node0 = F0F1_nodes[i]
            node1 = F0F1_nodes[i + 1]
            vdp_rank0 = node0.virtual_data_parallel_rank
            vdp_rank1 = node1.virtual_data_parallel_rank

            if i < len(F0F1_nodes) - 2:
                # self._recv_forward(F0F1_nodes[i + 2].virtual_data_parallel_rank)
                next_F0_recv = partial(self._recv_forward, vdp_rank=F0F1_nodes[i + 2].virtual_data_parallel_rank)
            else:
                next_F0_recv = partial(self._recv_forward, vdp_rank=self.schedule["F0B1F1B0"][0].virtual_data_parallel_rank)
                # recv next down pipeline
                # self._recv_forward(self.schedule["F0B1F1B0"][0].virtual_data_parallel_rank)
            F1_recv = partial(self._recv_forward, vdp_rank=node1.virtual_data_parallel_rank)
            if is_middle_rank:
                F0_send = partial(self._send_forward, vdp_rank=vdp_rank0)
                F0_post_forward = self._cat_comm_for_combine_chimera([next_F0_recv, F0_send, F1_recv], no_op)
            else:
                F0_send = None
                F0_post_forward = self._cat_comm_for_combine_chimera([next_F0_recv, F1_recv], no_op)
            # self._forward_with_comm(vdp_rank0, recv=False, send=is_middle_rank)
            self._forward_with_comm(vdp_rank0, pre_forward=self.commit_wrapper, post_forward=F0_post_forward)

            # dualpipe recv_forward(0)
            # if i < len(F0F1_nodes) - 2:
            #     self._recv_forward(F0F1_nodes[i + 2].virtual_data_parallel_rank)
            # else:
            #     # recv next down pipeline
            #     self._recv_forward(self.schedule["F0B1F1B0"][0].virtual_data_parallel_rank)
            if (not is_middle_rank) or (i < len(F0F1_nodes) - 2):
                F1_post_forward = self._cat_comm_for_combine_chimera([partial(self._send_forward, vdp_rank=vdp_rank1)], no_op)
            else:
                F1_post_forward = None
            # self._forward_with_comm(vdp_rank1, send=(not is_middle_rank) or (i < len(F0F1_nodes) - 2))
            self._forward_with_comm(vdp_rank1, pre_forward=self.commit_wrapper, post_forward=F1_post_forward)
            if not is_middle_rank:
                self._send_forward(vdp_rank=vdp_rank0)
        
        # Step 3: nB1W1F1
        B1F1_nodes = self.schedule["B1F1"]
        for i in range(0, len(B1F1_nodes), 2):
            node0 = B1F1_nodes[i]   # B1
            node1 = B1F1_nodes[i + 1]   # F1
            self._recv_backward(vdp_rank=node0.virtual_data_parallel_rank)
            B1_pre_backward = self.commit_wrapper
            B1_send_func = partial(self._send_backward, vdp_rank=node0.virtual_data_parallel_rank)
            F1_recv_func = partial(self._recv_forward, vdp_rank=node1.virtual_data_parallel_rank)
            # handle wgrad compute and p2p overlap
            wgrad_and_p2p_overlap = self.wgrad_compute_and_p2p_commit_helper(node0.virtual_data_parallel_rank)
            B1_post_backward = self._cat_comm_for_combine_chimera([F1_recv_func, B1_send_func], wgrad_and_p2p_overlap)
            self._backward_with_comm(node0.virtual_data_parallel_rank, pre_backward=B1_pre_backward, post_backward=B1_post_backward)
            # self._recv_forward(node1.virtual_data_parallel_rank)
            # if self.config.delay_wgrad_compute:
            #     self._wgrad_compute_and_p2p_overlap(node0.virtual_data_parallel_rank)
            F1_post_forward = self._cat_comm_for_combine_chimera([partial(self._send_forward, vdp_rank=node1.virtual_data_parallel_rank)], no_op)
            # self._forward_with_comm(node1.virtual_data_parallel_rank, recv=False)
            self._forward_with_comm(node1.virtual_data_parallel_rank, self.commit_wrapper, post_forward=F1_post_forward)
        

        # Step 4 : nF0B1F1B0
        F0B1F1F0_nodes = self.schedule["F0B1F1B0"]
        for i in range(0, len(F0B1F1F0_nodes), 4):
            F0_node = F0B1F1F0_nodes[i]
            B1_node = F0B1F1F0_nodes[i + 1]
            F1_node = F0B1F1F0_nodes[i + 2]
            B0_node = F0B1F1F0_nodes[i + 3]
            
            F0_post_forward = self._cat_comm_for_combine_chimera(
                [
                    partial(self._send_forward, vdp_rank=F0_node.virtual_data_parallel_rank),
                    partial(self._recv_forward, vdp_rank=F1_node.virtual_data_parallel_rank),
                ],
                no_op
            )
            F0B1_post_backward = self._cat_comm_for_combine_chimera(
                [
                    partial(self._send_backward, vdp_rank=B1_node.virtual_data_parallel_rank),
                    partial(self._recv_backward, vdp_rank=B0_node.virtual_data_parallel_rank),
                ],
                self.wgrad_compute_and_p2p_commit_helper(B1_node.virtual_data_parallel_rank)
            )
            if i == 0:
                if is_middle_rank:
                    # self._forward_with_comm(F0_node.virtual_data_parallel_rank, recv=False, send=False)
                    self._forward_with_comm(F0_node.virtual_data_parallel_rank, pre_forward=self.commit_wrapper)
                    self._send_forward(vdp_rank=F0F1_nodes[-1].virtual_data_parallel_rank)
                    self._recv_backward(vdp_rank=B1_node.virtual_data_parallel_rank)
                    self.comm_mgr.commit()
                    self._send_forward(vdp_rank=F0_node.virtual_data_parallel_rank)
                    self._recv_forward(vdp_rank=F1_node.virtual_data_parallel_rank)
                    # self._backward_with_comm(B1_node.virtual_data_parallel_rank, send=False)
                    # self._send_forward(F0_node.virtual_data_parallel_rank)
                    # self._send_backward(B1_node.virtual_data_parallel_rank)
                    self._backward_with_comm(B1_node.virtual_data_parallel_rank, post_backward=F0B1_post_backward)
                else:    
                    # self._forward_backward_with_comm(F0_node.virtual_data_parallel_rank, B1_node.virtual_data_parallel_rank, recv_0=False)
                    self._recv_backward(vdp_rank=B1_node.virtual_data_parallel_rank)
                    B1_pre_backward = self.commit_wrapper
                    self._forward_backward_with_comm(F0_node.virtual_data_parallel_rank, B1_node.virtual_data_parallel_rank, pre_forward=self.commit_wrapper, post_forward=F0_post_forward, pre_backward=B1_pre_backward, post_backward=F0B1_post_backward)
                # self.last_backward_vdp_rank = B1_node.virtual_data_parallel_rank
            else:
                # These two recv are execute at the previous_F1B0 post_backward
                self._forward_backward_with_comm(F0_node.virtual_data_parallel_rank, B1_node.virtual_data_parallel_rank, pre_forward=self.commit_wrapper, post_forward=F0_post_forward, pre_backward=None, post_backward=F0B1_post_backward)
            # Add next F0B1 recv comm to this F1B0 post_backward
            # When this is the last F0B1F1B0 node, it's too complex to add next comm so that overlap. 
            # As a consequence, the last node will not overlap the comm, as seen that wgrad_compute_helper's wait is False, meaning that comm just are commited but not start.
            
            F1_post_forward = [partial(self._send_forward, vdp_rank=F1_node.virtual_data_parallel_rank), no_op]
            B0_post_backward = [partial(self._send_backward, vdp_rank=B0_node.virtual_data_parallel_rank), no_op]
            if i + 4 < len(F0B1F1F0_nodes):
                F1_post_forward[1] = partial(self._recv_forward, vdp_rank=F0B1F1F0_nodes[i + 4].virtual_data_parallel_rank)
                B0_post_backward[1] = partial(self._recv_backward, vdp_rank=F0B1F1F0_nodes[i + 5].virtual_data_parallel_rank)
            F1_post_forward = self._cat_comm_for_combine_chimera(F1_post_forward, no_op)
            B0_post_backward = self._cat_comm_for_combine_chimera(B0_post_backward, self.wgrad_compute_and_p2p_commit_helper(B0_node.virtual_data_parallel_rank, commit=(i + 4 < len(F0B1F1F0_nodes))))
            self._forward_backward_with_comm(F1_node.virtual_data_parallel_rank, B0_node.virtual_data_parallel_rank, pre_forward=self.commit_wrapper, post_forward=F1_post_forward, pre_backward=None, post_backward=B0_post_backward)
        
        # Step 5: n partial F0B1F1B0
        partial_F0B1F1B0_nodes = self.schedule["Partial_F0B1F1B0"]
        for i in range(0, len(partial_F0B1F1B0_nodes), 2):
            F_node = partial_F0B1F1B0_nodes[i]
            B_node = partial_F0B1F1B0_nodes[i + 1]

            # [send_B, rev_next_f, recv_next_B]
            FB_post_backward_func_list = [None for _ in range(3)]
            if i < len(partial_F0B1F1B0_nodes) - 2:
                next_F_node = partial_F0B1F1B0_nodes[i + 2]
                next_B_node = partial_F0B1F1B0_nodes[i + 3]
            else:
                next_F_node = None
                next_B_node = None
            # set the nessesary comm func
            FB_post_backward_func_list[0] = partial(self._send_backward, vdp_rank=B_node.virtual_data_parallel_rank)
            # print(f"[rank {self.rank}]: wait = {i + 2 < len(partial_F0B1F1B0_nodes)}, i = {i}, len = {len(partial_F0B1F1B0_nodes)}")
            # # set the rest comm func according to the F node and next_F and next_B node status
            if next_F_node is None:
                FB_post_backward_func_list[1] = no_op
                FB_post_backward_func_list[2] = no_op
            elif next_F_node.is_idle():
                FB_post_backward_func_list[1] = no_op
                FB_post_backward_func_list[2] = partial(self._recv_backward, vdp_rank=next_B_node.virtual_data_parallel_rank)

            else:
                FB_post_backward_func_list[1] = partial(self._recv_forward, vdp_rank=next_F_node.virtual_data_parallel_rank)
                FB_post_backward_func_list[2] = partial(self._recv_backward, vdp_rank=next_B_node.virtual_data_parallel_rank)
            # set the pre_forward and post_backward for the FB_node according to F_node status
            if F_node.is_idle():
                if i == 0:
                    self._recv_backward(vdp_rank=B_node.virtual_data_parallel_rank)
                    B_pre_backward = self.commit_wrapper
                else:
                    B_pre_backward = self.commit_wrapper
                # print(f"[rank {self.rank}]: checkpoint 1, pre_backward = {B_pre_backward}, post_backward = {FB_post_backward_func_list}")
                self._backward_with_comm(B_node.virtual_data_parallel_rank, pre_backward=B_pre_backward, post_backward=self._cat_comm_for_combine_chimera(FB_post_backward_func_list, self.wgrad_compute_and_p2p_commit_helper(B_node.virtual_data_parallel_rank, commit=(i + 2 < len(partial_F0B1F1B0_nodes)))))
            else:
                # prepare pre_forward of FB_node
                if i == 0:
                    FB_pre_forward = self._cat_comm_for_combine_chimera(
                        [
                            partial(self._recv_forward, vdp_rank=F_node.virtual_data_parallel_rank),
                            partial(self._recv_backward, vdp_rank=B_node.virtual_data_parallel_rank)
                        ],
                        self.commit_wrapper
                    )
                else:
                    FB_pre_forward = self.commit_wrapper
                # print(f"[rank {self.rank}]: checkpoint 2")
                F_post_forward = self._cat_comm_for_combine_chimera([partial(self._send_forward, vdp_rank=F_node.virtual_data_parallel_rank)], no_op)
                self._forward_backward_with_comm(F_node.virtual_data_parallel_rank, B_node.virtual_data_parallel_rank, pre_forward=FB_pre_forward, post_forward=F_post_forward, pre_backward=None, post_backward=self._cat_comm_for_combine_chimera(FB_post_backward_func_list, self.wgrad_compute_and_p2p_commit_helper(B_node.virtual_data_parallel_rank, commit=(i + 2 < len(partial_F0B1F1B0_nodes)))))
            # if B1_enable_decouple_dw:
            #     if not F1_node.is_idle():
            #         self._recv_forward(F1_node.virtual_data_parallel_rank)
            #     else:
            #         self._recv_backward(B0_node.virtual_data_parallel_rank)
            #     self._wgrad_compute_and_p2p_overlap()
            # if F1_node.is_idle():
            #     self._backward_with_comm(B0_node.virtual_data_parallel_rank, recv=not B1_enable_decouple_dw)
            # else:
            #     self._forward_backward_with_comm(F1_node.virtual_data_parallel_rank, B0_node.virtual_data_parallel_rank, recv_0=not B1_enable_decouple_dw)
            ## TODO
            # if F1_node.is_idle():
            #     if self.config.delay_wgrad_compute:
            #         self._recv_backward(B0_node.virtual_data_parallel_rank)
            #         self._wgrad_compute_and_p2p_overlap(self.last_backward_vdp_rank)
            #         self.last_backward_vdp_rank = None
            #     self._backward_with_comm(B0_node.virtual_data_parallel_rank, recv=not self.config.delay_wgrad_compute)
            #     self.last_backward_vdp_rank = B0_node.virtual_data_parallel_rank
            # else:
            #     if self.config.delay_wgrad_compute:
            #         self._recv_forward(F1_node.virtual_data_parallel_rank)
            #         self._recv_backward(B0_node.virtual_data_parallel_rank)
            #         self._wgrad_compute_and_p2p_overlap(self.last_backward_vdp_rank)
            #         self.last_backward_vdp_rank = None
            #         recv_0, recv_1 = False, False
            #     else:
            #         recv_0, recv_1 = True, True
            #     self._forward_backward_with_comm(F1_node.virtual_data_parallel_rank, B0_node.virtual_data_parallel_rank, recv_0=recv_0, recv_1=recv_1)
            #     self.last_backward_vdp_rank = B0_node.virtual_data_parallel_rank


        
        # Step 6: n B1B0
        # B1B0_nodes = self.schedule["B1B0"]
        # for i in range(0, len(B1B0_nodes), 2):
        #     # backward B1, enable decouple dw in the not middle rank
        #     B1_node = B1B0_nodes[i]
        #     B0_node = B1B0_nodes[i + 1]
        #     if self.config.delay_wgrad_compute:
        #         self._recv_backward(B1_node.virtual_data_parallel_rank)
        #         self._wgrad_compute_and_p2p_overlap(self.last_backward_vdp_rank)
        #         self.last_backward_vdp_rank = None
        #     self._backward_with_comm(B1_node.virtual_data_parallel_rank, recv=not self.config.delay_wgrad_compute)
        #     self.last_backward_vdp_rank = B1_node.virtual_data_parallel_rank
        #     # self._recv_backward(B0_node.virtual_data_parallel_rank)
        #     # if B1_enable_decouple_dw:
        #     #     self._wgrad_compute_and_p2p_overlap()
        #     # backward B0
        #     # B0_enable_decouple_dw = i == len(B1B0_nodes) - 2
        #     if self.config.delay_wgrad_compute:
        #         self._recv_backward(B0_node.virtual_data_parallel_rank)
        #         self._wgrad_compute_and_p2p_overlap(self.last_backward_vdp_rank)
        #         self.last_backward_vdp_rank = None
        #     self._backward_with_comm(B0_node.virtual_data_parallel_rank, recv=not self.config.delay_wgrad_compute)
        #     self.last_backward_vdp_rank = B0_node.virtual_data_parallel_rank
        
        # Step 7: n B1 or B0
        single_B_nodes = self.schedule["B1B0"] + self.schedule["Single_B"]
        # if len(single_B_nodes) > 0:
        #     self._recv_backward(single_B_nodes[0].virtual_data_parallel_rank)
        for i in range(0, len(single_B_nodes)):
            node = single_B_nodes[i]
            # if self.config.delay_wgrad_compute:
            #     self._recv_backward(node.virtual_data_parallel_rank)
            #     self._wgrad_compute_and_p2p_overlap(self.last_backward_vdp_rank)
            #     self.last_backward_vdp_rank = None
            # prepare pre_backward for B node
            if i == 0:
                self._recv_backward(vdp_rank=node.virtual_data_parallel_rank)
            pre_backward = self.commit_wrapper
            # prepare post_backward for B node
            if i < len(single_B_nodes) - 1:
                next_B_node = single_B_nodes[i + 1]
            else:
                next_B_node = None
            # [send_B, recv_next_B, wgrad_and_p2p_overlap]
            B_post_backward_func_list = [None for _ in range(2)]
            B_post_backward_func_list[0] = partial(self._send_backward, vdp_rank=node.virtual_data_parallel_rank)
            if next_B_node is None:
                B_post_backward_func_list[1] = no_op
            else:
                B_post_backward_func_list[1] = partial(self._recv_backward, vdp_rank=next_B_node.virtual_data_parallel_rank)
            self._backward_with_comm(node.virtual_data_parallel_rank, pre_backward=pre_backward, post_backward=self._cat_comm_for_combine_chimera(B_post_backward_func_list, self.wgrad_compute_and_p2p_commit_helper(node.virtual_data_parallel_rank)))
            # self.last_backward_vdp_rank = node.virtual_data_parallel_rank
            # if node is not the last one, recv next one and send this one
            # if i < len(single_B_nodes) - 1:
            #     self._recv_backward(single_B_nodes[i + 1].virtual_data_parallel_rank)
            # if self.enable_decouple_dw:
            #     self._wgrad_compute_and_p2p_overlap()
        # pipeline finish
        # if self.config.delay_wgrad_compute:
        #     self._wgrad_compute_and_p2p_overlap(self.last_backward_vdp_rank)
        #     self.last_backward_vdp_rank = None
        self.comm_mgr.commit()
        self._finish_sync()
        # Set to the last pipeline stage needed virtual data parallel rank.
        # Only the stage that maybe be the last stage will be set.
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        pp = parallel_state.get_pipeline_model_parallel_world_size()
        offset_unit = pp // (self.virtual_data_parallel_world_size // 2)
        if pp_rank % offset_unit == 0:
            # formmer half of vdp ranks
            vdp_rank = pp_rank // offset_unit
            last_stage_vdp_rank = vdp_rank + self.virtual_data_parallel_world_size // 2
        else:
            # latter half of vdp ranks
            vdp_rank = (pp - 1 - pp_rank) // offset_unit
            last_stage_vdp_rank = vdp_rank
        parallel_state.set_virtual_data_parallel_rank(last_stage_vdp_rank)
        self.enable_grad_sync()
        if self.forward_only:
            self.config.grad_sync_func, self.config.param_sync_func = self.grad_sync_func, self.param_sync_func
        return self.forward_data_store

    def __call__(self, *args, **kwds):
        if self.first_run:
            self.prepare(*args, **kwds)
        else:
            self.reset()
        return self.run()