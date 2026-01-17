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
import logging
logger = logging.getLogger(__name__)
from .auto_schedule import ChimeraAutoSchedule, ChimeraNode
from .communication import CommunicationManager


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
        # Megatron config
        # import here to avoid circle import
        from megatron.core.pipeline_parallel.schedules import clear_embedding_activation_buffer, deallocate_output_tensor
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

        # Distributed config
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.virtual_data_parallel_world_size = parallel_state.get_virtual_data_parallel_world_size()
        self.schedule = ChimeraAutoSchedule(self.rank, self.world_size, parallel_state.get_pipeline_model_parallel_world_size(), parallel_state.get_virtual_data_parallel_world_size(), num_microbatches).get_schedule()
        self.reorder_models(model)
        self.param_sync_model_chunk_id_queue: List[int] = None
        pp_world_size = parallel_state.get_pipeline_model_parallel_world_size()

        # Check whether there are 1f1b_overlap block
        F0B1F1B0_nodes = self.schedule["F0B1F1B0"]
        if len(F0B1F1B0_nodes) == 0:
            self.config.overlap_moe_expert_parallel_comm = False
            logger.warning(f"Overlap_moe_expert_parallel_comm is not supported under the number of microbatches {num_microbatches}, set it to False")
        if self.rank == 0:
            logger.info(f">>>ChimeraScheduler finishes initialization.")

        # Megatron config postprocess
        if self.config.overlap_p2p_comm and self.config.batch_p2p_comm:
            raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")
        self.num_model_chunks = len(model)
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

        self.model_type = get_model_type(model[0])
        # Chimera config
        self.comm_mgr = CommunicationManager(
            num_microbatches=self.num_microbatches,
            tensor_shape=self.tensor_shape,
            tensor_dtype=self.config.pipeline_dtype,
            deallocate_fn=partial(deallocate_output_tensor, deallocate_pipeline_outputs=self.config.deallocate_pipeline_outputs),
            virtual_data_parallel_world_size=self.virtual_data_parallel_world_size
        )
        self.reset()
        self.reorder_param_sync_funcs_flag = False
    
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

        self.comm_mgr.reset()
        # Disable config.grad_sync_func and config.param_sync_func if only running forward passes.
        # They will be re-enabled at the end of this function.
        self.grad_sync_func, self.param_sync_func = None, None
        if self.forward_only:
            self.grad_sync_func, self.param_sync_func = self.config.grad_sync_func, self.config.param_sync_func
            self.config.grad_sync_func, self.config.param_sync_func = None, None
        if not self.first_run:
            self.reorder_param_sync_funcs()

    
    def reorder_models(self, model):
        self.config = get_model_config(model[0])
        self.ddp_config = model[0].ddp_config
        if self.config.grad_sync_func is not None and not isinstance(self.config.grad_sync_func, list):
            self.config.grad_sync_func = [self.config.grad_sync_func for _ in model]
        vdp_size = self.virtual_data_parallel_world_size
        tmp_models = [None] * vdp_size
        tmp_no_sync_funcs = [None] * vdp_size
        if self.config.grad_sync_func is not None:
            tmp_grad_sync_funcs = [None] * vdp_size
        else:
            tmp_grad_sync_funcs = None
        for i in range(vdp_size):
            first_vdp_pp_stage = parallel_state.get_pp_stage_of_first_vdp_rank(i)
            model_correct_idx = parallel_state.pipeline_stage_to_virtual_data_parallel_rank(first_vdp_pp_stage)
            tmp_models[model_correct_idx] = model[i]
            tmp_no_sync_funcs[model_correct_idx] = self.config.no_sync_func[i]
            if tmp_grad_sync_funcs is not None:
                tmp_grad_sync_funcs[model_correct_idx] = self.config.grad_sync_func[i]
        self.model = tmp_models
        self.config.no_sync_func = tmp_no_sync_funcs
        self.config.grad_sync_func = tmp_grad_sync_funcs
    
    def reorder_param_sync_funcs(self):
        if self.reorder_param_sync_funcs_flag:
            return
        vdp_size = self.virtual_data_parallel_world_size
        if self.config.param_sync_func is not None:
            tmp_param_sync_funcs = [None] * vdp_size
        else:
            tmp_param_sync_funcs = None
        for i in range(vdp_size):
            first_vdp_pp_stage = parallel_state.get_pp_stage_of_first_vdp_rank(i)
            model_correct_idx = parallel_state.pipeline_stage_to_virtual_data_parallel_rank(first_vdp_pp_stage)
            if tmp_param_sync_funcs is not None:
                tmp_param_sync_funcs[model_correct_idx] = self.config.param_sync_func[i]
        self.config.param_sync_func = tmp_param_sync_funcs
        self.reorder_param_sync_funcs_flag = True

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
        self.current_b_microbatch_id[vdp_rank] += 1
        if self._is_last_microbatch(vdp_rank) and self.config.grad_sync_func is None:
            self.enable_grad_sync()
            self.sync_model_queue.append(vdp_rank)
            self.synchronized_model_chunks.add(vdp_rank)
            logger.debug(f"[rank{self.rank}]: add model chunk {vdp_rank} to sync queue")
        return input_tensor, output_tensor, output_tensor_grad

    def _backward_step_helper_postprocess(self, vdp_rank, intput_tensor_grad):
        """Postprocess for backward_step_helper"""
        self.input_tensor_grads[vdp_rank].append(intput_tensor_grad)
    
    def _combined_1f1b_schedule_for_chimera(self, f_vdp_rank: Optional[int] = None, b_vdp_rank: Optional[int] = None) -> None:
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
            b_model_chunk_id=b_vdp_rank,
            collect_non_loss_data=self.collect_non_loss_data,
            checkpoint_activations_microbatch=None,
            is_first_microbatch=check_first_val_step(
                first_val_step=self.first_val_step, 
                forward_only=self.forward_only,
                cond=(self.current_f_microbatch_id[f_vdp_rank] == 0) if f_vdp_rank is not None else None
            ),
            current_microbatch=f_microbatch_id,
        )
        # forward post process
        if f_vdp_rank is not None:
            self._forward_step_helper_postprocess(f_vdp_rank, output_tensor, num_tokens)
        # backward post process
        if b_vdp_rank is not None:
            # The same as the backward_step_helper
            self._backward_step_helper_postprocess(b_vdp_rank, input_tensor_grad)
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
        self._backward_step_helper_postprocess(vdp_rank, input_tensor_grad)
        return input_tensor_grad

    def _forward_with_comm(self, vdp_rank: int, recv: bool=True, send: bool=True) -> None:
        parallel_state.set_virtual_data_parallel_rank(vdp_rank)
        if recv:
            self._recv_forward(vdp_rank)
        self.comm_mgr.commit()
        torch.cuda.nvtx.range_push(f"model_{vdp_rank}_mbc_{self.current_f_microbatch_id[vdp_rank] + vdp_rank * 10}_f")
        if self.config.overlap_moe_expert_parallel_comm:
            self._combined_1f1b_schedule_for_chimera(f_vdp_rank=vdp_rank)
        else:
            self._forward_step_helper(vdp_rank)
        torch.cuda.nvtx.range_pop()
        if send:
            self._send_forward(vdp_rank)

    def _backward_with_comm(self, vdp_rank: int, recv: bool=True, send: bool=True) -> None:
        parallel_state.set_virtual_data_parallel_rank(vdp_rank)
        if recv:
            self._recv_backward(vdp_rank)
        self.comm_mgr.commit()
        torch.cuda.nvtx.range_push(f"model_{vdp_rank}_mbc_{self.current_b_microbatch_id[vdp_rank] + vdp_rank * 10}_b")
        if self.config.overlap_moe_expert_parallel_comm:
            self._combined_1f1b_schedule_for_chimera(b_vdp_rank=vdp_rank)
        else:
            self._backward_step_helper(vdp_rank)
        torch.cuda.nvtx.range_pop()
        if send:
            self._send_backward(vdp_rank)
    
    def _forward_backward_with_comm(self, f_vdp_rank: int, b_vdp_rank: int, recv_0: bool = True, recv_1: bool = True) -> None:
        """
            pre_forward: function to be called before forward, most likely is recv_forward and recv_backward
            post_backward: function to be called after backward, most likely is send_forward and send_backward
        """
        if recv_0:
            self._recv_forward(f_vdp_rank)
        if recv_1:
            self._recv_backward(b_vdp_rank)
        self.comm_mgr.commit()
        torch.cuda.nvtx.range_push(f"model_{f_vdp_rank}_mbc_{self.current_f_microbatch_id[f_vdp_rank] + f_vdp_rank * 10}_f_model_{b_vdp_rank}_mbc_{self.current_b_microbatch_id[b_vdp_rank] + b_vdp_rank * 10}_b")
        if self.config.overlap_moe_expert_parallel_comm and not self.forward_only:
            self._combined_1f1b_schedule_for_chimera(f_vdp_rank, b_vdp_rank)
        else:
            self._forward_step_helper(f_vdp_rank)
            self._backward_step_helper(b_vdp_rank)
        torch.cuda.nvtx.range_pop()
        self._send_forward(f_vdp_rank)
        self._send_backward(b_vdp_rank)

    def _wgrad_compute_and_p2p_overlap(self, vdp_rank: int = None, commit: bool = True) -> None:
        """Weight backward and wait communication, if doesn't enable decouple dw, only wait communication"""
        def te_backward_dw(vdp_rank: int):
            model_wrapped = self.model[vdp_rank]
            decoder = get_attr_wrapped_model(model_wrapped, "decoder")
            transformer_layers = decoder.layers
            for layer in transformer_layers:
                layer.mlp.backward_dw()
                layer.self_attention.backward_dw()

        def local_backward_dw():
            WeightGradStore.pop()

        if self.forward_only:
            return
        # start async p2p communication to overlap
        if commit:
            self.comm_mgr.commit(wait=False)
        if self.config.transformer_impl == 'transformer_engine':
            te_backward_dw(vdp_rank)
        else:
            local_backward_dw()
        if commit:
            self.comm_mgr.wait()
        if self.config.grad_sync_func is not None and self._is_last_microbatch(vdp_rank):
            self.enable_grad_sync()
            self.config.grad_sync_func[vdp_rank](self.model[vdp_rank].parameters())
            self.sync_model_queue.append(vdp_rank)
            self.synchronized_model_chunks.add(vdp_rank)
            logger.debug(f"[rank{self.rank}]: add model chunk {vdp_rank} to sync queue")
        self.disable_grad_sync()

    def _recv_forward(self, vdp_rank: int = None) -> None:
        parallel_state.set_virtual_data_parallel_rank(vdp_rank)
        is_first_stage = parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vdp_rank)
        if is_first_stage:
            return

        self.current_recv_f_microbatch_id[vdp_rank] += 1
        tensors = self.comm_mgr.recv_forward()
        self.input_tensors[vdp_rank].append(tensors)

    def _send_forward(self, vdp_rank: int = None) -> None:
        parallel_state.set_virtual_data_parallel_rank(vdp_rank)
        is_last_stage = parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vdp_rank)
        if is_last_stage:
            return

        microbatch_id = self.current_send_f_microbatch_id[vdp_rank]
        self.current_send_f_microbatch_id[vdp_rank] += 1
        tensor_to_comm = self.output_tensors[vdp_rank][microbatch_id]
        self.comm_mgr.send_forward(tensor_to_comm)

    def _recv_backward(self, vdp_rank: int = None) -> None:
        if self.forward_only:
            return
        parallel_state.set_virtual_data_parallel_rank(vdp_rank)
        is_last_stage = parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vdp_rank)
        if is_last_stage:
            return

        self.current_recv_b_microbatch_id[vdp_rank] += 1
        tensors = self.comm_mgr.recv_backward()
        self.output_tensor_grads[vdp_rank].append(tensors)

    def _send_backward(self, vdp_rank: int = None) -> None:
        if self.forward_only:
            return
        parallel_state.set_virtual_data_parallel_rank(vdp_rank)
        is_first_stage = parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vdp_rank)
        if is_first_stage:
            return

        microbatch_id = self.current_send_b_microbatch_id[vdp_rank]
        self.current_send_b_microbatch_id[vdp_rank] += 1
        tensor_to_comm = self.input_tensor_grads[vdp_rank][microbatch_id]
        self.input_tensor_grads[vdp_rank][microbatch_id] = None
        self.comm_mgr.send_backward(tensor_to_comm)
    
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
            logger.debug(f"[rank{self.rank}]: model chunk {i} finish grads sync")
            parallel_state.set_virtual_data_parallel_rank(i)
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
        # Synchronize params for first model chunks
        if self.config.param_sync_func is not None:
            param_sync_model_chunk_id = self.param_sync_model_chunk_id_queue[0]
            self.config.param_sync_func[param_sync_model_chunk_id](self.model[param_sync_model_chunk_id].parameters())
            self.param_sync_idx += 1
        # Step 1: nF0
        logger.debug(f"[rank{self.rank}]: Step 1")
        F0_nodes = self.schedule["F0"]
        for node in F0_nodes:
            if node.is_idle():
                self.comm_mgr.commit()
            else:
                vdp_rank = node.virtual_data_parallel_rank
                self._forward_with_comm(vdp_rank)
        # Step 2: nF0F1
        logger.debug(f"[rank{self.rank}]: Step 2")
        F0F1_nodes = self.schedule["F0F1"]
        self_vdp = F0F1_nodes[0].virtual_data_parallel_rank
        # If the vdp_size > 2, all the microbatch need to send immediately.
        if not F0F1_nodes[0].is_idle():
            self._recv_forward(F0F1_nodes[0].virtual_data_parallel_rank)
        send_immediately = False
        for i in range(0, len(F0F1_nodes), 2):
            node0 = F0F1_nodes[i]
            node1 = F0F1_nodes[i + 1]
            vdp_rank0 = node0.virtual_data_parallel_rank
            vdp_rank1 = node1.virtual_data_parallel_rank
            if node0.is_idle():
                self.comm_mgr.commit()
            else:
                self._forward_with_comm(vdp_rank0, recv=send_immediately, send=is_middle_rank or send_immediately)
            if i < len(F0F1_nodes) - 2:
                # When new vdp_rank occurs, set the send_immediately. Otherwise, recv the next F0.
                next_F_node = F0F1_nodes[i + 2]
                if not next_F_node.is_idle():
                    if next_F_node.virtual_data_parallel_rank == self_vdp:
                        self._recv_forward(next_F_node.virtual_data_parallel_rank)
                    else:
                        send_immediately = True
            else:
                # If the F0F1_nodes are finished, recv the first node of F0B1F1B0_nodes.
                if len(self.schedule["F0B1F1B0"]) > 0:
                    self._recv_forward(self.schedule["F0B1F1B0"][0].virtual_data_parallel_rank)
            self._forward_with_comm(vdp_rank1, send=(not is_middle_rank) or (i < len(F0F1_nodes) - 2))
            if not node0.is_idle() and not (is_middle_rank or send_immediately):
                self._send_forward(vdp_rank=vdp_rank0)
        # If the F0B1F1B0_nodes is empty, where the num_microbatch is below than 2 * pp - vdp_size.
        # The middle ranks need to send the last node of F0F1_nodes.
        # If the vdp_size > 2, there will leave a node of the self_vdp_rank does not be sent, and send it here.
        # The corresponding code is provided in Line 598.
        if is_middle_rank:
            if len(self.schedule["F0B1F1B0"]) == 0:
                self._send_forward(F0F1_nodes[-1].virtual_data_parallel_rank)
        else:
            if not F0F1_nodes[0].is_idle() and send_immediately:
                self._send_forward(self_vdp)

        # Step 3: nB1W1F1
        logger.debug(f"[rank{self.rank}]: Step 3")
        B1F1_nodes = self.schedule["B1F1"]
        for i in range(0, len(B1F1_nodes), 2):
            node0 = B1F1_nodes[i]   # B1
            node1 = B1F1_nodes[i + 1]   # F1
            self._backward_with_comm(node0.virtual_data_parallel_rank)
            self._recv_forward(node1.virtual_data_parallel_rank)
            self._wgrad_compute_and_p2p_overlap(node0.virtual_data_parallel_rank)
            self._forward_with_comm(node1.virtual_data_parallel_rank, recv=False)
        

        # Step 4 : nF0B1F1B0
        logger.debug(f"[rank{self.rank}]: Step 4")
        F0B1F1F0_nodes = self.schedule["F0B1F1B0"]
        for i in range(0, len(F0B1F1F0_nodes), 4):
            F0_node = F0B1F1F0_nodes[i]
            B1_node = F0B1F1F0_nodes[i + 1]
            F1_node = F0B1F1F0_nodes[i + 2]
            B0_node = F0B1F1F0_nodes[i + 3]

            if i == 0:
                if is_middle_rank:
                    # self._forward_with_comm(F0_node.virtual_data_parallel_rank, recv=False, send=False)
                    self._forward_with_comm(F0_node.virtual_data_parallel_rank, recv=False, send=False)
                    self._send_forward(vdp_rank=F0F1_nodes[-1].virtual_data_parallel_rank)
                    self._backward_with_comm(B1_node.virtual_data_parallel_rank, send=False)
                    self._send_forward(F0_node.virtual_data_parallel_rank)
                    self._send_backward(B1_node.virtual_data_parallel_rank)
                else:    
                    self._forward_backward_with_comm(F0_node.virtual_data_parallel_rank, B1_node.virtual_data_parallel_rank, recv_0=False)
                # self.last_backward_vdp_rank = B1_node.virtual_data_parallel_rank
            else:
                # These two recv are execute at the previous_F1B0 post_backward
                self._forward_backward_with_comm(F0_node.virtual_data_parallel_rank, B1_node.virtual_data_parallel_rank)
            self._wgrad_compute_and_p2p_overlap(vdp_rank=B1_node.virtual_data_parallel_rank, commit=False)
            # Add next F0B1 recv comm to this F1B0 post_backward
            # When this is the last F0B1F1B0 node, it's too complex to add next comm so that overlap. 
            # As a consequence, the last node will not overlap the comm, as seen that wgrad_compute_helper's wait is False, meaning that comm just are commited but not start.
            self._forward_backward_with_comm(F1_node.virtual_data_parallel_rank, B0_node.virtual_data_parallel_rank)
            self._wgrad_compute_and_p2p_overlap(vdp_rank=B0_node.virtual_data_parallel_rank, commit=False)
        # Step 5: n partial F0B1F1B0
        logger.debug(f"[rank{self.rank}]: Step 5")
        partial_F0B1F1B0_nodes = self.schedule["Partial_F0B1F1B0"]
        for i in range(0, len(partial_F0B1F1B0_nodes), 2):
            F_node = partial_F0B1F1B0_nodes[i]
            B_node = partial_F0B1F1B0_nodes[i + 1]
            recv = (i == 0)
            if F_node.is_idle():
                self._backward_with_comm(B_node.virtual_data_parallel_rank, recv=recv)
            else:
                self._forward_backward_with_comm(F_node.virtual_data_parallel_rank, B_node.virtual_data_parallel_rank, recv_0=recv, recv_1=recv)
            if i < len(partial_F0B1F1B0_nodes) - 2:
                next_F_node = partial_F0B1F1B0_nodes[i + 2]
                next_B_node = partial_F0B1F1B0_nodes[i + 3]
                if not next_F_node.is_idle():
                    self._recv_forward(next_F_node.virtual_data_parallel_rank)
                self._recv_backward(next_B_node.virtual_data_parallel_rank)
                commit = True
            else:
                commit = False
            self._wgrad_compute_and_p2p_overlap(B_node.virtual_data_parallel_rank, commit=commit)
        
        # Step 7: n B1 or B0
        logger.debug(f"[rank{self.rank}]: Step 6")
        single_B_nodes = self.schedule["B1B0"] + self.schedule["Single_B"]
        for i in range(0, len(single_B_nodes)):
            node = single_B_nodes[i]
            recv = (i == 0)
            self._backward_with_comm(node.virtual_data_parallel_rank, recv=recv)
            if i < len(single_B_nodes) - 1:
                next_B_node = single_B_nodes[i + 1]
                self._recv_backward(next_B_node.virtual_data_parallel_rank)
            self._wgrad_compute_and_p2p_overlap(node.virtual_data_parallel_rank)
        logger.debug(f"[rank{self.rank}]: Sync")
        self.comm_mgr.commit()
        self._finish_sync()
        logger.debug(f"[rank{self.rank}]: pipeline finish")
        # Set to the last pipeline stage needed virtual data parallel rank.
        # Only the stage that maybe be the last stage will be set.
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
        return self.forward_data_store

    def __call__(self, *args, **kwds):
        if self.first_run:
            self.prepare(*args, **kwds)
        else:
            self.reset()
        return self.run()