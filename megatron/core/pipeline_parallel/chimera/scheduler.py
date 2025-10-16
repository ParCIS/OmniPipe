import contextlib
from functools import partial
from megatron.core import parallel_state
from typing import Callable, Iterator, List, Optional, Tuple, Union
import torch
from megatron.core.utils import (
    get_model_config,
    get_model_type
)
from megatron.core.tensor_parallel.weight_grad_store import WeightGradStore
import torch.distributed as dist
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
            num_iteration=self.num_microbatches // self.virtual_data_parallel_world_size,
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
        self.enable_decouple_dw = False
    
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
        ## TODO: param sync

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
    
    def _backward_step_helper_preprocess(self, vdp_rank, enable_decouple_dw: bool = False):
        """Preprocess for backward_step_helper"""
        # launch grad synchronization (default)
        # if self.config.grad_sync_func is None and self._is_last_microbatch(
        #     vdp_rank
        # ):
        #     ## TODO: async allreduce 
        #     # self.enable_grad_sync()
        #     self.synchronized_model_chunks.add(vdp_rank)

        parallel_state.set_virtual_data_parallel_rank(vdp_rank)
        self.enable_decouple_dw = enable_decouple_dw
        WeightGradStore.set_enabled(enable_decouple_dw)
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

    def _backward_step_helper_postprocess(self, vdp_rank, input_tensor_grad):
        """Postprocess for backward_step_helper"""
        self.current_b_microbatch_id[vdp_rank] += 1
        self.input_tensor_grads[vdp_rank].append(input_tensor_grad)
        if self.enable_decouple_dw:
            self._weight_backward()
            self.enable_decouple_dw = False
            WeightGradStore.set_enabled(self.enable_decouple_dw)
        if self._is_last_microbatch(vdp_rank):
            self.model[vdp_rank].start_grad_sync()
            self.synchronized_model_chunks.add(vdp_rank)
    
    def _combined_1f1b_schedule_for_chimera(self, f_vdp_rank: Optional[int] = None, b_vdp_rank: Optional[int] = None):
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
                cond=(self.current_f_microbatch_id[f_microbatch_id] == 0) if f_vdp_rank is not None else None
            ),
            current_microbatch=f_microbatch_id
        )
        # forward post process
        if f_vdp_rank is not None:
            self._forward_step_helper_postprocess(f_vdp_rank, output_tensor, num_tokens)
        # backward post process
        if b_vdp_rank:
            # The same as the backward_step_helper
            self._backward_step_helper_postprocess(b_vdp_rank, input_tensor_grad)
            if input_tensor is not None:
                assert input_tensor_grad is not None
        

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

    def _backward_step_helper(self, vdp_rank: int, enable_decouple_dw: bool = False) -> None:
        from megatron.core.pipeline_parallel.schedules import backward_step
        if self.forward_only:
            return
        input_tensor, output_tensor, output_tensor_grad = self._backward_step_helper_preprocess(vdp_rank, enable_decouple_dw)
        input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad, self.model_type, self.config)
        self._backward_step_helper_postprocess(vdp_rank, input_tensor_grad)

    def _forward(self, vdp_rank: int, recv: bool=True, send: bool=True) -> None:
        parallel_state.set_virtual_data_parallel_rank(vdp_rank)
        if recv:
            self._recv_forward(vdp_rank)
        self.comm_mgr.commit_and_wait()
        self._forward_step_helper(vdp_rank)

        if send:
            self._send_forward(vdp_rank)

    def _backward(self, vdp_rank: int, recv: bool=True, send: bool=True, enable_decouple_dw: bool = False) -> None:
        parallel_state.set_virtual_data_parallel_rank(vdp_rank)
        if recv:
            self._recv_backward(vdp_rank)
        self.comm_mgr.commit_and_wait()
        self._backward_step_helper(vdp_rank, enable_decouple_dw)
        if send:
            self._send_backward(vdp_rank)
    
    def _forward_backward(self, vdp_rank_0: int, vdp_rank_1: int, recv_0: bool = True, enable_decouple_dw=False):
        """
            vdp_rank_0: F
            vdp_rank_1: B
        """
        if recv_0:
            self._recv_forward(vdp_rank_0)
        self._recv_backward(vdp_rank_1)
        self.comm_mgr.commit_and_wait()
        if self.config.overlap_moe_expert_parallel_comm and not self.forward_only:
            self._combined_1f1b_schedule_for_chimera(vdp_rank_0, vdp_rank_1)
        else:
            self._forward_step_helper(vdp_rank_0)
            self._backward_step_helper(vdp_rank_1, enable_decouple_dw)

        self._send_forward(vdp_rank_0)
        self._send_backward(vdp_rank_1)

    def _weight_backward(self) -> None:
        """Weight backward and commit communication, if doesn't enable decouple dw, only commit communication"""
        if self.forward_only:
            return
        # Assume FIFO
        self.comm_mgr.commit_and_wait()
        WeightGradStore.pop()

    def _recv_forward(self, vdp_rank: int) -> None:
        parallel_state.set_virtual_data_parallel_rank(vdp_rank)
        is_first_stage = parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vdp_rank)
        if is_first_stage:
            return

        self.current_recv_f_microbatch_id[vdp_rank] += 1
        tensors = self.comm_mgr.recv_forward()
        self.input_tensors[vdp_rank].append(tensors)

    def _send_forward(self, vdp_rank: int) -> None:
        parallel_state.set_virtual_data_parallel_rank(vdp_rank)
        is_last_stage = parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vdp_rank)
        if is_last_stage:
            return

        microbatch_id = self.current_send_f_microbatch_id[vdp_rank]
        self.current_send_f_microbatch_id[vdp_rank] += 1
        tensors = self.output_tensors[vdp_rank][microbatch_id]
        self.comm_mgr.send_forward(tensors)

    def _recv_backward(self, vdp_rank: int) -> None:
        if self.forward_only:
            return
        parallel_state.set_virtual_data_parallel_rank(vdp_rank)
        is_last_stage = parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vdp_rank)
        if is_last_stage:
            return

        self.current_recv_b_microbatch_id[vdp_rank] += 1
        tensors = self.comm_mgr.recv_backward()
        self.output_tensor_grads[vdp_rank].append(tensors)

    def _send_backward(self, vdp_rank: int) -> None:
        if self.forward_only:
            return
        parallel_state.set_virtual_data_parallel_rank(vdp_rank)
        is_first_stage = parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vdp_rank)
        if is_first_stage:
            return

        microbatch_id = self.current_send_b_microbatch_id[vdp_rank]
        self.current_send_b_microbatch_id[vdp_rank] += 1
        tensors = self.input_tensor_grads[vdp_rank][microbatch_id]
        self.input_tensor_grads[vdp_rank][microbatch_id] = None
        self.comm_mgr.send_backward(tensors)
    
    def _is_last_microbatch(self, vdp_rank):
        microbatch_id = self.current_b_microbatch_id[vdp_rank]
        return microbatch_id == self.num_microbatches // self.virtual_data_parallel_world_size
    
    def _start_sync(self, vdp_rank):
        if self._is_last_microbatch(vdp_rank):
            self.model[vdp_rank].start_grad_sync()
            self.synchronized_model_chunks.add(vdp_rank)
    
    def _finish_sync(self):
        if len(self.synchronized_model_chunks) != len(self.model):
            raise RuntimeError("Number of synchronized model is not equal to the models")
        for model in self.model:
            self.config.finalize_model_grads_func(
                [model], self.total_num_tokens if self.config.calculate_per_token_loss else None
            )

    def run(self):
        """
        Because the communication of multiple pipeline is complex, 
        reader can refence to the dualpipe
        """
        self.disable_grad_sync()
        is_middle_rank = ChimeraNode.is_middle_rank
        # Step 1: nF0
        F0_nodes = self.schedule["F0"]
        for node in F0_nodes:
            vdp_rank = node.virtual_data_parallel_rank
            self._forward(vdp_rank)
        # Step 2: nF0F1
        F0F1_nodes = self.schedule["F0F1"]
        self._recv_forward(F0F1_nodes[0].virtual_data_parallel_rank)    # dualpipe: recv_forward(0)
        for i in range(0, len(F0F1_nodes), 2):
            node0 = F0F1_nodes[i]
            node1 = F0F1_nodes[i + 1]
            vdp_rank0 = node0.virtual_data_parallel_rank
            self._forward(vdp_rank0, recv=False, send=is_middle_rank)

            # dualpipe recv_forward(0)
            if i < len(F0F1_nodes) - 2:
                self._recv_forward(F0F1_nodes[i + 2].virtual_data_parallel_rank)
            else:
                # recv next down pipeline
                self._recv_forward(self.schedule["F0B1F1B0"][0].virtual_data_parallel_rank)

            vdp_rank1 = node1.virtual_data_parallel_rank
            self._forward(vdp_rank1, send=(not is_middle_rank) or (i < len(F0F1_nodes) - 2))
            if not is_middle_rank:
                self._send_forward(vdp_rank0)
        
        # Step 3: nB1W1F1
        B1F1_nodes = self.schedule["B1F1"]
        for i in range(0, len(B1F1_nodes), 2):
            node0 = B1F1_nodes[i]   # B1
            node1 = B1F1_nodes[i + 1]   # F1
            self._backward(node0.virtual_data_parallel_rank, enable_decouple_dw=True)
            self._recv_forward(node1.virtual_data_parallel_rank)
            self._weight_backward()
            self._forward(node1.virtual_data_parallel_rank, recv=False)
        

        # Step 4 : nF0B1F1B0
        F0B1F1F0_nodes = self.schedule["F0B1F1B0"]
        for i in range(0, len(F0B1F1F0_nodes), 4):
            F0_node = F0B1F1F0_nodes[i]
            B1_node = F0B1F1F0_nodes[i + 1]
            F1_node = F0B1F1F0_nodes[i + 2]
            B0_node = F0B1F1F0_nodes[i + 3]

            if i == 0:
                if is_middle_rank:
                    self._forward(F0_node.virtual_data_parallel_rank, recv=False, send=False)
                    self._send_forward(F0F1_nodes[-1].virtual_data_parallel_rank)
                    self._backward(B1_node.virtual_data_parallel_rank, send=False)
                    self._send_forward(F0_node.virtual_data_parallel_rank)
                    self._send_backward(B1_node.virtual_data_parallel_rank)
                else:    
                    self._forward_backward(F0_node.virtual_data_parallel_rank, B1_node.virtual_data_parallel_rank, recv_0=False)
            else:
                self._forward_backward(F0_node.virtual_data_parallel_rank, B1_node.virtual_data_parallel_rank)

            self._forward_backward(F1_node.virtual_data_parallel_rank, B0_node.virtual_data_parallel_rank)
        
        # Step 5: n partial F0B1F1B0
        partial_F0B1F1B0_nodes = self.schedule["Partial_F0B1F1B0"]
        for i in range(0, len(partial_F0B1F1B0_nodes), 4):
            F0_node = partial_F0B1F1B0_nodes[i]
            B1_node = partial_F0B1F1B0_nodes[i + 1]
            F1_node = partial_F0B1F1B0_nodes[i + 2]
            B0_node = partial_F0B1F1B0_nodes[i + 3]
            B1_enable_decouple_dw = not (is_middle_rank or F0_node.is_idle())
            if F0_node.is_idle():
                self._backward(B1_node.virtual_data_parallel_rank, enable_decouple_dw=B1_enable_decouple_dw)
            else:
                self._forward_backward(F0_node.virtual_data_parallel_rank, B1_node.virtual_data_parallel_rank, enable_decouple_dw=B1_enable_decouple_dw)
            if B1_enable_decouple_dw:
                if not F1_node.is_idle():
                    self._recv_forward(F1_node.virtual_data_parallel_rank)
                else:
                    self._recv_backward(B0_node.virtual_data_parallel_rank)
                self._weight_backward()
            if F1_node.is_idle():
                self._backward(B0_node.virtual_data_parallel_rank, recv=not B1_enable_decouple_dw)
            else:
                self._forward_backward(F1_node.virtual_data_parallel_rank, B0_node.virtual_data_parallel_rank, recv_0=not B1_enable_decouple_dw)


        
        # Step 6: n B1B0
        B1B0_nodes = self.schedule["B1B0"]
        for i in range(0, len(B1B0_nodes), 2):
            # backward B1, enable decouple dw in the not middle rank
            B1_node = B1B0_nodes[i]
            B0_node = B1B0_nodes[i + 1]
            B1_enable_decouple_dw = not is_middle_rank
            self._backward(B1_node.virtual_data_parallel_rank, enable_decouple_dw=B1_enable_decouple_dw)
            self._recv_backward(B0_node.virtual_data_parallel_rank)
            if B1_enable_decouple_dw:
                self._weight_backward()
            self._start_sync(B1_node.virtual_data_parallel_rank)
            # backward B0
            B0_enable_decouple_dw = i == len(B1B0_nodes) - 2
            self._backward(B0_node.virtual_data_parallel_rank, recv=False, enable_decouple_dw=B0_enable_decouple_dw)
            if B0_enable_decouple_dw:
                self._weight_backward()
            self._start_sync(B0_node.virtual_data_parallel_rank)
        
        # Step 7: n B1 or B0
        single_B_nodes = self.schedule["Single_B"]
        if len(single_B_nodes) > 0:
            self._recv_backward(single_B_nodes[0].virtual_data_parallel_rank)
        for i in range(0, len(single_B_nodes)):
            node = single_B_nodes[i]
            self._backward(node.virtual_data_parallel_rank, enable_decouple_dw=True, recv=False)
            # if node is not the last one, recv next one and send this one
            if i < len(single_B_nodes) - 1:
                self._recv_backward(single_B_nodes[i + 1].virtual_data_parallel_rank)
            self._weight_backward()
            self._start_sync(node.virtual_data_parallel_rank)
        # pipeline finish
        self.comm_mgr.commit_and_wait()
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
        return self.forward_data_store

    def __call__(self, *args, **kwds):
        if self.first_run:
            self.prepare(*args, **kwds)
        else:
            self.reset()
        return self.run()