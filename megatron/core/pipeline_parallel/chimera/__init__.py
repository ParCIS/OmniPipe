from datetime import timedelta
import torch
from megatron.core import parallel_state
from megatron.core.parallel_state import RankGenerator, create_group, get_nccl_options
import torch.distributed as dist
from megatron.training import get_args
from .scheduler import ChimeraScheduler

def generator_wrapper(
        group_type, 
        is_expert=False, 
        decoder_rank_generator: RankGenerator=None, 
        expert_decoder_rank_generator: RankGenerator=None, 
        **kwargs
    ):
        """Copy from megatron.core.parallel_state.initialize_model_parallel.generator_wrapper"""
        if is_expert:
            d_ranks = expert_decoder_rank_generator.get_ranks(group_type, **kwargs)
        else:
            d_ranks = decoder_rank_generator.get_ranks(group_type, **kwargs)
        for x in d_ranks:
            yield x
        return


def update_global_config_for_chimera(args):
    ep = args.expert_model_parallel_size
    vdp = args.virtual_data_parallel_size
    args.delay_wgrad_compute = True
    if args.num_experts is not None:
        args.expert_model_parallel_size = ep * vdp
    from megatron.core.models.common.model_chunk_schedule_plan import TransformerModelChunkSchedulePlan
    TransformerModelChunkSchedulePlan.run = transformer_model_chunk_schedule_plan_run_patch
    

def init_chimera_parallel_state():
    """
    Initialize Chimera parallel groups.
    Note that some communication of M-core doesn't use yet, so we don't patch them here
    """
    # 启动调试器（仅在需要调试时启用，可通过环境变量控制）
    if os.environ.get("ENABLE_DEBUG", "False").lower() == "true":
        init_debugger()
    args = get_args()
    pp = args.pipeline_model_parallel_size
    tp = args.tensor_model_parallel_size
    etp = args.expert_tensor_parallel_size
    if etp is None:
        etp = tp
    cp = args.context_parallel_size
    ep = args.expert_model_parallel_size
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    dp = world_size // (pp * tp)
    edp = world_size // (pp * etp * ep)
    vdp = args.virtual_data_parallel_size

    num_devices_per_stage = world_size // pp
    num_dp_group_per_stage = num_devices_per_stage // dp
    num_dp_cp_group_per_stage = num_devices_per_stage // (dp * cp)
    num_ep_group_per_stage = num_devices_per_stage // ep
    num_ep_tp_group_per_stage = num_devices_per_stage // (ep * tp)
    num_edp_group_per_stage = num_devices_per_stage // edp

    dp_ranks_list = []
    dp_cp_ranks_list = []
    ep_ranks_list = []
    ep_tp_ranks_list = []
    edp_ranks_list = []

    decoder_rank_generator = RankGenerator(
        tp=tp,
        ep=1,
        dp=dp,
        pp=pp,
        cp=cp,
        order="tp-cp-ep-dp-pp",
        rank_offset=0,
    )

    expert_decoder_rank_generator = RankGenerator(
        tp=etp,
        ep=ep,
        dp=edp,
        pp=pp,
        cp=1,
        order="tp-cp-ep-dp-pp",
        rank_offset=0,
    )

    for ranks_with_cp in generator_wrapper('dp-cp', decoder_rank_generator=decoder_rank_generator, expert_decoder_rank_generator=expert_decoder_rank_generator):
        dp_cp_ranks_list.append(ranks_with_cp)
    for ranks in generator_wrapper('dp', decoder_rank_generator=decoder_rank_generator, expert_decoder_rank_generator=expert_decoder_rank_generator):
        dp_ranks_list.append(ranks)
    for ranks in generator_wrapper('ep', is_expert=True, decoder_rank_generator=decoder_rank_generator, expert_decoder_rank_generator=expert_decoder_rank_generator):
        ep_ranks_list.append(ranks)
    for ranks in generator_wrapper('tp-ep', is_expert=True, decoder_rank_generator=decoder_rank_generator, expert_decoder_rank_generator=expert_decoder_rank_generator):
        ep_tp_ranks_list.append(ranks)
    for ranks in generator_wrapper('dp', is_expert=True, decoder_rank_generator=decoder_rank_generator, expert_decoder_rank_generator=expert_decoder_rank_generator):
        edp_ranks_list.append(ranks)

    nccl_communicator_config_path = args.nccl_communicator_config_path
    nccl_comm_cfgs = {}
    if nccl_communicator_config_path is not None:
        try:
            import yaml
        except ImportError:
            raise RuntimeError(
                "Cannot import `yaml`. Setting custom nccl communicator configs "
                "requires the yaml package."
            )

        with open(nccl_communicator_config_path, "r") as stream:
            nccl_comm_cfgs = yaml.safe_load(stream)

    offset_unit = pp // vdp * 2
    timeout = timedelta(minutes=args.distributed_timeout_minutes)
    for pp_rank in range(pp // vdp):
        dual_pp_rank = pp - 1 - pp_rank
        pp_ranks = [i for i in range(pp_rank, pp, offset_unit)] + [i for i in range(dual_pp_rank, -1, -offset_unit)]
        if global_rank in pp_ranks:
            parallel_state._LOCAL_VIRTUAL_DATA_PARALLEL_RANK_TO_PIPELINE_MODEL_PARALLEL_RANK_MAP = pp_ranks
            print(f">>> [rank{global_rank}]: Chimera local virtual data parallel rank to pipeline rank map = {pp_ranks}")

        # constructe new dp groups
        for i in range(num_dp_group_per_stage):
            new_dp_ranks = []
            for pp_r in pp_ranks:
                new_dp_ranks.extend(dp_ranks_list[pp_r * num_dp_group_per_stage + i])
            new_dp_group = create_group(new_dp_ranks, timeout=timeout, pg_options=get_nccl_options("dp", nccl_comm_cfgs))
            if global_rank in new_dp_ranks:
                parallel_state._DATA_PARALLEL_GLOBAL_RANKS = new_dp_ranks
                parallel_state._DATA_PARALLEL_GROUP = new_dp_group
        
        # construct new dp-cp groups
        for i in range(num_dp_cp_group_per_stage):
            new_dp_cp_ranks = []
            for pp_r in pp_ranks:
                new_dp_cp_ranks.extend(dp_cp_ranks_list[pp_r * num_dp_cp_group_per_stage + i])
            new_dp_cp_group = create_group(new_dp_cp_ranks, timeout=timeout, pg_options=get_nccl_options("dp-cp", nccl_comm_cfgs))
            if global_rank in new_dp_cp_ranks:
                parallel_state._DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = new_dp_cp_ranks
                parallel_state._DATA_PARALLEL_GROUP_WITH_CP = new_dp_cp_group
                # no use, just to be compatible with mcore
                parallel_state._INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = new_dp_cp_group
        if args.extend_vdp_to_ep:
            # construct new ep groups
            for i in range(num_ep_group_per_stage):
                new_ep_ranks = []
                for pp_r in pp_ranks:
                    new_ep_ranks.extend(ep_ranks_list[pp_r * num_ep_group_per_stage + i])
                new_ep_group = create_group(new_ep_ranks, timeout=timeout, pg_options=get_nccl_options("ep", nccl_comm_cfgs))
                if global_rank in new_ep_ranks:
                    parallel_state._EXPERT_MODEL_PARALLEL_GROUP = new_ep_group
            # construct new ep-tp groups
            for i in range(num_ep_tp_group_per_stage):
                new_ep_tp_ranks = []
                for pp_r in pp_ranks:
                    new_ep_tp_ranks.extend(ep_tp_ranks_list[pp_r * num_ep_tp_group_per_stage + i])
                new_ep_tp_group = create_group(new_ep_tp_ranks, timeout=timeout, pg_options=get_nccl_options("ep-tp-mp", nccl_comm_cfgs))
                if global_rank in new_ep_tp_ranks:
                    parallel_state._EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = new_ep_tp_group
            args.expert_model_parallel_size = ep * vdp
        else:
            # construct new ep-dp groups
            for i in range(num_edp_group_per_stage):
                new_ep_dp_ranks = []
                for pp_r in pp_ranks:
                    new_ep_dp_ranks.extend(edp_ranks_list[pp_r * num_edp_group_per_stage + i])
                new_ep_dp_group = create_group(new_ep_dp_ranks, timeout=timeout, pg_options=get_nccl_options("ep-dp", nccl_comm_cfgs))
                if global_rank in new_ep_dp_ranks:
                    parallel_state._EXPERT_DATA_PARALLEL_GROUP = new_ep_dp_group
        
    # construct embedding groups
    for vdp_rank in range(vdp // 2):
        first_stage_rank = offset_unit * vdp_rank
        last_stage_rank = pp - 1 - offset_unit * vdp_rank
        for rank in range(num_devices_per_stage):
            global_first_stage_rank = first_stage_rank * num_devices_per_stage + rank
            global_last_stage_rank = last_stage_rank * num_devices_per_stage + rank
            embedding_ranks = [global_first_stage_rank, global_last_stage_rank]
            embedding_group = create_group(embedding_ranks, timeout=timeout, pg_options=get_nccl_options("embedding", nccl_comm_cfgs))
            if global_rank in embedding_ranks:
                parallel_state._EMBEDDING_GROUP = embedding_group
                parallel_state._EMBEDDING_GLOBAL_RANKS = embedding_ranks
    parallel_state._VIRTUAL_DATA_PARALLEL_SIZE = vdp
    args.delay_wgrad_compute = True
    if global_rank == 0:
        print(f">>> Set expert model parallel size to {ep * vdp}")

    from megatron.core.models.common.model_chunk_schedule_plan import TransformerModelChunkSchedulePlan
    TransformerModelChunkSchedulePlan.run = transformer_model_chunk_schedule_plan_run_patch


def transformer_layer_schedule_plan_run_patch(f_layer, b_layer, f_input=None, b_grad=None):
    """
    Chimera is unlike to interleaved 1f1b. We delay the all backward_dw to get less bubble.
    """
    if b_layer is not None:
            b_grad = b_layer.mtp_post_process.backward(b_grad)
            b_grad = b_layer.moe_combine.backward(b_grad)

    if f_layer is not None:
        with f_layer.get_fp8_context():
            f_input = f_layer.attn.forward(f_input)
            f_input = f_layer.post_attn.forward(f_input)


    if f_layer is not None:
        with f_layer.get_fp8_context():
            f_input = f_layer.moe_dispatch.forward(f_input)

    if b_layer is not None:
        b_grad = b_layer.mlp.backward(b_grad)

    if b_layer is not None:
        b_grad = b_layer.moe_dispatch.backward(b_grad)

    if f_layer is not None:
        with f_layer.get_fp8_context():
            f_input = f_layer.mlp.forward(f_input)

    if f_layer is not None:
        with f_layer.get_fp8_context():
            f_input = f_layer.moe_combine.forward(f_input)
            f_input = f_layer.mtp_post_process.forward(f_input)

    if b_layer is not None:
        b_grad = b_layer.post_attn.backward(b_grad)
        b_grad = b_layer.attn.backward(b_grad)

    return f_input, b_grad


def transformer_model_chunk_schedule_plan_run_patch(f_schedule_plan, b_schedule_plan, b_grad=None, *unused_args, **unused_kwargs):
    from megatron.core.pipeline_parallel.utils import get_comm_stream
    f_input = None
    if f_schedule_plan:
        f_schedule_plan.record_current_stream()
        f_input = f_schedule_plan.pre_process.forward()

    if b_schedule_plan:
        b_schedule_plan.record_current_stream()
        assert b_grad is not None

        if b_schedule_plan.post_process is not None:
            b_grad = b_schedule_plan.post_process.backward(b_grad)

    f_num_layers = f_schedule_plan.num_layers() if f_schedule_plan is not None else 0
    b_num_layers = b_schedule_plan.num_layers() if b_schedule_plan is not None else 0
    overlapped_layers = min(f_num_layers, b_num_layers)

    # combined forward and backward pass for overlapped layers
    for i in range(overlapped_layers):
        f_layer = f_schedule_plan.get_layer(i)
        b_layer = b_schedule_plan.get_layer(b_num_layers - 1 - i)
        torch.cuda.nvtx.range_push(f"layer_{i}f-layer_{b_num_layers - 1 - i}b")
        f_input, b_grad = transformer_layer_schedule_plan_run_patch(
            f_layer,
            b_layer,
            f_input=f_input,
            b_grad=b_grad,
        )
        torch.cuda.nvtx.range_pop()

    # backward pass for the remaining layers
    for i in range(overlapped_layers, b_num_layers):
        b_layer = b_schedule_plan.get_layer(b_num_layers - 1 - i)
        torch.cuda.nvtx.range_push(f"layer_{b_num_layers - 1 - i}b")
        _, b_grad = transformer_layer_schedule_plan_run_patch(
            None, b_layer, b_grad=b_grad
        )
        torch.cuda.nvtx.range_pop()

    # forward pass for the remaining layers
    for i in range(overlapped_layers, f_num_layers):
        f_layer = f_schedule_plan.get_layer(i)
        torch.cuda.nvtx.range_push(f"layer_{i}f")
        f_input, _ = transformer_layer_schedule_plan_run_patch(f_layer, None, f_input=f_input)
        torch.cuda.nvtx.range_pop()
    # post process forward
    if f_schedule_plan is not None and f_schedule_plan.post_process is not None:
        f_input = f_schedule_plan.post_process.forward(f_input)

    # pre process backward
    if b_schedule_plan is not None:
        b_schedule_plan.pre_process.backward(b_grad)

    if f_schedule_plan:
        f_schedule_plan.wait_current_stream()

    if b_schedule_plan:
        b_schedule_plan.wait_current_stream()


    # Release reference as early as possible, this helps avoid memory leak.
    if b_schedule_plan is not None:
        b_schedule_plan.release_state()

    return f_input

CHIMERA_SCHEDULER = None
def get_chimera_forward_backward_func():
    global CHIMERA_SCHEDULER
    if CHIMERA_SCHEDULER is None:
        CHIMERA_SCHEDULER = ChimeraScheduler()
    return CHIMERA_SCHEDULER


__all__ = ["update_global_config_for_chimera", "get_chimera_forward_backward_func"]

# pretrain_gpt.py 头部添加
import os
import debugpy

# 调试配置：基础端口（可自定义，避免与其他程序冲突）
DEBUG_BASE_PORT = 5678
# 等待调试器连接的超时时间（秒，0 表示无限等待）
DEBUG_WAIT_TIMEOUT = 0
INIT_DEBUG = False

def init_debugger():
    global INIT_DEBUG
    if INIT_DEBUG:
        return
    INIT_DEBUG = True
    rank = dist.get_rank()  # 获取当前进程的 rank
    debug_port = DEBUG_BASE_PORT + rank  # 每个进程分配独立端口
    
    # 启动调试监听（允许外部连接）
    debugpy.listen(("0.0.0.0", debug_port))
    print(f"[Rank {rank}] 调试器已启动，等待连接 -> 端口: {debug_port}")
    
    # 等待调试器连接（超时后自动继续，0 表示无限等待）
    if DEBUG_WAIT_TIMEOUT == 0:
        debugpy.wait_for_client()
        print(f"[Rank {rank}] 调试器已连接！")
    else:
        debugpy.wait_for_client(timeout=DEBUG_WAIT_TIMEOUT)
    
