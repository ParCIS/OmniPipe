import torch
from megatron.core import parallel_state
import torch.distributed as dist
from .scheduler import ChimeraScheduler


def update_global_config_for_chimera(args):
    ep = args.expert_model_parallel_size
    vdp = args.virtual_data_parallel_size
    args.delay_wgrad_compute = True
    world_size = dist.get_world_size()
    device_count = torch.cuda.device_count()
    if ep > device_count or world_size <= device_count:
        extend_vdp_to_ep = True
    else:
        extend_vdp_to_ep = False
    if args.num_experts is not None and extend_vdp_to_ep:
        args.expert_model_parallel_size = ep * vdp
    from megatron.core.models.common.model_chunk_schedule_plan import TransformerModelChunkSchedulePlan
    TransformerModelChunkSchedulePlan.run = transformer_model_chunk_schedule_plan_run_patch
    # 启动调试器（仅在需要调试时启用，可通过环境变量控制）
    if os.environ.get("ENABLE_DEBUG", "False").lower() == "true":
        init_debugger()


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
    
