from typing import Callable, Iterator, Optional, Union, List, Dict
from dataclasses import dataclass
from enum import Enum


class ChimeraNodeType(Enum):
    IDLE = 0
    BACKWARD = 1
    FORWARD = 2


class ChimeraNode:

    forward_microbatch_count: Dict[int, int] = {}
    backward_microbatch_count: Dict[int, int] = {}
    virtual_data_parallel_size: int = 0
    pipeline_model_parallel_world_size: int = 0
    pipeline_rank: int = 0
    is_dual_rank: bool = False
    is_middle_rank: bool = False

    def __init__(self, type: ChimeraNodeType,  virtual_data_parallel_rank: int = 0):
        self.type = type
        
        # calculate the virtual data parallel rank mapping of each pipeline rank
        self_vdp_rank = ChimeraNode.pipeline_rank // (ChimeraNode.pipeline_model_parallel_world_size // (ChimeraNode.virtual_data_parallel_size // 2))

        base = 0
        dual_base = ChimeraNode.virtual_data_parallel_size // 2
        if ChimeraNode.is_dual_rank:
            self_vdp_rank = ChimeraNode.virtual_data_parallel_size - 1 - self_vdp_rank
            base = ChimeraNode.virtual_data_parallel_size // 2
            dual_base = 0
        if virtual_data_parallel_rank < ChimeraNode.virtual_data_parallel_size // 2:
            virtual_data_parallel_rank = (self_vdp_rank + virtual_data_parallel_rank) % (ChimeraNode.virtual_data_parallel_size // 2) + base
        else:
            if self_vdp_rank == 0:
                self_dual_vdp_rank = ChimeraNode.virtual_data_parallel_size // 2
            elif self_vdp_rank == ChimeraNode.virtual_data_parallel_size // 2:
                self_dual_vdp_rank = 0
            else:
                self_dual_vdp_rank = ChimeraNode.virtual_data_parallel_size - self_vdp_rank
            virtual_data_parallel_rank = (self_dual_vdp_rank + virtual_data_parallel_rank) % (ChimeraNode.virtual_data_parallel_size // 2) + dual_base
        
        self.virtual_data_parallel_rank = virtual_data_parallel_rank
        
        # calculate microbatch index of each virtual data parallel rank
        if self.type == ChimeraNodeType.FORWARD:
            if virtual_data_parallel_rank not in ChimeraNode.forward_microbatch_count:
                ChimeraNode.forward_microbatch_count[virtual_data_parallel_rank] = 0

            self.microbatch = ChimeraNode.forward_microbatch_count[virtual_data_parallel_rank]
            ChimeraNode.forward_microbatch_count[virtual_data_parallel_rank] += 1
        elif self.type == ChimeraNodeType.BACKWARD:
            if virtual_data_parallel_rank not in ChimeraNode.backward_microbatch_count:
                ChimeraNode.backward_microbatch_count[virtual_data_parallel_rank] = 0

            self.microbatch = ChimeraNode.backward_microbatch_count[virtual_data_parallel_rank]
            ChimeraNode.backward_microbatch_count[virtual_data_parallel_rank] += 1

    def __repr__(self):
        if self.is_idle():
            return ""
        if self.is_forward():
            type_str = "forward"
        else:
            type_str = "backward"
        return f"ChimeraNode({type_str}, vdp_rank={self.virtual_data_parallel_rank}, microbatch={self.microbatch})"
    
    def is_idle(self):
        return self.type == ChimeraNodeType.IDLE
    
    def is_forward(self):
        return self.type == ChimeraNodeType.FORWARD
    
    def is_backward(self):
        return self.type == ChimeraNodeType.BACKWARD

class ChimeraAutoSchedule:
    def __init__(
            self,
            rank: int = 0,
            world_size: int = 1,
            pipeline_model_parallel_world_size: int = None, 
            virtual_data_parallel_size: int = None, 
            num_microbatches: int = None
        ):
        self.pipeline_model_parallel_world_size: int = pipeline_model_parallel_world_size
        # chimera config
        self.schedule_graph: Dict[str, List[ChimeraNode]] = {}
        self.relative_rank: int = 0
        self.virtual_data_parallel_rank: int = 0
        self.virtual_data_parallel_size: int = virtual_data_parallel_size
        self.num_microbatches: int = num_microbatches

        # chimera node config
        pipeline_rank = rank // (world_size // pipeline_model_parallel_world_size)
        ChimeraNode.virtual_data_parallel_size = virtual_data_parallel_size
        ChimeraNode.pipeline_model_parallel_world_size = pipeline_model_parallel_world_size
        ChimeraNode.pipeline_rank = pipeline_rank
        offset_unit = pipeline_model_parallel_world_size // (virtual_data_parallel_size // 2)
        is_dual_rank = False if ((pipeline_rank % offset_unit) < (offset_unit // 2)) else True
        ChimeraNode.is_dual_rank = is_dual_rank
        if is_dual_rank:
            self.relative_rank = (offset_unit - 1 - pipeline_rank % offset_unit)
            # ChimeraNode.is_middle_rank = (self.relative_rank == (pipeline_model_parallel_world_size // virtual_data_parallel_size))
        else:
            self.relative_rank = pipeline_rank % offset_unit
        ChimeraNode.is_middle_rank = (self.relative_rank == (pipeline_model_parallel_world_size // virtual_data_parallel_size - 1))

    def __schedule(self):
        """
        Schedule the execution of the model across the Chimera pipeline.
        Note that this is only current rank's schedule.
        """
        num_microbatches_per_vdp = self.num_microbatches // self.virtual_data_parallel_size
        F_counts = [num_microbatches_per_vdp] * self.virtual_data_parallel_size
        B_counts = [num_microbatches_per_vdp] * self.virtual_data_parallel_size

        def get_dual_vdp_rank(vdp_rank):
            if vdp_rank == 0:
                return self.virtual_data_parallel_size // 2
            elif vdp_rank == self.virtual_data_parallel_size // 2:
                return 0
            else:
                return self.virtual_data_parallel_size - vdp_rank

        def F0():
            nodes = []
            n = self.pipeline_model_parallel_world_size // self.virtual_data_parallel_size - self.relative_rank - 1
            if self.num_microbatches >= 2 * self.pipeline_model_parallel_world_size - self.virtual_data_parallel_size:
                n *= 2
            # maybe there is a case that n > num_microbatches
            # n_F_down_F_up = self.pipeline_model_parallel_world_size // 2 - (self.pipeline_model_parallel_world_size // self.virtual_data_parallel_size - 1 - self.relative_rank)
            # if num_microbatches_per_vdp * 2 - n < n_F_down_F_up:
            #     # reserve microbatch for F_down_F_up
            #     n = num_microbatches_per_vdp * 2 - n_F_down_F_up
            for i in range(n):
                nodes.append(ChimeraNode(ChimeraNodeType.FORWARD, virtual_data_parallel_rank=0))
                F_counts[0] -= 1
            self.schedule_graph["F0"] = nodes
            return n

        def F_down_F_up(n_F: int):
            """
            Args:
                n_F (int): number of F
            """
            nodes = []
            # n = (self.pipeline_model_parallel_world_size - n_F) // 2
            if self.num_microbatches < 2 * self.pipeline_model_parallel_world_size - self.virtual_data_parallel_size:
                n_F = n_F * 2
            n = (self.pipeline_model_parallel_world_size - n_F) // 2

            # n = self.pipeline_model_parallel_world_size // 2 - (self.pipeline_model_parallel_world_size // self.virtual_data_parallel_size - 1 - self.relative_rank)

            micro_per_vdp = self.pipeline_model_parallel_world_size // self.virtual_data_parallel_size
            down_count = micro_per_vdp - self.relative_rank - 1
            up_count = 0
            down_vdp_rank = 0
            up_vdp_rank = self.virtual_data_parallel_size - 1
            for _ in range(n):
                nodes.append(ChimeraNode(ChimeraNodeType.FORWARD, virtual_data_parallel_rank=down_vdp_rank))
                nodes.append(ChimeraNode(ChimeraNodeType.FORWARD, virtual_data_parallel_rank=up_vdp_rank))
                F_counts[down_vdp_rank] -= 1
                F_counts[up_vdp_rank] -= 1

                down_count += 1
                if down_count % micro_per_vdp == 0:
                    down_vdp_rank  = (down_vdp_rank - 1 + self.virtual_data_parallel_size // 2) % (self.virtual_data_parallel_size // 2)
                up_count += 1
                if up_count % micro_per_vdp == 0:
                    up_vdp_rank  = up_vdp_rank - 1
            self.schedule_graph["F0F1"] = nodes
            return n

        def B_up_F_up():
            nodes = []
            n = self.pipeline_model_parallel_world_size // self.virtual_data_parallel_size - 1 - self.relative_rank
            for _ in range(n):
                B_vdp_rank = self.virtual_data_parallel_size // 2
                nodes.append(ChimeraNode(ChimeraNodeType.BACKWARD, virtual_data_parallel_rank=B_vdp_rank))
                nodes.append(ChimeraNode(ChimeraNodeType.FORWARD, virtual_data_parallel_rank=B_vdp_rank))
                F_counts[B_vdp_rank] -= 1
                B_counts[B_vdp_rank] -= 1
            self.schedule_graph["B1F1"] = nodes
            return n
        
        def F_down_B_up_F_up_B_down(num_F: int, num_F_down_F_up: int):
            """
            This stage is the most important stage, the forward-backward-overlapping is here.
            Note that the all forward nodes are finished here, so the number of stage is equal to the (number of microbatches // 2) - (num_F + num_F_down_F_up).
            """
            nodes = []
            micro_per_vdp = self.pipeline_model_parallel_world_size // self.virtual_data_parallel_size

            # Split the microbatch into an integer multiple of the number of pipeline stages and the remaining part.
            reminder = self.num_microbatches % self.pipeline_model_parallel_world_size
            multiple = (self.num_microbatches - reminder) // 2 - num_F - num_F_down_F_up
            vdp0_count = self.relative_rank + 1

            down_vdp_rank = 0
            up_vdp_rank = self.virtual_data_parallel_size - 1
            # because the first down vdp only has vdp0_count F_down_B_up_F_up_B_down, so set the down_count to micro_per_vdp - vdp0_count
            down_count = micro_per_vdp - vdp0_count
            up_count = 0

            n = multiple + reminder // self.virtual_data_parallel_size
            for _ in range(n):
                # TODO
                B_up_vdp_rank = get_dual_vdp_rank(down_vdp_rank)
                B_down_vdp_rank = get_dual_vdp_rank(up_vdp_rank)
                nodes.append(ChimeraNode(ChimeraNodeType.FORWARD, virtual_data_parallel_rank=down_vdp_rank))
                nodes.append(ChimeraNode(ChimeraNodeType.BACKWARD, virtual_data_parallel_rank=B_up_vdp_rank))
                nodes.append(ChimeraNode(ChimeraNodeType.FORWARD, virtual_data_parallel_rank=up_vdp_rank))
                nodes.append(ChimeraNode(ChimeraNodeType.BACKWARD, virtual_data_parallel_rank=B_down_vdp_rank))
                F_counts[down_vdp_rank] -= 1
                F_counts[up_vdp_rank] -= 1
                B_counts[B_up_vdp_rank] -= 1
                B_counts[B_down_vdp_rank] -= 1

                down_count += 1
                up_count += 1

                if down_count % micro_per_vdp == 0:
                    down_vdp_rank  = (down_vdp_rank - 1 + self.virtual_data_parallel_size // 2) % (self.virtual_data_parallel_size // 2)
                if up_count % micro_per_vdp == 0:
                    up_vdp_rank  = up_vdp_rank - 1
                    if up_vdp_rank < self.virtual_data_parallel_size // 2:
                        up_vdp_rank = self.virtual_data_parallel_size - 1
            self.schedule_graph["F0B1F1B0"] = nodes
            return n, down_count, down_vdp_rank, up_count, up_vdp_rank, micro_per_vdp
        
        def partial_F_down_B_up_F_up_B_down(down_count: int, down_vdp_rank: int, up_count: int, up_vdp_rank: int, micro_per_vdp: int):
            """
            When the virtual data parallel size > 2, there are some F_down or F_up in this stage.
            But when the virtual data parallel size == 2, there are no F_down or F_up.
            In order to properly schedule this F, consider it as same to the F_down_B_up_F_up_B_down stage, while some F_down or F_up will be removed accoding to the number of F_down or F_up.

            This stage will schedule the rest of F_up or F_down.
            """
            nodes = []
            # n = self.pipeline_model_parallel_world_size // 2 - 1 - (self.pipeline_model_parallel_world_size // self.virtual_data_parallel_size - 1 - self.relative_rank)
            rest_F = sum(F_counts)
            while rest_F > 0:
                B_up_vdp_rank = get_dual_vdp_rank(down_vdp_rank)
                B_down_vdp_rank = get_dual_vdp_rank(up_vdp_rank)

                # if F_down > 0, insert a F_down
                if F_counts[down_vdp_rank] > 0:
                    nodes.append(ChimeraNode(ChimeraNodeType.FORWARD, virtual_data_parallel_rank=down_vdp_rank))
                    F_counts[down_vdp_rank] -= 1
                    rest_F -= 1
                else:
                    nodes.append(ChimeraNode(ChimeraNodeType.IDLE, virtual_data_parallel_rank=-1))
                # insert B_up
                nodes.append(ChimeraNode(ChimeraNodeType.BACKWARD, virtual_data_parallel_rank=B_up_vdp_rank))
                # if F_up > 0, insert a F_up
                if F_counts[up_vdp_rank] > 0:
                    nodes.append(ChimeraNode(ChimeraNodeType.FORWARD, virtual_data_parallel_rank=up_vdp_rank))
                    F_counts[up_vdp_rank] -= 1
                    rest_F -= 1
                else:
                    nodes.append(ChimeraNode(ChimeraNodeType.IDLE, virtual_data_parallel_rank=-1))
                # insert B_down
                nodes.append(ChimeraNode(ChimeraNodeType.BACKWARD, virtual_data_parallel_rank=B_down_vdp_rank))

                B_counts[B_up_vdp_rank] -= 1
                B_counts[B_down_vdp_rank] -= 1

                down_count += 1
                if down_count % micro_per_vdp == 0:
                    down_vdp_rank = (down_vdp_rank - 1 + self.virtual_data_parallel_size // 2) % (self.virtual_data_parallel_size // 2)
                up_count += 1
                if up_count % micro_per_vdp == 0:
                    up_vdp_rank = up_vdp_rank - 1
                    if up_vdp_rank < self.virtual_data_parallel_size // 2:
                        up_vdp_rank = self.virtual_data_parallel_size - 1
            self.schedule_graph["Partial_F0B1F1B0"] = nodes
            return down_count, down_vdp_rank, up_count, up_vdp_rank, micro_per_vdp

        def partial_B_up_B_down(down_count: int, down_vdp_rank: int, up_count: int, up_vdp_rank: int, micro_per_vdp: int):
            """
            Same to the partial_F_down_B_up_F_up_B_down, while this stage only includes B nodes.
            This stage will schedule the rest of B_up or B_down.
            It's ok to enable decoupling dw to optimize, and if enable, the schedule graph has a bit change.
            There are some single B node at the end of pipeline, they will enable decoupling dw with no harm.
            But at the middle of pipeline, there are B1-B0 dual node, if enable decouple dw always, they will be slower. We suggest enabling at he last one dual node.
            """
            B1B0_nodes = []
            single_B_nodes = []
            rest_B = sum(B_counts)
            while rest_B > 0:
                B_up_vdp_rank = get_dual_vdp_rank(down_vdp_rank)
                B_down_vdp_rank = get_dual_vdp_rank(up_vdp_rank)
                dual_nodes = 0
                nodes = []

                if B_counts[B_up_vdp_rank] > 0:
                    nodes.append(ChimeraNode(ChimeraNodeType.BACKWARD, virtual_data_parallel_rank=B_up_vdp_rank))
                    B_counts[B_up_vdp_rank] -= 1
                    rest_B -= 1
                    dual_nodes += 1
                if B_counts[B_down_vdp_rank] > 0:
                    nodes.append(ChimeraNode(ChimeraNodeType.BACKWARD, virtual_data_parallel_rank=B_down_vdp_rank))
                    B_counts[B_down_vdp_rank] -= 1
                    rest_B -= 1
                    dual_nodes += 1
                if dual_nodes == 2:
                    B1B0_nodes.extend(nodes)
                else:
                    single_B_nodes.extend(nodes)
                down_count += 1
                if down_count % micro_per_vdp == 0:
                    down_vdp_rank = (down_vdp_rank - 1 + self.virtual_data_parallel_size // 2) % (self.virtual_data_parallel_size // 2)

                up_count += 1
                if up_count % micro_per_vdp == 0:
                    up_vdp_rank = up_vdp_rank - 1
                    if up_vdp_rank < self.virtual_data_parallel_size // 2:
                        up_vdp_rank = self.virtual_data_parallel_size - 1
            self.schedule_graph["B1B0"] = B1B0_nodes
            self.schedule_graph["Single_B"] = single_B_nodes


        n_F = F0()
        n_F_down_F_up = F_down_F_up(n_F)
        n_B_up_F_up = B_up_F_up()
        n_F_down_B_up_F_up_B_down, down_count, down_vdp_rank, up_count, up_vdp_rank, micro_per_vdp = F_down_B_up_F_up_B_down(n_F, n_F_down_F_up)
        # n_B_up_F_up_B_down, down_count, down_vdp_rank, up_count, up_vdp_rank, micro_per_vdp = B_up_F_up_B_down(down_count, down_vdp_rank, up_count, up_vdp_rank, micro_per_vdp)
        down_count, down_vdp_rank, up_count, up_vdp_rank, micro_per_vdp = partial_F_down_B_up_F_up_B_down(down_count, down_vdp_rank, up_count, up_vdp_rank, micro_per_vdp)
        partial_B_up_B_down(down_count, down_vdp_rank, up_count, up_vdp_rank, micro_per_vdp)

    def get_schedule(self) -> Dict[str, List[ChimeraNode]]:
        self.__schedule()
        return self.schedule_graph

if __name__ == "__main__":
    world_size = 8
    pipeline_model_parallel_world_size = 8
    virtual_data_parallel_size = 4
    num_microbatches = 12
    for rank in range(world_size):
        print("=" * 150)
        ChimeraNode.forward_microbatch_count = {}
        ChimeraNode.backward_microbatch_count = {}
        scheduler = ChimeraAutoSchedule(rank, world_size, pipeline_model_parallel_world_size, virtual_data_parallel_size, num_microbatches)
        print(f"rank {rank}: {scheduler.get_schedule()}")