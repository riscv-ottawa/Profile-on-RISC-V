from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.components.processors.cpu_types import CPUTypes
from gem5.isas import ISA
from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import (
    PrivateL1PrivateL2CacheHierarchy,
)
from gem5.components.memory.single_channel import SingleChannelDDR3_1600
from gem5.resources.resource import BinaryResource
from gem5.simulate.simulator import Simulator
from gem5.components.boards.simple_board import SimpleBoard
import sys

import m5

from gem5.simulate.exit_event import ExitEvent

def handle_roi_begin():
    print("ROI begin: resetting gem5 statistics")
    m5.stats.reset()
    yield False


def handle_roi_end():
    print("ROI end: dumping statistics and stopping simulation")
    m5.stats.dump()
    yield True



processor = SimpleProcessor(
    cpu_type=CPUTypes.TIMING,
    isa=ISA.RISCV,
    num_cores=1,
)


cache_hierarchy = PrivateL1PrivateL2CacheHierarchy(
    l1d_size="32KiB",
    l1i_size="32KiB",
    l2_size="256KiB",
)

memory = SingleChannelDDR3_1600(
    size="1GiB"
)

board = SimpleBoard(
    clk_freq="1GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy,
)

binary = BinaryResource(
    local_path="build/test_gemm_rvv",
)

board.set_se_binary_workload(
    binary=binary,
)

simulator = Simulator(
    board=board,
)

simulator.run()

print("RVV correctness simulation finished")