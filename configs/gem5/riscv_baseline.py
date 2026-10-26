
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
from gem5.simulate.exit_event import ExitEvent

import sys
import m5


def handle_roi_begin():
    print("ROI begin: resetting gem5 statistics")
    m5.stats.reset()
    yield False


def handle_roi_end():
    print("ROI end: dumping statistics and stopping simulation")
    m5.stats.dump()
    yield True


# Workload arguments

args = sys.argv[1:]

if len(args) not in (2, 4):
    raise SystemExit(
        "Usage: riscv_baseline.py <kernel> <size>\n"
        "   or: riscv_baseline.py <kernel> <M> <N> <K>"
    )

kernel = args[0]

if kernel not in ("scalar", "rvv"):
    raise SystemExit(f"Unknown kernel: {kernel}")

try:
    dimensions = [int(value) for value in args[1:]]
except ValueError:
    raise SystemExit("Matrix dimensions must be integers.")

if any(value <= 0 for value in dimensions):
    raise SystemExit("Matrix dimensions must be positive.")


# Processor

processor = SimpleProcessor(
    cpu_type=CPUTypes.TIMING,
    isa=ISA.RISCV,
    num_cores=1,
)


# Cache hierarchy

cache_hierarchy = PrivateL1PrivateL2CacheHierarchy(
    l1d_size="32KiB",
    l1i_size="32KiB",
    l2_size="256KiB",
)


# Memory

memory = SingleChannelDDR3_1600(
    size="1GiB"
)


# Board

board = SimpleBoard(
    clk_freq="1GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy,
)


# Workload

binary = BinaryResource(
    local_path="build/gemm_riscv",
)

board.set_se_binary_workload(
    binary=binary,
    arguments=[kernel] + [str(value) for value in dimensions],
)


# Simulation

simulator = Simulator(
    board=board,
    on_exit_event={
        ExitEvent.WORKBEGIN: handle_roi_begin(),
        ExitEvent.WORKEND: handle_roi_end(),
    },
)

simulator.run()

print("gem5 simulation completed")
