from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.components.processors.cpu_types import CPUTypes
from gem5.isas import ISA
from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import (
    PrivateL1PrivateL2CacheHierarchy,
)
from gem5.components.memory.single_channel import SingleChannelDDR3_1600

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

print("gem5 import successful")