"""
This module provides information about CUDA resources (GPUs).
"""
from textwrap import dedent

from miutil import cuinfo


def dev_info(showprop=False):
    """
    Obtain information about installed GPU devices.
    """
    if showprop:
        for i in range(cuinfo.num_devices()):
            print(
                dedent(
                    """\
                    Name: {0:s}
                     mem: {1[0]:d} MiB total, {1[1]:d} MiB free, {1[2]:d} MiB used
                      CC: {2[0]:d}.{2[1]:d}\
                    """
                ).format(
                    cuinfo.name(i),
                    [mem >> 20 for mem in cuinfo.memory(i)],
                    cuinfo.compute_capability(i),
                )
            )
    return [
        (cuinfo.name(i), cuinfo.memory(i)[0] >> 20, *cuinfo.compute_capability(i))
        for i in range(cuinfo.num_devices())
    ]


gpuinfo = dev_info
