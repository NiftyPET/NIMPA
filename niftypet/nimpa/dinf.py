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
        for i in range(cuinfo.get_device_count()):
            print(
                dedent(
                    """\
                    Name: {0:s}
                     mem: {1[0]:d} MiB total, {1[1]:d} MiB free, {1[2]:d} MiB used
                      CC: {2[0]:d}.{2[1]:d}\
                    """
                ).format(
                    cuinfo.get_name(i),
                    [mem >> 20 for mem in cuinfo.get_mem(i)],
                    cuinfo.get_cc(i),
                )
            )
    return [
        (cuinfo.get_name(i), cuinfo.get_mem(i)[0] >> 20, *cuinfo.get_cc(i))
        for i in range(cuinfo.get_device_count())
    ]


gpuinfo = dev_info
