import os
import segyio
import numpy as np
from seis_utils.gain import gain
def readsegy_ith_agc(data_dir, file, j,trace_per_shot, agc):
    filename = os.path.join(data_dir, file)
    with segyio.open(filename, 'r', ignore_geometry=True) as f:
        f.mmap()
        sourceX = f.attributes(segyio.TraceField.SourceX)[:]
        trace_num = len(sourceX)  # number of trace, The sourceX under the same shot is the same character.
        shot_num = int(float(trace_num / trace_per_shot))  # 224 787
        len_shot = trace_num // shot_num  # The length of the data in each shot data
        data = np.asarray([np.copy(x) for x in f.trace[j * len_shot:(j + 1) * len_shot]]).T
        if agc:
            data = gain(data, 0.004, 'agc', 0.05, 1)
        # data = data/data.max()
        # data = data  # 先不做归一化处理
        x = data[:, :]
        f.close()
        return x

def readsegy(data_dir, file):
    filename = os.path.join(data_dir, file)
    with segyio.open(filename, 'r', ignore_geometry=True) as f:
        f.mmap()
        # # print binary header info
        # print(f.bin)
        # print(f.bin[segyio.BinField.Traces])
        # # read headerword inline for trace 10
        # print(f.header[10][segyio.TraceField.INLINE_3D])
        # # Print inline and crossline axis
        # print(f.xlines)
        # print(f.ilines)

        # Extract header word for all traces
        sourceX = f.attributes(segyio.TraceField.SourceX)[:]

        # # Scatter plot sources and receivers color-coded on their number
        # plt.figure()
        # sourceY = f.attributes(segyio.TraceField.SourceY)[:]
        # nsum = f.attributes(segyio.TraceField.NSummedTraces)[:]
        # plt.scatter(sourceX, sourceY, c=nsum, edgecolor='none')
        #
        # groupX = f.attributes(segyio.TraceField.GroupX)[:]
        # groupY = f.attributes(segyio.TraceField.GroupY)[:]
        # nstack = f.attributes(segyio.TraceField.NStackedTraces)[:]
        # plt.scatter(groupX, groupY, c=nstack, edgecolor='none')

        trace_num = len(sourceX)  # number of trace, The sourceX under the same shot is the same character.
        data = np.asarray([np.copy(x) for x in f.trace[:trace_num]]).T
        x = data[:, :]
        f.close()
        return x
def readsegy(file):
    filename =  file
    with segyio.open(filename, 'r', ignore_geometry=True) as f:
        f.mmap()
        # Extract header word for all traces
        sourceX = f.attributes(segyio.TraceField.SourceX)[:]
        trace_num = len(sourceX)  # number of trace, The sourceX under the same shot is the same character.
        data = np.asarray([np.copy(x) for x in f.trace[:trace_num]]).T
        x = data[:, :]
        f.close()
        return x

