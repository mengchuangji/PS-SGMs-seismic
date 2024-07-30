# Created by UPO-JZSB on Jan/17/2021
# Released Under GPL-v3.0 License

# Version 0.1 is released on Jan/17/2021
# Implementation of wigb

# Version 0.2 is released on Jan/19/2021
# Add parameter direction - decide the plot direction

# Version 0.3 is released on Jan/20/2021
# Replace `plot' by `fill_between' function to boost the filling procession

# Version 0.4 is released on Nov/16/2021
# Fixed a problem which may cause inconsistent display
# Fixed a problem which may change original data

# Version 0.5 is released on Nov/16/2022
# Add aspect parameter to adjust data aspect
# One year passed~

# Version 0.6 is released on Feb/28/2024
# Support draw wiggle on the subplot via parameter ax

import numpy as np
import copy
import matplotlib.pyplot as plt


def wigb(a=None, scale=1, x=None, z=None, a_max=None, ax=None, figsize=(30, 15), aspect='auto', no_plot=False, direction='Vertical'):
    """
    wigb - plot seismic trace data
    Thanks to XINGONG LI's contribution on MATLAB (https://geog.ku.edu/xingong-li)

    :param a: Seismic data (trace data * traces)
    :param scale: Scale factor (Default 1)
    :param x: x-axis info (traces) (Default None)
    :param z: z-axis info (trace data) (Default None)
    :param a_max: Magnitude of input data (Default None)
    :param aspect: Display aspect (Default 'auto'). Can be 'auto', 'equal', or a positive real number
    :param ax: The axes handler (Default None). The default value indicate the axes is generated within this function
    :param figsize: Size of figure (Default (30, 15)). This parameter works only if ax is None.
    :param no_plot: Do not plot immediately (Default False)
    :param direction: Display direction (Default 'Vertical'). Either 'Vertical' or 'Horizontal'

    :return: if no_plot is False, plot the seismic data, otherwise, do not plot immediately,
            users can adjust plot parameters outside
    """
    a = copy.copy(a)
    n_data, n_trace = a.shape

    if x is None:
        x = np.arange(n_trace)
    if z is None:
        z = np.arange(n_data)
    if a_max is None:
        a_max = np.max(np.max(a, axis=0))
    if direction not in ['Horizontal', 'Vertical']:
        raise ValueError('Direction must be either \'Horizontal\' or \'Vertical\'')

    x = np.array(x)
    z = np.array(z)

    dx = np.mean(x[1:] - x[:n_trace - 1])
    dz = np.mean(z[1:] - z[:n_data - 1])

    a *= scale * dx / a_max

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect(aspect=aspect)

    if direction == 'Vertical':
        ax.set_xlim(-2 * dx, x[-1] + 2 * dx)
        ax.set_ylim(-dz, z[-1] + dz)
        ax.invert_yaxis()

        for index_x in range(n_trace):
            zero_offset = index_x*dx
            trace = a[:, index_x]
            ax.plot(zero_offset + trace, z, 'k-', linewidth=0.5)  #linewidth=2
            ax.fill_betweenx(
                np.array([y * dz for y in range(n_data)]),
                np.zeros_like(np.arange(n_data)) + zero_offset,
                trace + zero_offset,
                where=trace > 0,
                interpolate=True,
                color='k',
                antialiased=True
            )

    elif direction == 'Horizontal':
        ax.set_xlim(-dz, z[-1] + dz)
        ax.set_ylim(-2 * dx, x[-1] + 2 * dx)
        ax.invert_yaxis()

        for index_z in range(n_trace):
            zero_offset = index_z*dx
            trace = a[:, index_z]
            ax.plot(z, zero_offset + trace, 'k-', linewidth=2)

            ax.fill_between(
                np.array([y * dz for y in range(n_data)]),
                np.zeros_like(np.arange(n_data)) + zero_offset,
                trace + zero_offset,
                where=trace > 0,
                interpolate=True,
                color='k',
                antialiased=True
            )

    if not no_plot:
        plt.show()
