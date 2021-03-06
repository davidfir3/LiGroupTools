# -------------------------------------------------------------------------------------------------------------------------
# Author: Ke Shifeng, Date: 2021.09.16
# Description: this script is intended to calculte mobilities of OFET devices, using original data files acuqired from Keithley.
# Instructions for new users:
# 1. Install python (version >= 3.7) on your computer.
# 2. Install required modules using pip. For example, run 'pip install pandas numpy matplotlib xlrd' in powershell or cmd.
# 3. Copy this .py file to the directory which contains original data files. Then, run this script under that directory.
# 4. Get union and seperate pics, and results in .csv
# -------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def calMobility(GateV, DrainI, W=1000.0, L=50.0, Cd=10.0, **kwargs):
    '''
    Usage: pass two numpy ndarrays containing GateV and DrainI saperately

    Equation: u = L*2/W/Cd*1E9*(dI/dVg)^2

    Default parameters:
    W = 1000.0 um
    L = 50.0 um
    Cd = 10.0 nF/cm2
    '''

    DrainI = abs(DrainI)
    slopes = np.zeros([2, len(GateV) - 15])

    # select 15 points after i to calculate gradient at i
    for i in range(len(GateV) - 15):
        x = GateV[i : i + 15]
        y = DrainI[i : i + 15]
        p = np.polyfit(x, np.sqrt(y), 1)  # slope equation: y = p[0]*x + p[1]
        slopes[0, i], slopes[1, i] = p[0], p[1]

    mobilities = L * 2 / W / Cd * 1e9 * slopes[0, :] ** 2
    max_index = np.argmax(mobilities)  # indice of max mobility
    max_mobility = np.amax(mobilities)  # max mobility
    max_Vg = GateV[max_index]  # Vg at max mobility
    max_slope = slopes[0, max_index]  # slope at max mobility
    VT = -slopes[1, max_index] / slopes[0, max_index]
    Ion = max(DrainI)
    Ioff = min(DrainI)
    ONOFF = Ion / Ioff

    return mobilities, max_mobility, max_Vg, VT, max_slope, ONOFF, Ion, Ioff


def drawGraph(results, ax1, union=False):
    '''
    Usage: pass a list named 'results' and a Axes object 'ax1' to plot on this Figure
    '''

    mobilities, mobility, Vgmax, VT, slope, ONOFF, *_ = results
    # ???Vg-Ids???
    ax1.set_xlabel('$V_g (V)$')
    ax1.set_yscale('log')
    ax1.set_ylabel('|$I_{DS}$| ($\mu$A)')
    ax1.tick_params(axis='y', which='both', colors='k', direction='in')
    ax1.plot(GateV, abs(DrainI), 'k')

    if not union:
        # ???Vg-Ids????????????
        ax2 = ax1.twinx()
        ax2.set_ylabel('$(I_{DS})^{1/2} (A^{1/2})$', color='b')
        ax2.tick_params(axis='y', which='both', colors='b', direction='in')
        ax2.ticklabel_format(axis='y',style='sci', scilimits=[0, 0], useMathText=True)
        ax2.spines['right'].set_color('b')
        ax2.plot(GateV, np.sqrt(abs(DrainI)), 'b')
        ax2.plot(
            [VT, Vgmax + 10 * (1 if Vgmax > VT else -1)],
            [0, slope * (Vgmax + 10 * (1 if Vgmax > VT else -1) - VT)],
            'r',
        )  # plot slope in range (VT, Vgmax+10) or (VT, Vgmax-10)
        ax2.set_ylim(0)
        # ????????????
        ax2.text(
            0.5,
            0.9,
            '$I_{on}/I_{off}$:%.2E' % ONOFF,
            ha='left',
            va='center',
            transform=ax2.transAxes,
        )  # Ion/Ioff
        ax2.text(
            0.5,
            0.8,
            '$\mu$=%.2E $cm^2V^{-1}s^{-1}$' % mobility,
            ha='left',
            va='center',
            transform=ax2.transAxes,
        )  # max mobility
        ax2.text(
            0.5, 0.7, 'VT=%.2f $V$' % VT, ha='left', va='center', transform=ax2.transAxes
        )  # VT

    return ax1


if __name__ == '__main__':
    FIGSIZE = [10, 10]  # ??????????????????
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.subplots()

    files = [
        file
        for file in os.listdir('.')
        if file.endswith('.xls') or file.endswith('.csv')
    ]
    outputs = []

    for file in files:
        # ??????.csv/.xls????????????
        if file.endswith('.xls'):
            data = pd.read_excel(file, sheet_name='Data', usecols='A:B')
            GateV, DrainI = data['GateV'].to_numpy(), data['DrainI'].to_numpy()
        if file.endswith('.csv'):
            data = pd.read_csv(file, skiprows=258, usecols=[1, 2])
            GateV, DrainI = data[' VG'].to_numpy(), data[' ID'].to_numpy()
        results = calMobility(GateV, DrainI)
        outputs.append(results[1:])
        # ???????????????????????????????????????????????????
        drawGraph(results, ax, union=True)
        # ????????????
        fig_i = plt.figure(figsize=FIGSIZE)
        drawGraph(results, fig_i.subplots())
        fig_i.savefig('%s.png' % file.split('.')[0])

    fig.savefig('union.png')
    array = np.array(outputs)
    np.savetxt(
        'results.csv',
        array,
        delimiter=',',
        comments='',
        header='mobility,Vgmax,VT,slope,ONOFF,Ion,Ioff, ',
    )
    # plt.show()
