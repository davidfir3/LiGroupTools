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
    if GateV[0] == GateV[-1]:
        GateV, DrainI = GateV[:int(len(GateV)/2)], DrainI[:int(len(GateV)/2)]
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
    # 画Vg-Ids图
    ax1.set_xlabel('$V_g (V)$')
    ax1.set_yscale('log')
    ax1.set_ylabel('|$I_{DS}$| (A)')
    ax1.tick_params(axis='y', which='both', colors='k', direction='in')
    ax1.plot(GateV, abs(DrainI), 'k')

    if not union:
        # 画Vg-Ids平方根图
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
        # 显示结果
        ax2.text(
            0.65,
            0.95,
            '$I_{on}/I_{off}$:%.2e' % ONOFF,
            ha='left',
            va='center',
            transform=ax2.transAxes,
        )  # Ion/Ioff
        ax2.text(
            0.65,
            0.85,
            '$\mu$=%.3g $cm^2V^{-1}s^{-1}$' % mobility,
            ha='left',
            va='center',
            transform=ax2.transAxes,
        )  # max mobility
        ax2.text(
            0.65, 0.75, 'VT=%.2f $V$' % VT, ha='left', va='center', transform=ax2.transAxes
        )  # VT

    return ax1


if __name__ == '__main__':
    wdir = input('文件夹：').strip('\"')
    os.chdir(wdir)
    OUTFILE = '../results' + os.path.basename(os.getcwd()) + '.csv'
    files = [
        file
        for file in os.listdir('.')
        # 根据条件筛选待计算文件
        if file.endswith('.xls') or file.endswith('.csv') and 'ID-VG' in file# and '\'' not in file
    ]
    outputs = []

    FIGSIZE = [6, 6]  # 设置图标尺寸
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for file in files:
        # 读取.csv/.xls文件数据
        if file.endswith('.xls'):
            data = pd.read_excel(file, sheet_name='Data', usecols='A:B')
            GateV, DrainI = data['GateV'].to_numpy(), data['DrainI'].to_numpy()
        if file.endswith('.csv'):
            data = pd.read_csv(file, skiprows=258, usecols=[1, 2])
            GateV, DrainI = data[' VG'].to_numpy(), data[' ID'].to_numpy()
        results = calMobility(GateV, DrainI, W=500.0, L=100.0)
        outputs.append(results[1:])
        # 不同器件的转移特性曲线画在同一个图
        drawGraph(results, ax, union=True)
        # 单独画图
        fig_i, ax_i = plt.subplots(figsize=FIGSIZE)
        drawGraph(results, ax_i)
        fig_i.savefig('%s.png' % file[:-4])

    fig.savefig('union.png')
    array = np.array(outputs)
    array = np.hstack((np.asarray(files).reshape((-1, 1)), outputs), dtype=object)
    np.savetxt(
        OUTFILE,
        array,
        delimiter=',',
        comments='',
        header='filename,mobility,Vgmax,VT,slope,ONOFF,Ion,Ioff',
        fmt=['%s','%.4e','%.4e','%.4e','%.4e','%.4e','%.4e','%.4e']
    )
    # plt.show()
