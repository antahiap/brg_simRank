from qd.cae.dyna import Binout
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import seaborn as sns
from scipy import signal
import random
import tikzplotlib as tikz


def rmBarrier(ids, idsMax, barrier):
    idsMax1 = idsMax[np.where(ids[idsMax] < barrier[0])]
    idsMax2 = idsMax[np.where(ids[idsMax] > barrier[1])]

    if idsMax1.size == 0:
        return(idsMax2)
    elif idsMax2.size == 0:
        return(idsMax1)
    else:
        idsMax = np.concatenate(idsMax1, idsMax2)
        return(idsMax)


def read_curve(src, pidMin, pidMax, pidExc, ePerc, barrier=[0, 0]):
    print(src)
    binout = Binout(os.path.join(src, "binout*"))
    curve = binout.read(var1, var2)
    t = binout.read(var1, 'time')
    ids = binout.read(var1, 'ids')
    print(ids)
    pids0 = np.array(pidExc)
    ids0 = np.in1d(ids, pids0).nonzero()[0]
    curve0 = curve.T[ids0].T

    idsMax = np.argsort([max(x) for x in curve.T])
    idsMax = idsMax[::-1]
    idsMax = rmBarrier(ids, idsMax, barrier)
    idsK = idsMax[pidMin:pidMax]
    pidK = ids[idsK]
    curve0 = curve.T[idsMax][pidMin:pidMax].T
    energy_key = np.sum(np.max(curve, axis=0)) * ePerc

    return(curve0, t, pidK, idsMax, energy_key)


def curve_filter(cg):
    # filter
    n = 75
    b = [1.0 / n] * n
    print(b)
    a = 1

    # b, a = signal.butter(100, 0.5)
    # b, a = signal.butter(4, 1000, 'low', analog=True)
    sos = signal.ellip(13, 0.009, 80, 0.05, output='sos')
    cg_f1 = signal.sosfilt(sos, cg)
    cg_f2 = signal.filtfilt(b, a, cg, axis=0)
    cg_f3 = signal.lfilter(b, a, cg, axis=0)  # used
    # cg_fn = (cg_f - np.amin(cg_f, axis=0))/np.ptp(cg_f, axis=0 # normfiltered
    cg_fn1 = cg_f1 / np.max(cg_f1, axis=0)  # norm-filtered
    cg_fn2 = cg_f2 / np.max(cg_f2, axis=0)  # norm-filtered
    cg_fn3 = cg_f3 / np.max(cg_f3, axis=0)  # norm-filtered
    return(cg_fn1, cg_fn2, cg_fn3)


def plot_energy_shape_std():
    src = '/export/PAG_DATA/VORDERWAGEN/002_Daten_Kaan/*505/*/OUTPUT'
    global var1, var2
    var1 = 'matsum'
    var2 = 'internal_energy'
    sns.set(color_codes=True)

    pidExc = np.array([20005200, 20001800])
    ePerc = 0.0007
    maxPerc = 0.95
    pidMin, pidMax = 0, 20
    row, col = 4, 5

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels

    srci = glob.glob(src)[2]

    fig, axs = plt.subplots(row, col)
    curve0, t, ids, idxMax, energy_key = read_curve(
                        srci, pidMin, pidMax, pidExc, ePerc)

    c_n = (curve0 - np.amin(curve0, axis=0)) / np.ptp(curve0, axis=0)  # norm
    t_c_min = t[np.argmax(curve0 > energy_key, axis=0)]
    t_c_max_perc = [t[
        np.where(x >= np.max(x) * maxPerc)][0] for x in curve0.T
        ]

    a, B = t_c_min, t_c_max_perc
    for i, c in enumerate(c_n.T):
        c_eval = curve0[:, i]
        mean, std = np.ptp(c_eval) / 2, np.std(c_eval)
        std = np.sqrt(np.mean(abs(c_eval - mean) ** 2))

        k = int(i / row)  # + 2*int(i/row)
        j = i % row
        # ---------------------------------    axs[j, k].----------------------
        axs_jk = axs[j, k]

        c_ni = c_n[:, i]
        axs_jk.plot(t, curve0[:, i], 'k', linewidth=0.8)
        # axs_jk.plot(t, c_ni, 'k', linewidth=1)
        axs_jk.plot([0, 0.12], [mean + std, mean + std], 'b--')
        axs_jk.plot([0, 0.12], [mean - std, mean - std], 'b--', label='$\mu \pm \sigma$')
        axs_jk.plot([a[i], a[i]], [0, mean + std], 'limegreen', linewidth=1, label='$t_i$')
        axs_jk.plot([B[i], B[i]], [0, mean + std], 'r', linewidth=1, label='$t_n$')
        plt.subplots_adjust(wspace=0.1, hspace=0.08)
        fig.set_size_inches(5, 3.1)
        plt.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.14)
        axs_jk.set_xticks([])
        axs_jk.set_yticks([])
        plt.legend(ncol=3, bbox_to_anchor=(-0.7, 0.01), prop={'size':SMALL_SIZE})

    saveP = "../publication/06_KG_energyAbsorption/images/plot/nrg_fts_std.pdf"
    plt.savefig(saveP)
    plt.show()
    sns.set(color_codes=False)


def plot_energy_shape():
    src = '/export/PAG_DATA/VORDERWAGEN/002_Daten_Kaan/*505/*/OUTPUT'
    global var1, var2
    var1 = 'matsum'
    var2 = 'internal_energy'
    sns.set(color_codes=True)

    pidExc = np.array([20005200, 20001800])
    ePerc = 0.0007
    maxPerc = 0.95
    pidMin, pidMax = 0, 20
    row, col = 4, 5

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels

    srci = glob.glob(src)[2]

    fig, axs = plt.subplots(row, col)
    curve0, t, ids, idxMax, energy_key = read_curve(
                        srci, pidMin, pidMax, pidExc, ePerc)

    c_n = (curve0 - np.amin(curve0, axis=0)) / np.ptp(curve0, axis=0)  # norm
    t_c_min = t[np.argmax(curve0 > energy_key, axis=0)]
    t_c_max_perc = [t[
        np.where(x >= np.max(x) * maxPerc)][0] for x in curve0.T
        ]

    a, B = t_c_min, t_c_max_perc
    for i, c in enumerate(c_n.T):
        c_eval = curve0[:, i]
        mean, std = np.ptp(c_eval) / 2, np.std(c_eval)
        std = np.sqrt(np.mean(abs(c_eval - mean) ** 2))

        k = int(i / row)  # + 2*int(i/row)
        j = i % row
        # ---------------------------------    axs[j, k].----------------------
        axs_jk = axs[j, k]

        c_ni = c_n[:, i]
        axs_jk.plot(t, curve0[:, i], 'k', linewidth=0.8)
        # axs_jk.plot(t, c_ni, 'k', linewidth=1)
        plt.subplots_adjust(wspace=0.1, hspace=0.08)
        fig.set_size_inches(5, 3.1)
        plt.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.14)
        axs_jk.set_xticks([])
        axs_jk.set_yticks([])
        # plt.legend(ncol=3, bbox_to_anchor=(-0.7, 0.01), prop={'size':SMALL_SIZE})

    saveP = "../publication/06_KG_energyAbsorption/images/plot/nrg_fts_std.pdf"
    plt.savefig(saveP)
    plt.show()
    sns.set(color_codes=False)


def plot_energy_shape_ti_tn():
    src = '/export/PAG_DATA/VORDERWAGEN/002_Daten_Kaan/*505/*/OUTPUT'
    global var1, var2
    var1 = 'matsum'
    var2 = 'internal_energy'
    sns.set(color_codes=True)

    pidExc = np.array([20005200, 20001800])
    ePerc = 0.0007
    maxPerc = 0.95
    pidMin, pidMax = 0, 20
    row, col = 4, 5

    srci = glob.glob(src)[0]

    fig, axs = plt.subplots(row, col)
    curve0, t, ids, idxMax, energy_key = read_curve(
                        srci, pidMin, pidMax, pidExc, ePerc)

    c_n = (curve0 - np.amin(curve0, axis=0)) / np.ptp(curve0, axis=0)  # norm
    t_c_min = t[np.argmax(curve0 > energy_key, axis=0)]
    t_c_max_perc = [t[
        np.where(x >= np.max(x) * maxPerc)][0] for x in curve0.T
        ]

    a, B = t_c_min, t_c_max_perc
    for i, c in enumerate(c_n.T):

        k = int(i / row)  # + 2*int(i/row)
        j = i % row
        # ---------------------------------    axs[j, k].----------------------
        axs_jk = axs[j, k]

        c_ni = c_n[:, i]
        # axs_jk.plot(t, curve0[:, i], 'k', linewidth=1)
        axs_jk.plot(t, c_ni, 'k', linewidth=1)
        axs_jk.plot([a[i], a[i]], [0, 1], 'limegreen', linewidth=1, label='$t_i$')
        axs_jk.plot([B[i], B[i]], [0, 1], 'r', linewidth=1, label='$t_n$')
        plt.subplots_adjust(wspace=0.1, hspace=0.08)
        fig.set_size_inches(5, 3.1)
        plt.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.14)
        axs_jk.set_xticks([])
        axs_jk.set_yticks([])
        plt.legend(ncol=2, bbox_to_anchor=(-0.7, 0.01))

    saveP = "../publication/KG_energyAbsorption/images/plot/nrg_fts_shape_ti_tn.pdf"
    plt.savefig(saveP)
    plt.show()
    sns.set(color_codes=False)


def plot_single_energy():

    src = '/export/PAG_DATA/VORDERWAGEN/002_Daten_Kaan/*505/*/OUTPUT'
    global var1, var2
    var1 = 'matsum'
    var2 = 'internal_energy'
    # sns.set(color_codes=True)

    pidExc = np.array([20005200, 20001800])
    ePerc = 0.0007
    pidMin, pidMax = 0, 1

    srcList = glob.glob(src)
    srci = srcList[0]
    curv0, t, ids, idxMax, energy_key = read_curve(
                            srci, pidMin, pidMax, pidExc, ePerc)

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    fig, ax = plt.subplots()
    ti, tn, IE_min, IE_max = 42, 95, 0, 5.35e4
    x = 1
    dy = 0.5e4
    dyt = 30
    ax.plot(t * 1000, curv0 / 1000, 'k', linewidth=1)
    ax.plot([ti, ti], [IE_min - dy, IE_max], 'limegreen', linewidth=1)
    ax.annotate('$t_i$', xy=(ti + x, IE_min - dy), color='limegreen')
    ax.plot([tn, tn], [IE_min - dy, IE_max], 'r', linewidth=1)
    ax.annotate('$t_n$', xy=(tn + x, IE_min - dy), color='r')
    ax.plot([tn - dyt, tn], [IE_max, IE_max], '--k', linewidth=1)
    ax.annotate('$IE_{max}$', xy=(tn - dyt, IE_max + dy / 4))

    plt.ylim([-0.8e4, 6e4])
    ax.set(xlabel='$t\ [ms]$', ylabel='$IE\ [kN-mm]$')
    ax.ticklabel_format(axis='y', style='sci', scilimits=[0, 1])

    ax.set_xticks(range(0, 125, 40))
    ax.set_yticks(np.arange(0, 7e4, 2e4))

    fig.set_size_inches(3.1, 3.1)
    plt.subplots_adjust(left=0.2, right=0.975, top=0.93, bottom=0.16)

    saveP = "../publication/KG_energyAbsorption/images/plot/nrg_fts_1.pdf"
    plt.savefig(saveP)
    plt.show()


def plot_nrm_grd():
    srci = '/export/PAG_DATA/VORDERWAGEN/002_Daten_Kaan/ROB_VOWA_505/Design0025/OUTPUT'
    global var1, var2
    var1 = 'matsum'
    var2 = 'internal_energy'

    pidExc = np.array([20005200, 20001800])
    ePerc = 0.0007
    pidMin, pidMax = 0, 5

    curv, t, ids, idxMax, energy_key = read_curve(
                        srci, pidMin, pidMax, pidExc, ePerc)
    cg = np.gradient(curv, t, axis=0)

    cg_n = (cg - np.amin(cg, axis=0)) / np.ptp(cg, axis=0)  # norm
    c_n = (curv - np.amin(curv, axis=0)) / np.ptp(curv, axis=0)  # norm

    ti = 42
    x = 1

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    fig, ax = plt.subplots()

    ax.plot(t * 1000, c_n, 'k', linewidth=1)  # unit ms
    ax.plot(t * 1000, cg_n, 'k--', linewidth=1)  # unit ms
    ax.plot([ti, ti], [-0.1, 1], 'limegreen', linewidth=1)
    ax.annotate('$t_i$', xy=(ti + x, -0.1), color='limegreen')
    ax.set(xlabel='$t\ [ms]$')
    plt.legend(
        ['$IE_{nrm}$', '$\dot{IE}_{nrm}$'],
        loc='lower right', framealpha=1)

    # plt.xlim([0,100])
    fig.set_size_inches(3.1, 3.1)
    plt.subplots_adjust(left=0.2, right=0.975, top=0.93, bottom=0.16)

    saveP = "../publication/KG_energyAbsorption/images/plot/nrg_nrm_grd.pdf"
    # plt.savefig(saveP)
    plt.show()


def plot_nrm_grd_fltr():
    srci = '/export/PAG_DATA/VORDERWAGEN/002_Daten_Kaan/ROB_VOWA_505/Design0025/OUTPUT'
    global var1, var2
    var1 = 'matsum'
    var2 = 'internal_energy'

    pidExc = np.array([20005200, 20001800])
    ePerc = 0.0007
    pidMin, pidMax = 0, 1

    curv, t, ids, idxMax, energy_key = read_curve(
                        srci, pidMin, pidMax, pidExc, ePerc)
    cg = np.gradient(curv, t, axis=0)

    cg_n = (cg - np.amin(cg, axis=0)) / np.ptp(cg, axis=0)  # norm
    c_n = (curv - np.amin(curv, axis=0)) / np.ptp(curv, axis=0)  # norm
    cg_fn1, cg_fn2, cg_fn3 = curve_filter(cg)

    ti = 42
    x = 1

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    fig, ax = plt.subplots()

    ax.plot(t * 1000, c_n, 'k', linewidth=1)
    ax.plot(t * 1000, cg_fn3, 'k--', linewidth=1)
    ax.plot([ti, ti], [-0.1, 1], 'limegreen', linewidth=1)
    ax.annotate('$t_i$', xy=(ti + x, -0.1), color='limegreen')
    ax.set(xlabel='$t\ [ms]$')
    plt.legend(
        ['$IE_{nrm}$', '$\dot{IE}_{nrm}$'],
        loc='lower right', framealpha=1)

    # plt.xlim([0,100])
    ax.set_xticks(range(0, 125, 40))
    ax.set_yticks(np.arange(0.0, 1.2, 0.4))
    fig.set_size_inches(3.1, 3.1)
    plt.subplots_adjust(left=0.2, right=0.975, top=0.93, bottom=0.16)

    saveP = "../publication/KG_energyAbsorption/images/plot/nrg_nrm_grd_fltr.pdf"
    plt.savefig(saveP)
    plt.show()


def compare_fltr():
    srci = '/export/PAG_DATA/VORDERWAGEN/002_Daten_Kaan/ROB_VOWA_505/Design0025/OUTPUT'
    global var1, var2
    var1 = 'matsum'
    var2 = 'internal_energy'

    pidExc = np.array([20005200, 20001800])
    ePerc = 0.0007
    pidMin, pidMax = 0, 1

    ti = 42
    x = 1

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    fig, ax = plt.subplots()

    curv, t, ids, idxMax, energy_key = read_curve(
                        srci, pidMin, pidMax, pidExc, ePerc)
    # c_n = (curv - np.amin(curv, axis=0))/np.ptp(curv, axis=0)  # norm
    cg = np.gradient(curv, t, axis=0)
    cg_n = (cg - np.amin(cg, axis=0)) / np.ptp(cg, axis=0)  # norm
    cg_fn1, cg_fn2, cg_fn3 = curve_filter(cg)

    ax.plot(t * 1000, cg_fn1, 'c', linewidth=1)
    ax.plot(t * 1000, cg_fn2, 'b', linewidth=1)
    ax.plot(t * 1000, cg_fn3, 'r', linewidth=1)
    ax.plot(t * 1000, cg_n, 'k--', linewidth=1)
    # ax.plot(t*1000, c_n, 'k', linewidth=1)
    ax.set(xlabel='$t\ [ms]$', ylabel='$\dot{IE}_{nrm}$')
    fig.set_size_inches(3.1, 3.1)
    plt.legend(
        ['$sosfilt$', 'filtfilt', 'lfilter', 'input'],
        loc='upper left', prop={'size':9})

    plt.xlim([-10, 125])
    ax.set_xticks(range(0, 125, 40))
    ax.set_yticks(np.arange(0.0, 1.2, 0.4))
    plt.subplots_adjust(left=0.2, right=0.975, top=0.93, bottom=0.16)

    saveP = "../publication/KG_energyAbsorption/images/plot/nrg_fltr.pdf"
    plt.savefig(saveP)
    plt.show()


def sample():
    srci = '/export/PAG_DATA/VORDERWAGEN/002_Daten_Kaan/ROB_VOWA_505/Design0025/OUTPUT'
    global var1, var2
    var1 = 'matsum'
    var2 = 'internal_energy'

    pidExc = np.array([20005200, 20001800])
    ePerc = 0.0007
    pidMin, pidMax = 0, 1

    ti = 42
    x = 1

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    fig, ax = plt.subplots()

    curv, t, ids, idxMax, energy_key = read_curve(
                        srci, pidMin, pidMax, pidExc, ePerc)
    cg = np.gradient(curv, t, axis=0)
    ax.set_xticks(range(0, 125, 40))
    ax.set_yticks(np.arange(0.0, 1.2, 0.4))

    # plt.xlim([0,100])
    fig.set_size_inches(3.1, 3.1)
    plt.subplots_adjust(left=0.2, right=0.975, top=0.93, bottom=0.16)
    saveP = "../publication/KG_energyAbsorption/images/plot/change.pdf"
    # plt.savefig(saveP)
    plt.show()


def cevt_plot_energy_shape_std():
    # src = '/home/apakiman/Projects/kg01/dash-nrg/assets/CEVT/CEVT/Rep/3_stv0/fp3/runs/*'
    src = '/home/apakiman/Projects/kg01/src/CEVT/3_stv0/front/runs/*fp3*'
    global var1, var2
    var1 = 'matsum'
    var2 = 'internal_energy'
    sns.set(color_codes=True)

    pidExc = np.array([])
    ePerc = 0.0007
    maxPerc = 0.95
    pidMin, pidMax = 0, 20
    row, col = 4, 5

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels

    print(glob.glob(src))
    # srci = glob.glob(src)[1]

    for srci in glob.glob(src):
        print(srci)
        fig, axs = plt.subplots(row, col)
        curve0, t, ids, idxMax, energy_key = read_curve(
                            srci, pidMin, pidMax, pidExc, ePerc)

        c_n = (curve0 - np.amin(curve0, axis=0)) / np.ptp(curve0, axis=0)  # norm
        t_c_min = t[np.argmax(curve0 > energy_key, axis=0)]
        t_c_max_perc = [t[
            np.where(x >= np.max(x) * maxPerc)][0] for x in curve0.T
            ]

        a, B = t_c_min, t_c_max_perc

        for i, c in enumerate(c_n.T):
            c_eval = curve0[:, i]
            mean, std = np.ptp(c_eval) / 2, np.std(c_eval)
            std = np.sqrt(np.mean(abs(c_eval - mean) ** 2))

            k = int(i / row)  # + 2*int(i/row)
            j = i % row
            # ---------------------------------    axs[j, k].   ----------------------
            axs_jk = axs[j, k]

            c_ni = c_n[:, i]
            axs_jk.plot(t, curve0[:, i], 'k', linewidth=0.8)
            # axs_jk.plot(t, c_ni, 'k', linewidth=1)
            axs_jk.plot([0, 80], [mean + std, mean + std], 'b--')
            axs_jk.plot([0, 80], [mean - std, mean - std], 'b--', label='$\mu \pm \sigma$')
            axs_jk.plot([a[i], a[i]], [0, mean + std], 'limegreen', linewidth=1, label='$t_i$')
            axs_jk.plot([B[i], B[i]], [0, mean + std], 'r', linewidth=1, label='$t_n$')
            plt.subplots_adjust(wspace=0.1, hspace=0.08)
            fig.set_size_inches(5, 3.1)
            plt.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.14)
            axs_jk.set_xticks([])
            axs_jk.set_yticks([])
            plt.legend(ncol=3, bbox_to_anchor=(-0.7, 0.01), prop={'size':SMALL_SIZE})

        # saveP = "../publication/06_KG_energyAbsorption/images/plot/nrg_fts_std_fp3_stv0.svg"
        # plt.savefig(saveP)
        plt.show()
    sns.set(color_codes=False)


def cevt_plot_energy_shape_std_3lc():
    # src = '/home/apakiman/Projects/kg01/dash-nrg/assets/CEVT/CEVT/Rep/3_stv0/fp3/runs/*'
    src = '/home/apakiman/Projects/kg01/src/CEVT/3_m1/front/runs/{}'
    lcs = ['*fp3*', '*fod_*', '*fo5*']

    global var1, var2
    var1 = 'matsum'
    var2 = 'internal_energy'
    sns.set(color_codes=True)

    pidExc = np.array([])
    ePerc = 0.0007
    maxPerc = 0.95
    pidMin, pidMax = 0, 5
    row, col = 3, 5

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels

    print(glob.glob(src))
    # srci = glob.glob(src)[1]

    te = [80, 80, 140]
    for vis in [0, 0, 2]:  # range(0, 10):
        fig, axs = plt.subplots(row, col)
        for li, lci in enumerate(lcs):
            src_glob = glob.glob(src.format(lci))
            irand = [0, 0, 2][li]  # random.randint(0, len(src_glob))
            # print(src_glob)
            srci = src_glob[irand]
            print(srci, irand)
            curve0, t, ids, idxMax, energy_key = read_curve(
                                srci, pidMin, pidMax, pidExc, ePerc, barrier=[98000000, 99999999])
            # curve0 = curve0[:, ::2]
            # ids = ids [::2]
            print(' & '.join([str(x) for x in ids[pidMin:pidMax]]))

            c_n = (curve0 - np.amin(curve0, axis=0)) / np.ptp(curve0, axis=0)  # norm
            t_c_min = t[np.argmax(curve0 > energy_key, axis=0)]
            t_c_max_perc = [t[
                np.where(x >= np.max(x) * maxPerc)][0] for x in curve0.T
                ]

            a, B = t_c_min, t_c_max_perc

            for i in range(0, col):
                c_eval = curve0[:, i]
                mean, std = np.ptp(c_eval) / 2, np.std(c_eval)
                # print(std)
                # std = np.sqrt(np.mean(abs(c_eval - mean) ** 2))
                # print(std)
                # break

                k = i  # int(i / row)  # + 2*int(i/row)
                j = li  # % row
                # print(li, i)
                # ---------------------------------    axs[j, k].       ----------------------
                axs_jk = axs[j, k]

                c_ni = c_n[:, i]
                axs_jk.plot(t, curve0[:, i], 'k', linewidth=0.8)
                # axs_jk.plot(t, c_ni, 'k', linewidth=1)
                axs_jk.plot([0, te[li]], [mean + std, mean + std], 'b--')
                axs_jk.plot([0, te[li]], [mean - std, mean - std], 'b--', label='$\mu \pm \sigma$')
                axs_jk.plot([a[i], a[i]], [0, mean + std], 'limegreen', linewidth=1, label='$t_i$')
                axs_jk.plot([B[i], B[i]], [0, mean + std], 'r', linewidth=1, label='$t_n$')
                plt.subplots_adjust(wspace=0.1, hspace=0.08)
                fig.set_size_inches(5, 3.1)
                plt.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.02)
                axs_jk.set_xticks([])
                axs_jk.set_yticks([])
                plt.legend(ncol=3, bbox_to_anchor=(-0.7, 3.7), prop={'size':SMALL_SIZE})

            # saveP = "../publication/06_KG_energyAbsorption/images/plot/nrg_fts_std_fp3_m1_3lc.svg"
            saveP = "../publication/06_KG_energyAbsorption/images/plot/nrg_fts_std_fp3_m1_3lc.tex"
            # plt.savefig(saveP)
        tikz.save(saveP, standalone=True)
        plt.show()
    sns.set(color_codes=False)


def cevt_plot_single_energy():

    src = '/home/apakiman/Projects/kg01/dash-nrg/assets/CEVT/CEVT/Rep/3_stv0/fp3/runs/*'
    # src = '/home/apakiman/Projects/kg01/src/CEVT/3_stv0/front/runs/*'
    global var1, var2
    var1 = 'matsum'
    var2 = 'internal_energy'
    sns.set(color_codes=True)

    pidExc = np.array([])
    ePerc = 0.0007
    pidMin, pidMax = 0, 1

    srcList = glob.glob(src)
    srci = srcList[0]
    print(srci)
    curv0, t, ids, idxMax, energy_key = read_curve(
                            srci, pidMin, pidMax, pidExc, ePerc)
    curv0 = curv0 / 1000
    mean, std = np.ptp(curv0) / 2, np.std(curv0)
    std = np.sqrt(np.mean(abs(curv0 - mean) ** 2))

    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    fig, ax = plt.subplots()
    ti, tn, IE_min, IE_max = 19.5, 73.8, 0, 2.52e1
    x = 1
    dy = 0.3e1
    dyt = 30

    ax.plot(t , curv0, 'k', linewidth=1)
    ax.plot([ti, ti], [IE_min - dy, IE_max], 'limegreen', linewidth=1)
    ax.annotate('$t_i$', xy=(ti + x, IE_min - dy), color='limegreen')
    ax.plot([tn, tn], [IE_min - dy, IE_max], 'r', linewidth=1)
    ax.annotate('$t_n$', xy=(tn + x, IE_min - dy), color='r')
    ax.plot([tn - dyt, tn], [IE_max, IE_max], '--k', linewidth=1)
    ax.annotate('$IE_{max}$', xy=(tn - dyt, IE_max + dy / 4))

    hl = 1
    gap_t = 10
    ax.arrow(ti + gap_t, -0.5, tn - ti - hl - gap_t, 0, head_width=0.5, head_length=hl, fc='k', ec='k')
    ax.arrow(ti + gap_t, -0.5, -gap_t + hl, 0, head_width=0.5, head_length=hl, fc='k', ec='k')
    ax.annotate('$\Delta t$', xy=(ti + (tn - ti) / 2, 0))

    print(mean, std)
    ax.plot([0, 80], [mean + std, mean + std], 'b--')
    ax.plot([0, 80], [mean - std, mean - std], 'b--', label='$\mu \pm \sigma$')

    plt.ylim([-0.4e1, 2.8e1])
    plt.xlim([0, 80])
    ax.set(xlabel='$t\ [ms]$', ylabel='$IE\ [kNmm]$')
    # ax.ticklabel_format(axis='y', style='sci', scilimits=[0, 1])

    ax.set_xticks(range(0, 80, 30))
    ax.set_yticks(np.arange(0, 30, 10))

    fig.set_size_inches(4.1 , 2.79)
    plt.subplots_adjust(left=0.16, right=0.975, top=0.93, bottom=0.21)

    saveP = "../publication/06_KG_energyAbsorption/images/plot/cevt_nrg_fts_1.pdf"
    # saveP = "../publication/06_KG_energyAbsorption/images/plot/cevt_nrg_fts_1.tex"
    plt.savefig(saveP)
    print(saveP)
    # tikz.save(saveP, standalone=True)
    plt.show()


if __name__ == '__main__':
# PAG
    # plot_single_energy()
    # plot_energy_shape_ti_tn()
    # plot_nrm_grd()
    # compare_fltr()
    # plot_nrm_grd_fltr()
    # plot_energy_shape_std()
    # plot_energy_shape()

# CEVT
    # cevt_plot_energy_shape_std()
    cevt_plot_energy_shape_std_3lc()
    cevt_plot_single_energy()
