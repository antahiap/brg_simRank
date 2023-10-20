from qd.cae.dyna import Binout
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import glob
import seaborn as sns
import re


def curve_filter(cg):
    # filter
    n = 75
    b = [1.0 / n] * n
    a = 1

    # b, a = signal.butter(100, 0.5)
    # b, a = signal.butter(4, 1000, 'low', analog=True)
    # cg_f = signal.sosfilt(sos, cg)
    # cg_f = signal.filtfilt(b, a, cg, axis=0)
    cg_f = signal.lfilter(b, a, cg, axis=0)
    # cg_fn = (cg_f - np.amin(cg_f, axis=0))/np.ptp(cg_f, axis=0 # normfiltered
    cg_fn = cg_f/np.max(cg_f, axis=0)     # norm-filtered
    return(cg_fn)


def read_curve(src, pidMin, pidMax, pidExc, ePerc):
    binout = Binout(os.path.join(src, "binout*"))
    curve = binout.read(var1, var2)
    t = binout.read(var1, 'time')
    ids = binout.read(var1, 'ids')
    print(type(ids[0]))
    print(t.shape, curve.shape, ids.shape)
    pids0 = np.array(pidExc)
    ids0 = np.in1d(ids, pids0).nonzero()[0]
    print(curve.shape)
    curve0 = curve.T[ids0].T
    print(curve.shape, curve0.shape)

    # get kth max curves
    # kk = 0
    # k = 20
    # row, col = 4,5
    idxMax = np.argsort([max(x) for x in curve.T])
    idxMax = idxMax[::-1]
    curve0 = curve.T[idxMax][pidMin:pidMax].T
    energy_key = np.sum(np.max(curve, axis=0))*ePerc

    return(curve0, t, ids, idxMax, energy_key)


def plot_list(row, col, curve0, t, cg_n, b, a, B, plotOpt=True):
    fig, axs = plt.subplots(row, col)
    for i, c in enumerate(cg_n.T):
        c_eval = curve0[:, i]
        # mean, std, max = np.mean(c_eval), np.std(c_eval), np.max(c_eval)
        mean, std = np.ptp(c_eval)/2, np.std(c_eval)
        # max = np.max(c_eval)
        std = np.sqrt(np.mean(abs(c_eval - mean)**2))

        # s = 1
        # while not mean - std > 0:   # or mean + std < max:
        #     c_e = curve0[s*5:, i]
        #     mean, std = np.ptp(c_eval)/2, np.std(c_e)
        #     s += 1
        #     # break
        #     print(s)
        tdt = min(t[np.where(c_eval > np.ptp(c_eval)/2)])
        print(tdt)

        k = int(i/row)  # + 2*int(i/row)
        j = i % row
        print(i, j, k)
        # axs[j,k].plot([0, 0.12], [mean, mean])
        # axs[j,k].plot([0, 0.12], [mean+std, mean+std], 'b--')
        # axs[j,k].plot([0, 0.12], [mean-std, mean-std], 'b--')
        # axs[j,k].plot([a[i], a[i]], [0, max])
        # axs[j,k].plot([tdt, tdt], [0, max])

        final_list = [x for x in c_eval if (x > mean - std)]
        final_list = [x for x in final_list if (x < mean + std)]
        # tt = t[np.where((c_eval > (mean - std)) & (c_eval < (mean + std)))]

        # plt.suptitle('plot {}'. format(i))
        # axs[j, k].plot(tt, final_list, 'ro')
        # axs[j, k].plot(t, c_eval)
        # axs[j, k].set_title('{:2.2f} vs {:2.2f}'.format(np.min(tt), a[i]) )

        # ---------------------------------    axs[j, k].----------------------
        t_fn = b[i]
        if row > 1 and col > 1:
            axs_jk = axs[j, k]
        else:
            axs_jk = axs

        # axs_jk.plot(t, c)
        # axs_jk.plot(t, cg_fn[:, i])
        # axs_jk.plot([t_fn, t_fn], [0, 1])
        # axs_jk.set_title(ids[idxMax[i]])


        # ---------------------------------------------------------
        c_ni = c_n[:, i]
        mean_n, std_n = np.ptp(c_ni)/2, np.std(c_ni)
        # max_n = np.max(c_ni)
        std = np.sqrt(np.mean(abs(c_eval - mean)**2))

        cut = np.where((c_ni > (mean_n - std_n)) & (c_ni < (mean_n + std_n)))
        # c_n_cut = c_ni[cut]
        tt_n = t[cut]
        print(tt_n)

        # plt.suptitle('plot {}'. format(i))
        # axs[j, k+2].plot(tt_n, c_n_cut, 'ro')
        #
        axs_jk.plot(t, c_ni)
        # axs_jk.plot([0, 0.12], [mean_n, mean_n])
        # axs_jk.plot([0, 0.12], [mean_n+std_n, mean_n+std_n], 'b--')
        # axs_jk.plot([0, 0.12], [mean_n-std_n, mean_n-std_n], 'b--')
        # axs[j, k+2].plot([0, 0.12], [mean_n, mean_n])
        # axs[j, k+2].plot([0, 0.12], [mean_n+std_n, mean_n+std_n], 'b--')
        # axs[j, k+2].plot([0, 0.12], [mean_n-std_n, mean_n-std_n], 'b--')
        axs_jk.plot([a[i], a[i]], [0, 1], 'g--')
        axs_jk.plot([B[i], B[i]], [0, 1], 'r--')
        # axs[j, k+2].set_title('{:2.2f} vs {:2.2f}'.
        # formatenergy_key(np.min(tt_n), a[i]) )

    if plotOpt:
        plt.show()



if __name__ == '__main__':
    src = '/export/PAG_DATA/VORDERWAGEN/002_Daten_Kaan/*505/*/OUTPUT'
    var1 = 'matsum'
    var2 = 'internal_energy'

    pidExc = np.array([20005200, 20001800])
    ePerc = 0.0007
    maxPerc = 0.95
    pidMin, pidMax = 0, 20
    row, col = 4, 5

    srcList = glob.glob(src)
    name = [re.findall('VOWA_.*/Design[0-9]*', x)[0] for x in srcList]
    maxL = []
    stdL = []

    e_max_L, t, dt = [], [], []
    for i, srci in enumerate(srcList):
        # break
        curve0, t, ids, idxMax, energy_key = read_curve(
                            srci, pidMin, pidMax, pidExc, ePerc)

        cg = np.gradient(curve0, t, axis=0)

        cg_n = (cg - np.amin(cg, axis=0))/np.ptp(cg, axis=0)       # norm
        c_n = (curve0 - np.amin(curve0, axis=0))/np.ptp(curve0, axis=0)  # norm
        cg_fn = curve_filter(cg)

        t_c_min = t[np.argmax(curve0 > energy_key, axis=0)]

        c_max = np.max(curve0, axis=0)
        t_c_max_perc = [t[
            np.where(x >= np.max(x)*maxPerc)][0] for x in curve0.T
            ]
        t_c_max = [t[np.where(x == np.max(x))][0] for x in curve0.T]

        t_cg_fn_min = t[np.argmax(cg_fn > 0.005, axis=0)]

        comp = t_c_min - t_cg_fn_min
        mean_i, max_i, std_i = np.mean(comp), max(comp), np.std(comp)

        maxL.append(max_i)
        stdL.append(mean_i + std_i)
        # break
        print(srci)
        sns.set(color_codes=True)
        # if i>0:
        #     break

        plot_list(
            row, col, curve0, t, cg_n, t_cg_fn_min, t_c_min, t_c_max_perc,
            )

        # break

    # np.savetxt('curve_feature.csv', np.c_[name[0:i+1], maxL], delimiter=',',
    # fmt='%s')
    # np.savetxt('curve_max.csv', np.c_[name[0:i+1], maxL], delimiter=',',
    # fmt='%s')
    # np.savetxt('curve_std.csv', np.c_[name[0:i+1], stdL], delimiter=',',
    # fmt='%s')
    # plt.plot(name[0:i+1], maxL)
    # plt.plot(name[0:i+1], stdL)
    # plt.xticks(rotation=90)
    # plt.show()
        # plt.plot(ids[idxMax][pidMin:pidMax], t_cg_fn_min, 'o')
        # sns.distplot(t_cg_fn_min, kde=True, rug=True)
        # plt.show()
