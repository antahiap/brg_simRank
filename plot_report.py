import _paths
import pylab
import oems
import KG_feed_OEM_data as kg
import numpy as np
import matplotlib.pyplot as plt
import cypher_notebook_nrg as cyNrg
from random import randint
import os
import sys
import re
import pandas as pd
import plotly.express as px
from matplotlib.ticker import StrMethodFormatter
import tikzplotlib as tikz

sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../note_scripts/'))


def plot_nrg_embd_similarity_PAG2(dst=''):

    ft_opt = ["nrg_max", "ti_grad", "tn_pct"]
    norm_opt = []

    simList = cyNrg.make_simList(['.*'], ['.*'], 4)
    simList = ['ROB_VOWA_739_Design0014']
    # simList = ['ROB_VOWA_739_Design0023', 'ROB_VOWA_739_Design0027']
    # 'ROB_VOWA_739_Design0004']#, 'ROB_VOWA_739_Design0014']
    #     , 'ROB_VOWA_739_Design0017', 'ROB_VOWA_739_Design0014']
    nrmList = '.*'

    # --------------------------------
    # pids_sel = [20002000] #74516431, 88000008]# , 88000010, 74000157]#20002000, 74516435, ]#, 20002000]
    # pids_sel = [20002000, 20001400, 20001100, 20004400, 20001200, 20001800, 20004900]
    pids_sel = [20001400, 20005200, 20005300, 20001800]
    pids_sel = [20002600, 88000002, 88000004, 88000008, 88000010, 88001010,
                88001008, 88100104, 88100114]
    pids_sel = [88000004, 88100114, 88100104, 20001400, 74516435]

    SMALL_SIZE = 8
    MEDIUM_SIZE = 12
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    # pids_sel = None
    sims_nrg, sims_pid = cyNrg.feed_normalization(
        nrmList, simList, norm_opt, ft_opt, pids_sel=pids_sel)

    # convert unit to ms and kNmm
    sims_nrg = sims_nrg * [0.001, 1000, 1000]
    # Delta t
    sims_nrg[:, :, 2] = sims_nrg[:, :, 2] - sims_nrg[:, :, 1]

    c = get_color(len(sims_pid) * 20)
    c = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    fig0, ax = cyNrg.plt_nrg_embed_3d(
        sims_nrg[:], sims_pid[:], simList, m=20, grp=None, cN=c)
    # fig, ax = cyNrg.plt_nrg_embed(sims_nrg[:], sims_pid[:], simList, m=20, plt3=[1, 1, 1], grp=None, grpPid=True, cN=c)
    # fig, ax = cyNrg.plt_nrg_embed(sims_nrg[:], sims_pid[:], simList, m=20, plt3=[1, 0, 0], grp=None, fig=fig, axs=ax)

    ax.view_init(34, -27)
    ax.view_init(11, 146)
    ax.set(
        ylabel='$t_i$', zlabel='$IE_{max}$', xlabel='$\Delta t$')
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)

    fig0.set_size_inches(3.1, 3.1)
    plt.subplots_adjust(left=0.02, right=0.75, top=0.98, bottom=0.06)
    fig0.savefig(dst.format('embd_3d_739_14'))

    fig1 = plt.figure(2)
    plot_nrg_curve(simList, sims_pid, c, grpPid=None)
    # fig1.set_size_inches(4, 4)
    # plt.subplots_adjust(left=0.17, right=0.975, top=0.93, bottom=0.145)
    # plt.axis([0, 120, 0, 3e3])

    plt.legend([
        '$Part_1$',
        '$Part_2$',
        '$Part_3$',
        '$Part_4$',
        '$Part_5$',
    ], prop={'size': 9})
    fig1.set_size_inches(3.1, 3.1)

    ax1 = plt.gca()
    ax1.set_xticks(range(0, 125, 40))
    ax1.set_yticks(range(0, 12000, 4000))
    plt.subplots_adjust(left=0.21, right=0.975, top=0.93, bottom=0.17)
    fig1.savefig(dst.format('nrg_binout_739_14'))

    fig2 = plt.figure(3)
    plot_nrg_curve(simList, sims_pid, c, grpPid=None)
    # fig2.set_size_inches(4, 4)
    # plt.subplots_adjust(left=0.17, right=0.975, top=0.93, bottom=0.145)
    plt.axis([0, 120, 0, 3e3])

    plt.legend([
        '$Part_1$',
        '$Part_2$',
        '$Part_3$',
        '$Part_4$',
        '$Part_5$',
    ], prop={'size': 9})
    fig2.set_size_inches(3.1, 3.1)

    ax2 = plt.gca()
    ax2.set_xticks(range(0, 118, 40))
    ax2.set_yticks(range(0, 3000, 600))
    plt.subplots_adjust(left=0.2, right=0.975, top=0.93, bottom=0.16)

    fig2.savefig(dst.format('nrg_binout_zoom_739_14'))

    plt.show()


def plot_nrg_embd_similarity_cevt2(dst=''):

    ft_opt = ["nrg_max", "ti_grad", "tn_pct"]
    norm_opt = []

    simList = ['cm1e_stv0_004_fp3__001']
    nrmList = '.*'

    # --------------------------------
    pids_sel = [10020100, 55131230, 18620040, 10020210]
    # pids_sel = [10021520, 10020420, 18620120, 18620080, 10022010, 10021350 ]  # , 55131390, 55131400]  # ]

    SMALL_SIZE = 8
    MEDIUM_SIZE = 12
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    # pids_sel = None
    sims_nrg, sims_pid = cyNrg.feed_normalization(
        nrmList, simList, norm_opt, ft_opt, pids_sel=pids_sel)

    # convert unit to ms and kNmm
    sims_nrg = sims_nrg * [1, 1, 1]
    # Delta t
    sims_nrg[:, :, 2] = sims_nrg[:, :, 2] - sims_nrg[:, :, 1]

    c = get_color(len(sims_pid) * 20)
    c = ['b', 'r', 'g', 'y', 'c', 'm', 'k']
    fig0, ax = cyNrg.plt_nrg_embed_3d(
        sims_nrg[:], sims_pid[:], simList, m=20, grp=None, cN=c)
    # fig, ax = cyNrg.plt_nrg_embed(sims_nrg[:], sims_pid[:], simList, m=20, plt3=[1, 1, 1], grp=None, grpPid=True, cN=c)
    # fig, ax = cyNrg.plt_nrg_embed(sims_nrg[:], sims_pid[:], simList, m=20, plt3=[1, 0, 0], grp=None, fig=fig, axs=ax)

    ax.view_init(34, -27)
    ax.view_init(11, 146)
    ax.set(
        ylabel='$t_i$', zlabel='$IE_{max}$', xlabel='$\Delta t$')
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)

    fig0.set_size_inches(3.1, 3.1)
    plt.subplots_adjust(left=0.02, right=0.75, top=0.98, bottom=0.06)
    fig0.savefig(dst.format('embd_3d_fp3_stv0_004'))

    fig1 = plt.figure(2)
    plot_nrg_curve(simList, sims_pid, c, grpPid=None)
    # fig1.set_size_inches(4, 4)
    # plt.subplots_adjust(left=0.17, right=0.975, top=0.93, bottom=0.145)
    # plt.axis([0, 120, 0, 3e3])

    plt.legend([
        '$Part_1$',
        '$Part_2$',
        '$Part_3$',
        '$Part_4$',
        '$Part_5$',
        '$Part_6$',
    ], prop={'size': 8})  #
    fig1.set_size_inches(3.1, 3.1)

    ax1 = plt.gca()
    ax1.set_xticks(range(0, 85, 20))
    ax1.set_yticks(range(0, 5200, 1000))
    plt.subplots_adjust(left=0.21, right=0.975, top=0.93, bottom=0.17)
    fig1.savefig(dst.format('nrg_binout_fp3_stv0_004'))
    # print(dst.format('nrg_binout_fp3_stv0_004'))

    plt.show()


def export_legend(legend, filename="/home/apakiman/Projects/kg01/publication/06_KG_energyAbsorption/images/legend.pdf"):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def plot_nrg_embd_similarity_cevt1(dst):

    ft_opt = ["nrg_max", "ti_grad", "tn_pct"]
    norm_opt = []

    simList = ['cm1e_stv0_004_fp3__001', 'cm1e_stv0_116_fp3__001']
    nrmList = '.*'

    # --------------------------------
    pids_sel = [10020100, 10020210, 55131230]
    # , 10021350 ]  # , 55131390, 55131400]  # ]
    pids_sel = [10021520, 18620120, 10022010]
    SMALL_SIZE = 8
    MEDIUM_SIZE = 11
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    # pids_sel = None
    sims_nrg, sims_pid = cyNrg.feed_normalization(
        nrmList, simList, norm_opt, ft_opt, pids_sel=pids_sel)
    # return

    # convert unit to ms and kNmm
    sims_nrg = sims_nrg * [1, 1, 1]
    # Delta t
    sims_nrg[:, :, 2] = sims_nrg[:, :, 2] - sims_nrg[:, :, 1]

    c = ['c', 'b', 'g', 'm', 'y', 'r']
    fig0, ax = cyNrg.plt_nrg_embed_3d(
        sims_nrg[:], sims_pid[:], simList, m=20, grp=None, cN=c)
    ax0 = plt.gca()
    legend = plt.legend([
        '$Sim_1$  $Part_1$',
        '         $Part_2$',
        '         $Part_3$',
        '$Sim_2$  $Part_1$',
        '         $Part_2$',
        '         $Part_3$',
    ], prop={'size': 11}, bbox_to_anchor=(1.2, 1), framealpha=1, ncol=2, frameon=False)
    # export_legend(legend)
    # plt.show()
    # return
    ax0.get_legend().remove()
    ax0 = plt.gca()

    fig0.set_size_inches(2.2, 2.2)
    plt.subplots_adjust(left=0.04, right=0.67, top=0.98, bottom=0.06)

    ax0.set_yticks(range(10, 13, 2))
    ax0.set_zticks(range(5000, 26000, 10000))
    ax0.set_xticks(range(20, 55, 20))
    ax0.view_init(34, -27)
    ax0.view_init(19, 127)
    ax0.set(
        ylabel='$t_i$', zlabel='$IE_{max}$', xlabel='$\Delta t$')
    ax0.xaxis.set_rotate_label(False)
    ax0.yaxis.set_rotate_label(False)
    ax0.zaxis.set_rotate_label(False)

    # tikz.save(dst.format('embd_3d'), standalone=True)
    # -------------------------------------------------------------
    fig1 = plt.figure(2)
    plot_nrg_curve(simList, sims_pid, c, grpPid=None)
    # plt.xlim([21, 22.5])

    # plt.legend([
    #     '$Sim_1$  $Part_1$',
    #     '         $Part_2$',
    #     '         $Part_3$',
    #     '$Sim_2$  $Part_1$',
    #     '         $Part_2$',
    #     '         $Part_3$',
    # ], prop={'size': 10})
    # plt.legend()
    fig1.set_size_inches(2, 2)
    plt.subplots_adjust(left=0.225, right=0.983, top=0.895, bottom=0.224)

    ax2 = plt.gca()
    ax2.set_xticks(range(00, 85, 20))
    ax2.set_yticks(range(0, 26000, 10000))

    fig0.savefig(dst.format('embd_3d'))
    fig1.savefig(dst.format('nrg_binout'))
    # tikz.save(dst.format('nrg_binout'), standalone=True)

    plt.show()


def plot_nrg_embd_similarity_PAG1(dst):

    ft_opt = ["nrg_max", "ti_grad", "tn_pct"]
    norm_opt = []

    simList = ['ROB_VOWA_739_Design0014', 'ROB_VOWA_739_Design0025']
    nrmList = '.*'

    # --------------------------------
    pids_sel = [88000008, 88000010, 74516435]
    SMALL_SIZE = 8
    MEDIUM_SIZE = 12
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    # pids_sel = None
    sims_nrg, sims_pid = cyNrg.feed_normalization(
        nrmList, simList, norm_opt, ft_opt, pids_sel=pids_sel)
    # return

    # convert unit to ms and kNmm
    sims_nrg = sims_nrg * [0.001, 1000, 1000]
    # Delta t
    sims_nrg[:, :, 2] = sims_nrg[:, :, 2] - sims_nrg[:, :, 1]

    c = ['c', 'b', 'g', 'm', 'y', 'r']
    fig0, ax = cyNrg.plt_nrg_embed_3d(
        sims_nrg[:], sims_pid[:], simList, m=20, grp=None, cN=c)
    ax0 = plt.gca()
    fig0.set_size_inches(3.1, 3.1)
    plt.subplots_adjust(left=0.02, right=0.75, top=0.98, bottom=0.06)

    ax0.set_yticks(range(0, 45, 15))
    ax0.set_zticks(range(1600, 2200, 400))
    ax0.set_xticks(range(5, 90, 30))
    ax0.view_init(34, -27)
    ax0.view_init(11, 146)
    ax0.set(
        ylabel='$t_i$', zlabel='$IE_{max}$', xlabel='$\Delta t$')
    ax0.xaxis.set_rotate_label(False)
    ax0.yaxis.set_rotate_label(False)
    ax0.zaxis.set_rotate_label(False)

    fig1 = plt.figure(2)
    plot_nrg_curve(simList, sims_pid, c, grpPid=None)
    # plt.xlim([21, 22.5])

    plt.legend([
        '$Sim_1$  $Part_1$',
        '         $Part_2$',
        '         $Part_3$',
        '$Sim_2$  $Part_1$',
        '         $Part_2$',
        '         $Part_3$',
    ], prop={'size': 9})
    fig1.set_size_inches(3.1, 3.1)
    plt.subplots_adjust(left=0.21, right=0.975, top=0.93, bottom=0.17)

    ax2 = plt.gca()
    ax2.set_xticks(range(00, 125, 40))
    ax2.set_yticks(range(0, 2200, 600))

    fig0.savefig(dst.format('embd_3d'))
    fig1.savefig(dst.format('nrg_binout'))

    plt.show()


def plot_range_nrg_embd(dst):

    ft_opt = ["nrg_max", "ti_grad", "tn_pct"]
    norm_opt = []

    simList = cyNrg.make_simList(['.*'], ['.*'], 4)
    print(len(simList))
    simList.remove('cm1e_stv0_104_fp3__001')
    simList.remove('cm1e_stv0_105_fp3__001')
    simList.remove('cm1e_stv0_068_fp3__001')
    print(len(simList))
    # return
    simList = simList[:7]
    nrmList = '.*'

    # --------------------------------
    pids_sel = None
    sims_nrg, sims_pid = cyNrg.feed_normalization(
        nrmList, simList, norm_opt, ft_opt, pids_sel=pids_sel)

    c = get_color(len(sims_pid))
    fig0, ax = cyNrg.plt_nrg_embed_3d(
        sims_nrg[:], sims_pid[:], simList, m=20, grp=None, grpSim=True, leg=True, cN=c)

    # fig1 = plt.figure(2)
    # plot_nrg_curve(simList, sims_pid, c, grpPid=True)

    fig0.set_size_inches(7, 4)
    plt.subplots_adjust(left=0.007, right=0.526, top=1, bottom=0)
    fig0.tight_layout()
    # if not os.path.isfile(dst):

    plt.show()


def plot_range_nrg_embd_zoom(dst):

    ft_opt = ["nrg_max", "ti_grad", "tn_pct"]
    norm_opt = []

    simList = ['cm1e_stv0_004_fp3__001', 'cm1e_stv0_006_fp3__001']
    # 'cm1e_stv0_007_fp3__001', 'cm1e_stv0_010_fp3__001',
    # 'cm1e_stv0_011_fp3__001', 'cm1e_stv0_016_fp3__001']
    nrmList = '.*'

    simList = cyNrg.make_simList(['.*'], ['.*'], 4)
    print(len(simList))
    simList.remove('cm1e_stv0_104_fp3__001')
    simList.remove('cm1e_stv0_105_fp3__001')
    simList.remove('cm1e_stv0_068_fp3__001')
    simList.remove('cm1e_stv0_065_fp3__001')
    simList.remove('cm1e_stv0_060_fp3__001')
    simList.remove('cm1e_stv0_059_fp3__001')
    simList.remove('cm1e_stv0_055_fp3__001')
    simList.remove('cm1e_stv0_057_fp3__001')
    print(len(simList))
    # return
    # simList = simList[:7]
    simList = ['cm1e_stv0_001_fp3__001', 'cm1e_stv0_048_fp3__001'

               ]

    # --------------------------------
    # pids_sel = [ 10020420, 10021520]
    # 55021060, 10021520]
    # 10021350, 10021520, 10022010, 55021040, 55021060,
    #             55131390, 55131400]
    # pids_sel = [10021520, 10020420, 18620020, 18620120, 55021040,55021060,10021350, 10022010, 10020210, 10021410]
    pids_sel = None
    # pids_sel = [10020830, 55131230]
    # pids_sel = [10020210, 10021410, 55132510]# , 55132810]
    sims_nrg, sims_pid = cyNrg.feed_normalization(
        nrmList, simList, norm_opt, ft_opt, pids_sel=pids_sel)

    # c = get_color(len(simList)*len(pids_sel))
    c = get_color(len(simList) * 20)
    # pidU = np.unique(sims_pid[:, :20])
    # c = get_color(len(pidU))
    c = ['#466ACF', '#F31B6C', '#3EAAEA', '#C8A52B', '#D1DEF3', '#82F2FE',
         '#54D4F8', '#8D293B', '#140F74', '#AB66D7', '#2BCD98', '#ACD8CB',
         '#B807CF', '#1F19BE', '#C31809', '#D1703C', '#CFA7F5', '#586E21',
         '#CE8436', '#7AE8A7', '#E7CC9D', '#13168B', '#C73A3E', '#DDB9F6',
         '#859185', '#BD97DE', '#3BB8DD', '#A00F8E', '#F99815', '#1BC0A2',
         '#D50DB2', '#0EDDCB', '#8F33A8', '#91FAA7', '#37A28A', '#F48154',
         '#70A522', '#C532FE', '#31FFDC']
    # sims_nrg, sims_pid = cyNrg.fltr_nrg([15000, 0, 0], sims_nrg, sims_pid)
    # fig0, ax = cyNrg.plt_nrg_embed_3d(sims_nrg[:], sims_pid[:], simList, m=20, grp=None, cN=c, grpPid=True)

    fig0, ax = cyNrg.plt_nrg_embed(sims_nrg[:], sims_pid[:], simList, m=20, plt3=[
                                   0, 1, 1], grp=True, grpPid=True, cN=c)
    fig1 = plt.figure(2)
    plot_nrg_curve(simList, sims_pid, c, grpPid=True)

    fig0.set_size_inches(4, 4)
    # plt.subplots_adjust(left=0.095, right=0.9, top=0.88, bottom=0.11)
    # fig0.set_size_inches(1.6, 1.6)
    # if not os.path.isfile(dst):
    fig0.savefig(dst)

    plot_nrg_curve(simList, sims_pid, c)  # , groupPid=True)

    # print(np.array(sims_pid))
    # fig1 = plt.figure(2)
    # plot_nrg_curve(simList, sims_pid, c) #, grpPid=True)
    plt.show()


def plot_ti_selection():

    ft_opti = ["nrg_max", "ti_ll", "tn_pct"]
    ft_opt1 = ["nrg_max", "ti_grad", "tn_pct"]
    norm_opt = []
    saveP = "../publication/06_KG_energyAbsorption/images/plot/{}"
    # saveP = "plot/"

    simList = cyNrg.make_simList(['.*'], ['.*'], 4)
    simList = simList[:10]
    nrmList = '.*'

    pids_sel = [20005300, 20001800]
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    sims_nrg0, sims_pid0 = cyNrg.feed_normalization(
        nrmList, simList, norm_opt, ft_opt0, pids_sel=pids_sel)
    sims_nrg1, sims_pid1 = cyNrg.feed_normalization(
        nrmList, simList, norm_opt, ft_opt1, pids_sel=pids_sel)

    # fig1, ax = cyNrg.plt_nrg_embed(sims_nrg1[:], sims_pid1[:], simList, m=20, plt3=[1, 0, 0], grp=None, grpPid=True, OEM=OEM)
    # plot_nrg_curve(simList, sims_pid0, ['b', 'r'], grpPid=True)
    # fig1.set_size_inches(3.5, 3.5)
    # plt.subplots_adjust(left=0.2, right=0.975, top=0.93, bottom=0.16)
    # fig1.savefig(saveP.format('ti_grad_binout_embd.pdf'))
    #
    # fig0, ax = cyNrg.plt_nrg_embed(sims_nrg0[:], sims_pid0[:], simList, m=20, plt3=[1, 0, 0], grp=None, grpPid=True, OEM=OEM)
    # plot_nrg_curve(simList, sims_pid0, ['b', 'r'], grpPid=True)
    # fig0.set_size_inches(3.5, 3.5)
    # plt.subplots_adjust(left=0.2, right=0.975, top=0.93, bottom=0.16)
    # fig0.savefig(saveP.format('ti_ll_binout_embd.pdf'))
    #
    fig2 = plt.figure()
    plot_nrg_curve(simList, sims_pid0, ['b', 'r'], grpPid=True)
    plt.xlim([4, 75])
    plt.ylim([-10, 120])
    fig2.set_size_inches(3.5, 3.5)
    plt.subplots_adjust(left=0.2, right=0.975, top=0.93, bottom=0.16)
    fig2.savefig(saveP.format('ti_zoom.pdf'))

    plt.show()


def plot_ti_selection_sum():

    ft_opt0 = ["nrg_max", "ti_ll", "tn_pct"]
    ft_opt1 = ["nrg_max", "ti_grad", "tn_pct"]
    norm_opt = []
    saveP = "../publication/06_KG_energyAbsorption/images/plot/{}"
    # saveP = "plot/"

    simList = cyNrg.make_simList(['.*'], ['.*'], 4)
    simList = simList[:1]
    nrmList = '.*'

    pids_sel = [20005300, 20001800]
    SMALL_SIZE = 8
    MEDIUM_SIZE = 12
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    sims_nrg0, sims_pid0 = cyNrg.feed_normalization(
        nrmList, simList, norm_opt, ft_opt0, pids_sel=pids_sel)
    sims_nrg1, sims_pid1 = cyNrg.feed_normalization(
        nrmList, simList, norm_opt, ft_opt1, pids_sel=pids_sel)

 # fig1
    # fig, ax = plt.subplots(2, 1)
    # ax2 = plt.axes(ax[0])
    # plot_nrg_curve(simList, sims_pid0, ['k--', 'k'], grpPid=False)
    # a0, B0 = (sims_nrg0[0][:, 1]) * 1000
    # y = 70
    #
    # plt.plot([a0, a0], [-8, y], 'limegreen', linewidth=1)
    # ax2.annotate('$t_{i1,i2}$', xy=(a0 + 3, -4), color='limegreen')
    # plt.plot([B0, B0], [-8, y], 'limegreen', linewidth=1)
    # ax2.set_xticks(range(20, 90, 20))
    # ax2.set_yticks(range(30, 90, 30))
    # ax2.ticklabel_format(axis='y', style='plain')
    # plt.xlim([30, 78])
    # plt.ylim([-10, 80])
    # plt.ylabel('')
    # plt.xlabel('')
    #
    # ax3 = plt.axes(ax[1])
    # plot_nrg_curve(simList, sims_pid0, ['k--', 'k'], grpPid=False)
    # a, B = (sims_nrg1[0][:, 1]) * 1000
    # y = 70
    # plt.plot([a, a], [-8, y], 'limegreen', linewidth=1)
    # ax3.annotate('$t_{i1}$', xy=(a + 2, -5), color='limegreen')
    # plt.plot([B, B], [-8, y], 'limegreen', linewidth=1)
    # ax3.annotate('$t_{i2}$', xy=(B + 2, -5), color='limegreen')
    # ax3.set_xticks(range(20, 90, 20))
    # ax3.set_yticks(range(30, 90, 30))
    # ax3.ticklabel_format(axis='y', style='plain')
    # plt.xlim([30, 78])
    # plt.ylim([-10, 80])
    # fig.set_size_inches(3.5, 3.5)
    # plt.subplots_adjust(left=0.1, right=0.975, top=0.93, bottom=0.16)
    # plt.ylabel('')
    # fig.savefig(saveP.format('ti_ll_grad_zoom.pdf'))

 # fig1_edit
    fig2, ax = plt.subplots()
    plot_nrg_curve(simList, sims_pid0, ['k--', 'k'], grpPid=False)
    a, B = (sims_nrg1[0][:, 1]) * 1000
    y = 70

    plt.plot([a, a], [-8, y], 'limegreen', linewidth=1)
    ax.annotate('$t_{i1}$', xy=(a + 2, -5), color='limegreen')
    plt.plot([B, B], [-8, y], 'limegreen', linewidth=1)
    ax.annotate('$t_{i2}$', xy=(B + 2, -5), color='limegreen')
    ax.set_xticks(range(20, 90, 20))
    ax.set_yticks(range(30, 90, 30))
    ax.ticklabel_format(axis='y', style='plain')
    plt.xlim([30, 78])
    plt.ylim([-10, 80])
    fig2.set_size_inches(3.1, 3.1)
    plt.subplots_adjust(left=0.1, right=0.975, top=0.93, bottom=0.16)
    # plt.ylabel('')
    fig2.savefig(saveP.format('ti_ll_grad_zoom_02.pdf'))

 # fig2
    fig4, ax4 = plt.subplots()
    plot_nrg_curve(simList, sims_pid0, ['k--', 'k'], grpPid=False)
    plt.legend(['part 1', 'part 2'])
    ax4.ticklabel_format(axis='y', style='sci', scilimits=[0, 1])
    fig4.set_size_inches(3.1, 3.1)
    ax4.set_xticks(range(0, 125, 40))
    ax4.set_yticks(np.arange(0, 5e3, 2e3))
    plt.subplots_adjust(left=0.2, right=0.975, top=0.93, bottom=0.16)
    fig4.savefig(saveP.format('ti_ll_grad.pdf'))

    plt.show()


def plot_range_nrg_embd_2d(dst):

    ft_opt = ["nrg_max", "ti_grad", "tn_pct"]
    norm_opt = []

    simList = cyNrg.make_simList(['.*stv0.*fp3.*'], ['.*'], 4)
    simList.remove('cm1e_stv0_104_fp3__001')
    simList.remove('cm1e_stv0_105_fp3__001')
    simList.remove('cm1e_stv0_068_fp3__001')
    simList = simList[:]
    nrmList = '.*'

    # --------------------------------
    pids_sel = None
    sims_nrg, sims_pid = cyNrg.feed_normalization(
        nrmList, simList, norm_opt, ft_opt, pids_sel=pids_sel)

    c = get_color(len(sims_pid))
    # fig0, ax = cyNrg.plt_nrg_embed(sims_nrg[:], sims_pid[:], simList, m=3, plt3=[1, 1, 1], grp=True)
    fig0, ax = cyNrg.plt_nrg_embed(sims_nrg[:], sims_pid[:], simList, m=20, plt3=[
                                   0, 1, 1], grp=True, grpPid=True)

    # fig0.set_size_inches(8, 4)
    # plt.subplots_adjust(left=0.1, right=0.93, top=0.929, bottom=0.15, wspace=0.45)
    # plt.subplots_adjust(left=0.1, right=0.81, top=0.929, bottom=0.15, wspace=0.345)
    # fig0.tight_layout()
    # if not os.path.isfile(dst):
    # fig0.savefig(dst)

    plt.show()


def plot_range_nrg_embd_2d_ord(dst):

    ft_opt = ["nrg_max", "ti_grad", "tn_pct"]
    norm_opt = []

    simList = cyNrg.make_simList(['.*stv0.*fp3.*'], ['.*'], 4)
    simList.remove('cm1e_stv0_104_fp3__001')
    simList.remove('cm1e_stv0_105_fp3__001')
    simList.remove('cm1e_stv0_068_fp3__001')
    simList = simList[:]
    nrmList = '.*'

    # --------------------------------
    pids_sel = None
    sims_nrg, sims_pid = cyNrg.feed_normalization(
        nrmList, simList, norm_opt, ft_opt, pids_sel=pids_sel)

    c = get_color(len(sims_pid))
    fig0, ax = cyNrg.plt_nrg_embed(sims_nrg[:], sims_pid[:], simList, m=3, plt3=[
                                   1, 1, 1], grp=True, grpOrd=True)

    # fig0.set_size_inches(8, 4)
    # plt.subplots_adjust(left=0.1, right=0.93, top=0.929, bottom=0.15, wspace=0.45)
    # plt.subplots_adjust(left=0.1, right=0.81, top=0.929, bottom=0.15, wspace=0.345)
    # fig0.tight_layout()
    # if not os.path.isfile(dst):
    # fig0.savefig(dst)

    plt.show()


def plot_range_nrg_embd_2d_PAG(dst):

    ft_opt = ["nrg_max", "ti_grad", "tn_pct"]
    norm_opt = []

    simList = cyNrg.make_simList(['.*'], ['.*'], 4)
    simList = simList[:]
    print(len(simList))
    nrmList = '.*'

    # --------------------------------
    pids_sel = None
    sims_nrg, sims_pid = cyNrg.feed_normalization(
        nrmList, simList, norm_opt, ft_opt, pids_sel=pids_sel)

    c = get_color(len(sims_pid))
    fig0, ax = cyNrg.plt_nrg_embed(sims_nrg[:], sims_pid[:], simList, m=3, plt3=[
                                   1, 1, 1], grp=True, grpPid=True)
    # fig0, ax = cyNrg.plt_nrg_embed(sims_nrg[:], sims_pid[:], simList, m=3, plt3=[1, 1, 1], grp=None, grpPid=True)

    fig0.set_size_inches(8, 4)
    plt.subplots_adjust(left=0.1, right=0.81, top=0.929,
                        bottom=0.15, wspace=0.345)
    # plt.subplots_adjust(left=0.1, right=0.93, top=0.929, bottom=0.15, wspace=0.45)
    # fig0.tight_layout()
    # if not os.path.isfile(dst):
    # fig0.savefig(dst)

    plt.show()


def plot_embd_binout_cevt():

    ft_opt = ["nrg_max", "ti_grad", "tn_pct"]
    norm_opt = []

    simList = [
        'cm1e_stv0_001_fp3__001', 'cm1e_stv0_048_fp3__001',
        'cm1e_stv0_137_fp3__001']
    # simList = cyNrg.make_simList(['.*'], ['.*'], 4)
    nrmList = '.*'

    pids_sel = [55132830, 55131230, 10020830, ]

    sims_nrg, sims_pid = cyNrg.feed_normalization(
        nrmList, simList, norm_opt, ft_opt, pids_sel=pids_sel)

    c = ['b', 'r', 'g', 'c', 'm', 'y', 'k', '#4AFFB5', '#593F50']
    fig0, ax = cyNrg.plt_nrg_embed_3d(
        sims_nrg[:], sims_pid[:], simList, m=20, grp=None, cN=c)
    # fig, ax = cyNrg.plt_nrg_embed(sims_nrg[:], sims_pid[:], simList, m=20, plt3=[1, 1, 1], grp=None, grpPid=True, cN=c)
    # fig, ax = cyNrg.plt_nrg_embed(sims_nrg[:], sims_pid[:], simList, m=20, plt3=[1, 0, 0], grp=None, fig=fig, axs=ax)

    plt.figure(2)
    plot_nrg_curve(simList, sims_pid, c, grpPid=None)

    plt.show()


class plot_YARIS():

    def out_dataframe(slef):
        ft_opt = ["nrg_max", "ti_grad", "tn_pct"]
        norm_opt = []

        simList = cyNrg.make_simList(['.*'], ['.*'], 4)
        pids_sel = None
        nrmList = '.*'

        sims_nrg, sims_pid = cyNrg.feed_normalization(
            nrmList, simList, norm_opt, ft_opt, pids_sel=pids_sel)
        simListAbb = [s.split('_')[2] for s in simList]


class plot_CEVT():

    def __init__(self):
        self.cevt_Data = {"release": {
            'stcr': {
                'fp3': {
                    "errList": ['cm1e_stcr_173_fp3__001']
                },
                'fod': {
                    "errList": []
                },
                'fo5': {
                    "errList": [
                        'cm1e_stcr_386_fo5__001',
                        'cm1e_stcr_237_fo5__001',
                    ]
                }
            },
            'stv0': {
                'fp3': {
                    "errList": [
                        'cm1e_stv0_104_fp3__001',
                        'cm1e_stv0_105_fp3__001',
                        'cm1e_stv0_095_fp3__001',
                        'cm1e_stv0_068_fp3__001',
                        'cm1e_stv0_065_fp3__001',
                        'cm1e_stv0_060_fp3__001',
                        'cm1e_stv0_059_fp3__001',
                        'cm1e_stv0_055_fp3__001',
                        'cm1e_stv0_057_fp3__001',
                    ]
                },
                'fod': {
                    "errList": ['cm1e_stv0_044_fod__001']
                },
                'fo5': {
                    "errList": ['cm1e_stv0_103_fo5__001']
                }
            }
        }}

    def out_dataframe(self, rl, lc):

        ft_opt = ["nrg_max", "ti_grad", "tn_pct"]
        norm_opt = []
        self.rl = rl
        self.lc = lc
        self.errList = self.cevt_Data['release'][self.rl][self.lc]['errList']

        simList = cyNrg.make_simList([
            '.*{0}.*{1}_.*'.format(self.rl, self.lc)], ['.*'], 4)
        for s in self.errList:
            simList.remove(s)  # early termination

        # simList = simList[:3]
        pids_sel = None
        # pids_sel = [10021520, 10020420]

        nrmList = '.*'

        sims_nrg, sims_pid = cyNrg.feed_normalization(
            nrmList, simList, norm_opt, ft_opt, pids_sel=pids_sel)
        simListAbb = [s.split('_')[2] for s in simList]

        c_sim = get_color(len(simList))
        pidU = np.unique(sims_pid)
        c_pid = get_color(len(pidU))
        c_ord = np.repeat(
            [get_color(sims_pid.shape[1])],
            len(simList), axis=0)
        # c = ['b', 'r', 'g', 'c', 'm', 'y', 'k', '#4AFFB5', '#593F50']

        df = pd.DataFrame()
        pidU = np.unique(sims_pid)
        # for j, sj in enumerate(simListAbb):
        for i, pi in enumerate(pidU):
            id = np.where(sims_pid == pi)
            dfi = pd.DataFrame(sims_nrg[id], columns=['IE', 'ti', 'tn'])
            dfi['PID'] = int(pi)
            dfi['sim'] = np.array(simList)[id[0]]
            dfi['sim_abb'] = np.array(simListAbb)[id[0]]
            dfi['c_grPID'] = c_pid[i]
            dfi['c_grSim'] = np.array(c_sim)[id[0]]
            dfi['c_grOrd'] = c_ord[id]

            df = df.append(dfi)
            df = df.reset_index(drop=True)
        return(df)

    def nrg_embd_2d_ord(self, axs, fig):
        ft_opt = ["nrg_max", "ti_grad", "tn_pct"]
        norm_opt = []

        simList = cyNrg.make_simList([
            '.*{0}.*{1}_.*'.format(self.rl, self.lc)], ['.*'], 4)
        for s in self.errList:
            simList.remove(s)  # early termination
        nrmList = '.*'

        # simList = simList[10:]
        pids_sel = None

        sims_nrg, sims_pid = cyNrg.feed_normalization(
            nrmList, simList, norm_opt, ft_opt, pids_sel=pids_sel)

        fig, ax = cyNrg.plt_nrg_embed(
            sims_nrg[:], sims_pid[:], simList,
            m=5, plt3=[0, 1, 0], grp=None, grpOrd=True)

    def nrg_embd_2d(self, axs, fig):
        ft_opt = ["nrg_max", "ti_grad", "tn_pct"]
        norm_opt = []

        simList = cyNrg.make_simList([
            '.*{0}.*{1}_.*'.format(self.rl, self.lc)], ['.*'], 4)

        for s in self.errList:
            simList.remove(s)  # early termination

        nrmList = '.*'

        # simList = simList[10:]
        pids_sel = None

        sims_nrg, sims_pid = cyNrg.feed_normalization(
            nrmList, simList, norm_opt, ft_opt, pids_sel=pids_sel)

        fig, ax = cyNrg.plt_nrg_embed(
            sims_nrg[:], sims_pid[:], simList,
            m=5, plt3=[0, 1, 0], grp=None, grpPid=True, axs=[axs], fig=fig)
        axs.set_title('CEVT {0} {1}'.format(self.rl, self.lc))

    def nrg_embd_3d(self):
        ft_opt = ["nrg_max", "ti_grad", "tn_pct"]
        norm_opt = []

        simList = cyNrg.make_simList([
            '.*{0}.*{1}_.*'.format(self.rl, self.lc)], ['.*'], 4)

        for s in self.errList:
            simList.remove(s)

        nrmList = '.*'

        # simList = simList[10:]
        pids_sel = None

        sims_nrg, sims_pid = cyNrg.feed_normalization(
            nrmList, simList, norm_opt, ft_opt, pids_sel=pids_sel)

        pidU = np.unique(sims_pid)
        # c = ['b', 'r', 'g', 'c', 'm', 'y', 'k', '#4AFFB5', '#593F50']
        c = get_color(len(pidU))

        fig = px.scatter_3d(x=[0, 1, 2, 3, 4], y=[
                            0, 1, 4, 9, 16], z=[0, 1, 4, 9, 16])
        fig.show()
        # cyNrg.plt_nrg_embed_3d(
        #     sims_nrg[:], sims_pid[:], simList, m=20, grpPid=True, cN=c, grp=None)

    def dna(self):
        fig, axs = plt.subplots(2, 3)

        for ri, r in enumerate(self.cevt_Data['release']):
            self.rl = r
            for li, l in enumerate(self.cevt_Data['release'][r]):
                self.lc = l
                self.errList = self.cevt_Data['release'][r][l]['errList']
                self.nrg_embd_2d(axs[ri][li], fig)

    def sym(self, rl, lc):

        self.rl = rl
        self.lc = lc
        self.errList = self.cevt_Data['release'][self.rl][self.lc]['errList']

        self.nrg_embd_3d()

    def plot_symmetry_parts_cevt_old():

        ft_opt = ["nrg_max", "ti_grad", "tn_pct"]
        norm_opt = []

        simList = cyNrg.make_simList(['.*'], ['.*'], 4)
        nrmList = '.*'
        simList.remove('cm1e_stv0_104_fp3__001')
        simList.remove('cm1e_stv0_105_fp3__001')
        simList.remove('cm1e_stv0_095_fp3__001')
        simList.remove('cm1e_stv0_068_fp3__001')
        simList.remove('cm1e_stv0_065_fp3__001')
        simList.remove('cm1e_stv0_060_fp3__001')
        simList.remove('cm1e_stv0_059_fp3__001')
        simList.remove('cm1e_stv0_055_fp3__001')
        simList.remove('cm1e_stv0_057_fp3__001')

        # print(simList[10:])
        # simList = simList[10:]
        pids_sel = [10021520, 10020420]

        sims_nrg, sims_pid = cyNrg.feed_normalization(
            nrmList, simList, norm_opt, ft_opt)  # , pids_sel=pids_sel)

        c_sim = get_color(len(simList) * 2)
        pidU = np.unique(sims_pid)
        c_pid = get_color(len(pidU))
        c = c_pid
        print(len(c))
        # c = ['b', 'r', 'g', 'c', 'm', 'y', 'k', '#4AFFB5', '#593F50']
        fig0, ax = cyNrg.plt_nrg_embed_3d(
            sims_nrg[:], sims_pid[:], simList, m=20, grpPid=True, cN=c, grp=None)
        # fig, ax = cyNrg.plt_nrg_embed(
        #     sims_nrg[:], sims_pid[:], simList,
        #     m=20, plt3=[1, 1, 1], grp='Yes', grpPid=None, cN=c)
        # fig, ax = cyNrg.plt_nrg_embed(
        #     sims_nrg[:], sims_pid[:], simList,
        #     m=20, plt3=[1, 0, 0], grp=None, fig=fig, axs=ax)

        # plt.figure(2)
        # plot_nrg_curve(simList, sims_pid, c, grpPid=None)
        return()


def plot_nrg_embd_2d_yaris_sub():
    '''
    view fo3 8411 2059639 7903

    '''

    ft_opt = ["nrg_max", "ti_grad", "tn_pct"]
    norm_opt = []

    simLists = [
        # ['CCSA_submodel_0004'],
        # ['CCSA_submodel_0004', 'CCSA_submodel_0006'],
        # # ['CCSA_submodel_0004', 'CCSA_submodel_0007'],
        # ['CCSA_submodel_0006', 'CCSA_submodel_0007'],
        # ['CCSA_submodel_0005', 'CCSA_submodel_0006'],
        ['CCSA_submodel_0004', 'CCSA_submodel_0005', 'CCSA_submodel_0006']

    ]
    dsts = [
        # # '../publication/06_KG_energyAbsorption/images/plot/submodel_0004.pdf',
        # '../publication/06_KG_energyAbsorption/images/plot/submodel_0004.svg',
        # '../publication/06_KG_energyAbsorption/images/plot/submodel_00046.pdf',
        # # '../publication/06_KG_energyAbsorption/images/plot/submodel_00047.pdf',
        # '../publication/06_KG_energyAbsorption/images/plot/submodel_00067.pdf',
        # '../publication/06_KG_energyAbsorption/images/plot/submodel_00056.pdf',
        '../publication/06_KG_energyAbsorption/images/plot/submodel_000456.pdf'
    ]
    cs = [
        # ['b'],
        # ['b', 'r'],
        # # ['b', 'lime'],
        # ['r', 'lime'],
        # ['y', 'r'],
        ['b', 'y', 'r'],
    ]
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    for sub, simList in enumerate(simLists):
        dst = dsts[sub]
        simList.sort()
        simListAbb = ['00' + str(int(s.split('_')[-1]) - 3) for s in simList]
        # simListAbb = [s.split('_')[-1] for s in simList]
        nrmList = '.*'

        pids_sel = [2000001, 2000501, 2000502, 2000002, 2000000]
        mDic = {"2000001": 4, "2000501": 5,
                "2000502": 5, "2000002": 4, "2000000": 's'}

        sims_nrg, sims_pid = cyNrg.feed_normalization(
            nrmList, simList, norm_opt, ft_opt, pids_sel=pids_sel)

        # convert unit to ms and kNmm
        sims_nrg = sims_nrg * [0.001, 1000, 1000]

        fig, ax = cyNrg.plt_nrg_embed_simple(
            # fig, ax = cyNrg.plt_nrg_embed(
            sims_nrg[:], sims_pid[:], simListAbb,
            m=20, plt3=[0, 1, 0], grp=None, grpPid=False,
            cN=cs[sub], marker=mDic)
        plot_center_embd(sims_nrg, cs[sub], simListAbb)
        # plt.xlim([21, 22.5])
        plt.xlim([6, 24])

        fig.set_size_inches(3.1, 3.1)
        plt.subplots_adjust(left=0.2, right=0.95, top=0.93, bottom=0.16)
        plt.legend(loc=2)
        # plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

        fig.savefig(dst)
    plt.show()
    # break
    # return()


def plot_nrg_embd_2d_yaris_sub_rev():
    '''
    view fo3 8411 2059639 7903

    '''

    ft_opt = ["nrg_max", "ti_grad", "tn_pct"]
    norm_opt = []

    simLists = [
        ['CCSA_submodel_6003'],
        ['CCSA_submodel_6003', 'CCSA_submodel_6030'],
        ['CCSA_submodel_6030', 'CCSA_submodel_6031'],
        ['CCSA_submodel_6030', 'CCSA_submodel_6060'],
        ['CCSA_submodel_6003', 'CCSA_submodel_6030',
            'CCSA_submodel_6031', 'CCSA_submodel_6060'],
        # ['CCSA_submodel_0005', 'CCSA_submodel_0006'],
        # ['CCSA_submodel_6003', 'CCSA_submodel_6030', 'CCSA_submodel_6031']

    ]
    dsts = [
        '../publication/06_KG_energyAbsorption/submition/submodel_6003.svg',
        '../publication/06_KG_energyAbsorption/submition/submodel_600br.pdf',
        '../publication/06_KG_energyAbsorption/submition/submodel_600rg.pdf',
        '../publication/06_KG_energyAbsorption/submition/submodel_600ry.pdf',
        '../publication/06_KG_energyAbsorption/submition/submodel_6brgy.pdf',
        # '../publication/06_KG_energyAbsorption/submition/submodel_00056.pdf',
        # '../publication/06_KG_energyAbsorption/submition/submodel_000456.pdf'
    ]
    cs = [
        ['b'],
        ['b', 'r'],
        # ['b', 'lime'],
        ['r', 'lime'],
        ['r', 'y'],
        ['b', 'r', 'lime', 'y'],
    ]
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    for sub, simList in enumerate(simLists):
        dst = dsts[sub]
        simList.sort()
        simListAbb = [str(int(s.split('_')[-1])-6000) for s in simList]
        # simListAbb = [s.split('_')[-1] for s in simList]
        nrmList = '.*'

        pids_sel = [2000001, 2000501, 2000502, 2000002, 2000000]
        mDic = {"2000001": 4, "2000501": 5,
                "2000502": 5, "2000002": 4, "2000000": 's'}

        sims_nrg, sims_pid = cyNrg.feed_normalization(
            nrmList, simList, norm_opt, ft_opt, pids_sel=pids_sel)

        # convert unit to ms and kNmm
        sims_nrg = sims_nrg * [0.001, 1000, 1000]

        fig, ax = cyNrg.plt_nrg_embed_simple(
            # fig, ax = cyNrg.plt_nrg_embed(
            sims_nrg[:], sims_pid[:], simListAbb,
            m=20, plt3=[0, 1, 0], grp=None, grpPid=False,
            cN=cs[sub], marker=mDic)
        plot_center_embd(sims_nrg, cs[sub], simListAbb)
        # plt.xlim([21, 22.5])
        plt.xlim([6, 24])

        fig.set_size_inches(3.1, 3.1)
        plt.subplots_adjust(left=0.2, right=0.95, top=0.93, bottom=0.16)
        plt.legend(loc=2)
        # plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

        fig.savefig(dst)
    plt.show()
    # break
    # return()


def plot_nrg_embd_2d_yaris_sub_9part():
    '''
    view fo3 8411 2059639 7903

    '''

    ft_opt = ["nrg_max", "ti_grad", "tn_pct"]
    norm_opt = []

    simLists = [
        ['CCSA_submodel_0004'],
        ['CCSA_submodel_0004', 'CCSA_submodel_0006'],
        ['CCSA_submodel_0004', 'CCSA_submodel_0007'],
        ['CCSA_submodel_0006', 'CCSA_submodel_0007'],
        ['CCSA_submodel_0005', 'CCSA_submodel_0006'],
        ['CCSA_submodel_0004', 'CCSA_submodel_0005', 'CCSA_submodel_0006']
    ]
    dsts = [
        '../publication/06_KG_energyAbsorption/images/plot/submodel_9part_0004.pdf',
        '../publication/06_KG_energyAbsorption/images/plot/submodel_9part_00046.pdf',
        '../publication/06_KG_energyAbsorption/images/plot/submodel_9part_00047.pdf',
        '../publication/06_KG_energyAbsorption/images/plot/submodel_9part_00067.pdf',
        '../publication/06_KG_energyAbsorption/images/plot/submodel_9part_00056.pdf',
        '../publication/06_KG_energyAbsorption/images/plot/submodel_9part_000456.pdf'
    ]
    cs = [
        ['b'],
        ['b', 'r'],
        ['b', 'lime'],
        ['r', 'lime'],
        ['y', 'r'],
        ['b', 'y', 'r'],
    ]
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    for sub, simList in enumerate(simLists):
        dst = dsts[sub]
        simList.sort()
        simListAbb = ['00' + str(int(s.split('_')[-1]) - 3) for s in simList]
        # simListAbb = [s.split('_')[-1] for s in simList]
        nrmList = '.*'

        pids_sel = [2000001, 2000501, 2000502, 2000002,
                    2000511, 2000512, 2000011, 2000012, 2000000]
        mDic = {
            "2000001": 4,
            "2000002": 4,
            "2000011": "3",
            "2000012": "3",
            "2000501": 5,
            "2000502": 5,
            "2000511": "4",
            "2000512": "4",
            "2000000": '*'
        }

        sims_nrg, sims_pid = cyNrg.feed_normalization(
            nrmList, simList, norm_opt, ft_opt, pids_sel=pids_sel)

        # convert unit to ms and kNmm
        sims_nrg = sims_nrg * [0.001, 1000, 1000]

        fig, ax = cyNrg.plt_nrg_embed_simple(
            # fig, ax = cyNrg.plt_nrg_embed(
            sims_nrg[:], sims_pid[:], simListAbb,
            m=20, plt3=[0, 1, 0], grp=None, grpPid=False,
            cN=cs[sub], marker=mDic)
        plot_center_embd(sims_nrg, cs[sub], simListAbb)
        # plt.xlim([21, 22.5])

        # fig.set_size_inches(3.1, 3.1)
        # plt.subplots_adjust(left=0.2, right=0.95, top=0.93, bottom=0.16)

        # ## plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

        fig.savefig(dst)
    # plt.show()
        # break
    return()


def plot_nrg_embd_2d_yaris_sub_debug_tn():
    ft_opt = ["nrg_max", "ti_grad", "tn_pct"]
    norm_opt = []

    simList = ['CCSA_submodel_0006', 'CCSA_submodel_0007']
    # , 'CCSA_submodel_0006']#, 'CCSA_submodel_0007']

    simList.sort()
    simListAbb = [s.split('_')[-1] for s in simList]
    nrmList = '.*'

    pids_sel = [2000011]  # , 2000511]
    pids_sel = None
    pids_sel = [2000001, 2000501, 2000502, 2000002, 2000011, 2000511,
                2000012, 2000512]
    # pids_sel = [2000001, 2000501, 2000502, 2000002]
    c = get_color(len(simList) * 20)

    sims_nrg, sims_pid = cyNrg.feed_normalization(
        nrmList, simList, norm_opt, ft_opt, pids_sel=pids_sel)

    # fig, ax = cyNrg.plt_nrg_embed(
    #     sims_nrg[:], sims_pid[:], simListAbb,
    #     m=20, plt3=[0, 1, 0], grp=None, grpPid=False)
    # plot_center_embd(sims_nrg, c, simListAbb)

    # fig.set_size_inches(10, 10)
    # plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    # plt.subplots_adjust(left=0.17, right=0.975, top=0.93, bottom=0.145)

    cyNrg.out_dataframe(sims_nrg, sims_pid, simListAbb)
    # plot_nrg_curve(simList, sims_pid, c, grpPid=None)
    return()


def plot_center_embd(sims, c, names):
    for i, s in enumerate(sims):
        x, y, z = np.mean(s, axis=0)

        plt.scatter(z, x, c=c[i], s=20, label=names[i])
        for v in s:
            xx, yy, zz = v
            plt.plot([z, zz], [x, xx], c=c[i], linewidth=0.1)
            plt.legend(loc=7)


def get_color(n):
    colors = []
    for i in range(n):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    return(colors)


def plot_nrg_curve(simList, pids, c, grpPid=None):
    sim, src_path, simList = setProject(OEM, simList)
    ii = 0
    for i, s in enumerate(simList):
        pidU = np.unique(pids[i])
        binPath = src_path.format(s)
        sPath = src_path.format(s)
        eval('sim.data' + OEM + '(sPath)')
        print(sim)
        t, ids, curv, nrgFtr = kg.energy_feature(sim, out=True)
        if OEM == 'PAG':
            t = t * 1000  # to ms
            curv = curv / 1000  # to kNmm

        nrg_max = []
        for p, pid in enumerate(pids[i]):
            # for p, pid in enumerate(pidU):
            # print(int(pid))
            idF = np.isin(nrgFtr.pid, pid)
            v = nrgFtr.tnPct[idF]
            # plt.plot([v, v], [0, 2e6], c=c[ii])
            nrg_max.append(nrgFtr.nrgMax[idF])
            id = np.isin(ids, pid)
            if pid == 0:
                ii += 1
                continue
            if grpPid:
                # print(t.shape, curv.shape, pid)
                plt.plot(t, curv[:, id], c[p], linewidth=1)
            else:
                plt.plot(t, curv[:, id], c[ii], label=str(
                    int(pid)) + '-' + s.replace('Design00', ''), linewidth=1)
                ii += 1
            # plt.legend()
        if grpPid:
            plt.legend([str(int(x)) for x in pidU])
        plt.xlabel('$t\ [ms]$')
        plt.ylabel('$IE_{part}\ [kNmm]$')
        ax = plt.gca()
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    return


def setProject(OEM, simList):
    sim = kg.CaeSim(OEM)
    print(sim)
    if OEM == 'CEVT':
        lc = 'fp3'
        rls = 'stv0'
        src_path = '../src/pic/CEVT/cm1e_stv0_{}_fp3__001'
        src_path = ('/export/work/anahita/CEVT/runs/cm1e_{0}_'.format(rls)
                    + '{}' + '_{0}__001'.format(lc))
        src_path = ('../dash-nrg/assets/CEVT/CEVT/Rep/3_stv0/fp3/runs/cm1e_{0}_'.format(
            rls) + '{}' + '_{0}__001'.format(lc))
        # simList = ['001', '055']
        simList = [s.split('_')[2] for s in simList]
    elif OEM == 'PAG':
        src_path = '/export/PAG_DATA/VORDERWAGEN/002_Daten_Kaan/ROB_VOWA_739/{}'
        src_path = '/export/PAG_DATA/VORDERWAGEN/002_Daten_Kaan/ROB_VOWA_739/{}'
        simList = [s.split('_')[-1] for s in simList]
    elif OEM == 'YARIS':
        # '/OUTPUT/binout*'
        src_path = '/home/apakiman/leo1/Projects/carGraph/runs/YARIS/full_front/CCSA_submodel/crash_modes/{}'

    return(sim, src_path, simList)


def plot_doe_lc_ord():
    import seaborn as sns
    lcs = ['fp3']  # , 'fo5', 'fod_']
    rls = ['stv0_', 'stcr', 'stv03']
    # rls = ['stv03', 'm1']
    sns.set(color_codes=False)
    c_grp = 'c_grPID'
    c_grp = 'c_grOrd'
    ords = 8
    ns = 100
    nPid = 8
    pids = ''

    colors = px.colors.cyclical.IceFire_r[0::2]
    colors = [
        '#000000', '#820000', '#c65400', '#e7b000', '#9be4ef',
        # '#f3d573',
        # '#e1e9d1',
        '#54c8df', '#217eb8', '#003786', '#000000']

    for lc in lcs:
        for rl in rls:
            dst = '../publication/06_KG_energyAbsorption/images/plot/doe_{0}_{1}_ord.svg'.format(
                lc, rl)
            sims = '.*{1}.*{0}.*'.format(lc, rl)
            oem = oems.oems('CEVT')
            df1 = oem.cypher().out_dataframe(
                ns=int(ns), nPID=int(nPid), nOrd=ords, regs=sims, regp=pids)

            df1 = df1.sort_values(by=[c_grp])
            df1['IE'] = df1['IE'] / 1000  # kNmm
            # fig1 = px.scatter_3d(
            # , x="tn", y="ti", z="IE", color=c_grp, custom_data=["pic"],
            # hover_name="PID",)
            # df1[c_grp] = df1[c_grp].astype(str)

            fig1 = px.scatter_matrix(
                df1, ["dt", "tn", "IE"], color=c_grp,
                #  df1, ["ti", "dt", "tn", "IE"], color=c_grp,
                hover_name="PID",
                labels={
                    c_grp: "part enrgy<br>absorption order",
                    "dt": u"\u0394 t",
                    "ti": "t<sub>i</sub>",
                    "tn": "t<sub>n</sub>",
                    "IE": "IE<sub>max</sub>",
                },
                # px.colors.sequential.Jet_r  # ["red","goldenrod", "blue", "gray"],
                color_discrete_sequence=colors
            )
            # for ti, trace in enumerate(fig1.data):
            #     trace.legendrank = ti

            fig1.update_layout(
                width=400, height=720,
                # width=800, height=720,
                font_size=20,
                showlegend=False,
                # legend=dict(
                #     yanchor='top', y=0.99, xanchor='right', x=0.98
                # )
                #     margin=dict(t=30, r=0, l=20, b=10),
                #     scene_camera=dict(
                #         eye=dict(x=2, y=2, z=0.1)
                #         )
            )
            # fig1.update_layout({'yaxis1': {'range': [5, 20]}, 'yaxis2': {'range': [0, 58]}, 'yaxis3': {'range': [15, 70]}, 'yaxis4': {'range': [2, 38]}})

            fig1.update_layout({'yaxis1': {'range': [6, 58]}, 'yaxis2': {
                               'range': [15, 70]}, 'yaxis3': {'range': [6, 34]}})

            fig1.update_traces(
                marker_size=6,
                diagonal_visible=False,
                showupperhalf=False,
            )
            # name = 'eye (x:2, y:2, z:0.1)'
            # camera = dict(eye=dict(x=2, y=2, z=0.1))
            # fig1.update_layout(
            #     scene_camera=camera
            # )

            fig1.write_image(dst)
            print(dst)
            fig1.show()
            # break


def plot_doe_lc_pid():
    import seaborn as sns
    lcs = ['fp3']  # , 'fo5', 'fod_']
    rls = ['stv0_', 'stcr']
    # rls = ['stv03', '_m1_']
    sns.set(color_codes=False)
    c_grp = 'PID'
    # c_grp = 'c_grOrd'
    ords = 5
    ns = 100
    nPid = 5
    pids = ''

    for lc in lcs:
        for rl in rls:
            dst = '../publication/06_KG_energyAbsorption/images/plot/doe_pid_{0}_{1}.pdf'.format(
                lc, rl)
            sims = '.*{1}.*{0}.*'.format(lc, rl)
            oem = oems.oems('CEVT')
            df1 = oem.cypher().out_dataframe(
                ns=int(ns), nPID=int(nPid), nOrd=ords, regs=sims, regp=pids)

            df1 = df1.sort_values(by=['c_grOrd'])
            df1[c_grp] = df1[c_grp].astype(str)
            df1['IE'] = df1['IE'] / 1000  # kNmm

            fig1 = px.scatter_matrix(
                df1, ["ti", "dt", "tn", "IE"], color=c_grp,
                hover_name="sim",
                labels={
                    c_grp: "PID",
                    "dt": u"\u0394 t",
                    "ti": "t<sub>i</sub>",
                    "IE": "IE<sub>max</sub>",
                    "tn": "t<sub>n</sub>",
                },
            )

            fig1.update_layout(
                width=600, height=600,
                font_size=15,
                legend=dict(
                    yanchor='top', y=1, xanchor='right', x=0.99
                ),
            )
            fig1.update_traces(
                marker_size=4,
                showupperhalf=False,
            )

            fig1.show()
            fig1.write_image(dst)


def plot_doe_runs(runs='.*',
                  nPid=20
                  ):

    import seaborn as sns
    sns.set(color_codes=False)
    c_grp = 'sim_abb'
    ords = 20
    ns = 100
    pids = ''

    dst = '../publication/06_KG_energyAbsorption/images/plot/doe_pid_runs_{0}.svg'.format(
        runs.replace(',', '_'))
    oem = oems.oems('CEVT')

    df1 = oem.cypher().out_dataframe(
        ns=int(ns), nPID=int(nPid), nOrd=ords, regs=runs, regp=pids)

    # color order for HHL
    sims = runs.split(',')
    df1['order'] = pd.Categorical(df1.sim, sims, ordered=True)
    df1 = df1.sort_values(by=['order'])

    df1[c_grp] = df1[c_grp].astype(str)
    df1['IE'] = df1['IE'] / 1000  # kNm
    fig1 = px.scatter(
        df1, x="dt", y="IE", color=c_grp,
        hover_name="PID",
        labels={
            # c_grp: "PID",
            "dt": u"\u0394 t",
            "ti": "t<sub>i</sub>",
            "IE": "IE<sub>max</sub>",
            "tn": "t<sub>n</sub>",
        })

    fig1.update_traces(
        marker_size=8)
    fig1.update_layout(
        width=500, height=500,
        font_size=18,
        # yaxis_range=[2, 32],
        # xaxis_range=[2, 59]
    )

    fig1.write_image(dst)
    print(dst)
    # fig1.show()


def plot_doe_runs_HHLL(runs='.*',
                       nPid=20
                       ):

    import seaborn as sns
    sns.set(color_codes=False)
    c_grp = 'sim_abb'
    ords = 20
    ns = 100
    pids = ''

    dst = '../publication/06_KG_energyAbsorption/submition/HHLL_{0}_{1}_{2}.pdf'
    oem = oems.oems('CEVT')

    df1 = oem.cypher().out_dataframe(
        ns=int(ns), nPID=int(nPid), nOrd=ords, regs=runs, regp=pids)

    # ORDER OF HHL
    sims = runs.split(',')
    sims_abb = [s.split('_')[2] for s in sims]
    df1['order'] = pd.Categorical(df1.sim, sims, ordered=True)
    df1 = df1.sort_values(by=['order'])

    # ADD TAGS
    tags = [
        'H<sub>1</sub>: ',
        'H<sub>2</sub>: ',
        'L  : '
    ]
    for i, s in enumerate(sims_abb):
        df1.loc[df1.sim_abb == s, 'sim_abb'] = tags[i] + s

    # SET THE COLOR CODE
    cList = ['#636EFA', '#00CC96', '#EF553B']
    colors = {tags[i]+x: cList[i] for i, x in enumerate(sims_abb)}

    # PLOT
    df1[c_grp] = df1[c_grp].astype(str)
    df1['IE'] = df1['IE'] / 1000  # kNm
    fig1 = px.scatter(
        df1, x="dt", y="IE", color=c_grp,
        hover_name="PID",
        color_discrete_map=colors,
        labels={
            # c_grp: "PID",
            "dt": u"\u0394 t",
            "ti": "t<sub>i</sub>",
            "IE": "IE<sub>max</sub>",
            "tn": "t<sub>n</sub>",
            "sim_abb": 'Simulations',
        })

    fig1.update_traces(
        marker_size=10,
    )
    fig1.update_layout(
        width=400, height=500,
        font_size=20,
        # yaxis_range=[2, 22],
    )
    fig1.update_layout(legend=dict(
        yanchor="top",
        y=1,
        xanchor="right",
        x=1.1
    ))

    dst = dst.format(sims[0], sims_abb[1], sims_abb[2])
    # fig1.write_image(dst)
    fig1.show()
    # input(dst)


def plot_doe_lc_pid_1fig():
    import seaborn as sns
    lcs = ['fp3']  # , 'fo5', 'fod_']
    rls = ['stv0_', 'stv03']
    sns.set(color_codes=False)
    c_grp = 'PID'
    # c_grp = 'c_grOrd'
    ords = 8  # 5
    ns = 100
    nPid = 10  # 5
    # pids = [
    # ' 10020420, 10021520, 55131400, 55132410, 18620120, 18620080, 55021060, 18620110, 18620070',  # stv0
    # '18620090, 18620070, 10021870, 10021320, 55131440, 55131220, 55021060, 55021040',  # fp3 stv03
    # ]
    pids = [
        ' 10020420, 10021520, 18620120, 18620080, 55021060, 55021040, 55131400, 55132410, 55131390, 18620110, 18620070, 10021350, 10022010',  # stv0
        '10021870, 10021320, 18620090, 18620070, 55021060, 55021040,  55131440, 55131220, 55132390, 55132820, 55131010, 10021220, 10021830',  # fp3 stv03

    ]
    # pids = ['', '']

    for lc in lcs:
        for ri, rl in enumerate(rls):
            dst = '../publication/06_KG_energyAbsorption/images/plot/doe_pid_{0}_{1}_1fig.pdf'.format(
                lc, rl)
            dst = '../publication/06_KG_energyAbsorption/images/plot/doe_pid_{0}_{1}_dt_1fig_old.svg'.format(
                lc, rl)
            sims = '.*{1}.*{0}.*'.format(lc, rl)
            oem = oems.oems('CEVT')
            df1 = oem.cypher().out_dataframe(
                ns=int(ns), nPID=int(nPid), nOrd=ords, regs=sims, regp=pids[ri])

            df1 = df1.sort_values(by=[c_grp])
            df1[c_grp] = df1[c_grp].astype(str)
            df1['IE'] = df1['IE'] / 1000  # kNmm
            # for pk, pid in enumerate(df1['PID'].unique()):
            #     df1 = df1.replace(to_replace=pid.strip(), value=str(pk + 1))
            for i, ordi in enumerate(df1.c_grOrd):
                ordN = int(re.findall(r'\d+', ordi)[0])
                if ordN > 10:
                    df1.at[i, 'PID'] = ''

            fig1 = px.scatter(
                df1, x='dt', y="IE", color=c_grp,
                hover_name="PID",
                labels={
                    c_grp: "PID",
                    "tn": "t<sub>n</sub> [ms]",
                    "dt": u"\u0394 t [ms]",
                    "IE": "IE<sub>max</sub> [kNmm]",
                },
                color_discrete_sequence=['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
                                         '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']
                #  px.colors.cyclical.HSV  # ["red","goldenrod", "blue", "gray"],
            )

            fig1.update_layout(
                width=380, height=560,
                font_size=18,
                # legend=dict(
                #     yanchor='top', y =1, xanchor='right', x=0.99
                yaxis_range=[2, 32],
                xaxis_range=[2, 59]
                # ),
            )
            fig1.update_traces(
                marker_size=7,
            )

            # fig1.show()
            print(dst)
            fig1.write_image(dst)
            # break


def plot_doe_ord_1fig():
    import seaborn as sns
    lcs = ['fp3']  # , 'fo5', 'fod_']
    rls = ['stcr', 'stv0_', 'stv03']
    sns.set(color_codes=False)
    c_grp = 'c_grOrd'
    ords = 8  # 5
    ns = 100
    nPid = 8  # 5

    for lc in lcs:
        for ri, rl in enumerate(rls):
            sims = '.*{1}.*{0}.*'.format(lc, rl)
            oem = oems.oems('CEVT')
            df1 = oem.cypher().out_dataframe(
                ns=int(ns), nPID=int(nPid), nOrd=ords, regs=sims, regp='')

            df1 = df1.sort_values(by=[c_grp])
            df1['IE'] = df1['IE'] / 1000  # kNmm

            xy_range = [
                # [[6, 59], [6, 32]],
                [[7, 18], [6, 36]],
                # [[6, 59], [15, 70]],
                # [[15, 70], [6, 32]]
            ]
            pair_list = [
                # ['dt', 'IE'],
                ['ti', 'IE'],
                # ['dt', 'tn'],
                # ['tn', 'IE']
            ]
            leg = [False]
            # leg = [True, False, False, False]
            for pi, pairs in enumerate(pair_list):
                print(xy_range[pi][0])
                xp, yp = pairs
                dst = '../publication/06_KG_energyAbsorption/images/plot/doe_ord_{0}_{1}_{2}{3}_{4}_1fig.svg'.format(
                    lc, rl, xp, yp, leg[pi])
                fig1 = px.scatter(
                    df1, x=xp, y=yp, color=c_grp,
                    hover_name="PID",
                    labels={
                        c_grp: "part enrgy<br>absorption order",
                        "tn": "t<sub>n</sub> [ms]",
                        "ti": "t<sub>i</sub> [ms]",
                        "dt": u"\u0394 t [ms]",
                        "IE": "IE<sub>max</sub> [kNmm]",
                    },
                    # sequential.Blackbody  # ["red",  "goldenrod", "blue", "gray"],
                    color_discrete_sequence=px.colors.cyclical.IceFire_r[0::2]
                )

                fig1.update_layout(
                    # width=380, height=560,
                    width=380, height=380,
                    font_size=18,
                    # legend=dict(
                    #     yanchor='top', y =1, xanchor='right', x=0.99
                    yaxis_range=xy_range[pi][1],
                    xaxis_range=xy_range[pi][0],
                    showlegend=leg[pi]
                    # ),
                )
                fig1.update_traces(
                    marker_size=7,
                )
                # fig1.show()
                print(dst)
                fig1.write_image(dst)
                # break
            # break
                # break


def plot_doe_rls():
    import seaborn as sns
    lc = 'fp3'
    rlsComb = [
        ['stcr', 'stv0_'],
        ['stv0_', 'stv03'],
        ['stv03', '_m1_'],
        # ['stcr', 'stv0_', 'stv03', '_m1_']
    ]
    color = ["red", "blue", "goldenrod", "green"]
    sns.set(color_codes=False)
    c_grp = 'c_rls'
    ords = 8
    ns = 10000
    nPid = 8
    pids = ''

    for ri, rls in enumerate(rlsComb):
        sims_txt = ','.join(['.*' + x + '.*{0}.*' for x in rls])
        fName = '_'.join(rls)

        dst = '../publication/06_KG_energyAbsorption/images/plot/doe_fp3_{0}_{1}.svg'.format(
            lc, fName)
        sims = sims_txt.format(lc)
        oem = oems.oems('CEVT')
        df1 = oem.cypher().out_dataframe(
            ns=int(ns), nPID=int(nPid), nOrd=ords, regs=sims, regp=pids)

        # df1 = df1.replace('m1', 'zm1')
        # df1 = df1.replace('zm1', 'm1')

        df1['rlsNo'] = df1['c_rls'].map(
            {'stcr': 10, 'stv0': 20, 'stv03': 30, 'm1': 40})
        df1 = df1.sort_values(by=['rlsNo'])
        df1['IE'] = df1['IE'] / 1000  # kNmm
        df1 = df1.replace(to_replace='stcr', value='Primary')
        df1 = df1.replace(to_replace='stv0', value='Early')
        df1 = df1.replace(to_replace='stv03', value='Middle')
        df1 = df1.replace(to_replace='m1', value='Late')

        fig1 = px.scatter_matrix(
            df1, ["ti", "tn", "IE"], color=c_grp, size='rlsNo',
            # df1, ["ti", "dt", "tn", "IE"], color=c_grp, size='rlsNo',
            hover_name="sim",
            # hover_name="PID",
            labels={
                c_grp: "Development Phase",
                "dt": u"\u0394 t",
                "ti": "t<sub>i</sub>",
                "IE": "IE<sub>max</sub>",
                "tn": "t<sub>n</sub>",
            },
            color_discrete_sequence=color[ri:ri + 2],
            # color_discrete_sequence=color,
        )

        fig1.update_layout(
            # width=600, height=600,
            width=200, height=300,
            font=dict(size=10),
            margin=dict(l=0, r=0, b=0, t=0),
            showlegend=False
            # legend=dict(
            #     yanchor='top', y=0.99, xanchor='right', x=0.99
            # )
        )
        # fig1.update_layout({'yaxis1': {'range': [5, 20]}, 'yaxis2': {'range': [0, 58]}, 'yaxis3': {'range': [15, 70]}, 'yaxis4': {'range': [2, 38]}})

        fig1.update_layout({'yaxis1': {'range': [7, 20]}, 'yaxis2': {
                           'range': [15, 70]}, 'yaxis3': {'range': [2, 38]}})

        fig1.update_traces(
            marker_size=3,
            diagonal_visible=False,
            showupperhalf=False,
        )
        print(dst)
        fig1.write_image(dst)
        fig1.show()


def plot_doe_rls_1fig():
    import seaborn as sns
    lc = 'fp3'
    rlsComb = [
        ['stcr', 'stv0_'],
        # ['stv0_', 'stv03'],
        # ['stv03', '_m1_']
    ]
    color = ["red", "blue", "goldenrod", "green"]
    sns.set(color_codes=False)
    c_grp = 'c_rls'
    ords = 5
    ns = 10000
    nPid = 5
    pids = ''

    for ri, rls in enumerate(rlsComb):
        sims_txt = ','.join(['.*' + x + '.*{0}.*' for x in rls])
        fName = '_'.join(rls)

        dst = '../publication/06_KG_energyAbsorption/images/plot/doe_rls_{0}_{1}_1fig.pdf'.format(
            lc, fName)
        sims = sims_txt.format(lc)
        oem = oems.oems('CEVT')
        df1 = oem.cypher().out_dataframe(
            ns=int(ns), nPID=int(nPid), nOrd=ords, regs=sims, regp=pids)

        # df1 = df1.replace('m1', 'zm1')
        # df1 = df1.replace('zm1', 'm1')

        df1['rlsNo'] = df1['c_rls'].map(
            {'stcr': 10, 'stv0': 20, 'stv03': 30, 'm1': 40})
        df1 = df1.sort_values(by=['rlsNo'])
        df1['IE'] = df1['IE'] / 1000  # kNmm
        df1 = df1.replace(to_replace='stcr', value='Primary')
        df1 = df1.replace(to_replace='stv0', value='Early')
        df1 = df1.replace(to_replace='stv03', value='Middle')
        df1 = df1.replace(to_replace='m1', value='Late')

        fig1 = px.scatter(
            df1, x='tn', y='IE', color=c_grp,
            hover_name="sim",
            labels={
                c_grp: "Development Phase",
                "dt": u"\u0394 t",
                "ti": "t<sub>i</sub>",
                "IE": "IE<sub>max</sub>",
                "tn": "t<sub>n</sub>",
            },
            color_discrete_sequence=color[ri:ri + 2]
        )

        fig1.update_layout(
            width=300, height=400,
            font=dict(size=20),
            showlegend=False
            # legend=dict(
            #     yanchor='top', y=0.99, xanchor='right', x=0.95
            # )
        )
        # fig1.update_layout({'yaxis1': {'range': [5, 20]}, 'yaxis2': {'range': [0, 58]}, 'yaxis3': {'range': [15, 70]}, 'yaxis4': {'range': [2, 38]}})

        # fig1.update_layout({ 'yaxis1': {'range': [0, 58]}, 'yaxis2': {'range': [2, 38]}})

        fig1.update_traces(
            marker_size=6,
        )
        print(dst)
        fig1.write_image(dst)
        fig1.show()


def plot_doe_lc_all():
    import seaborn as sns
    lcs = ['fp3', 'fo5', 'fod_']
    sns.set(color_codes=False)
    c_grp = 'c_lc'
    ords = 5
    ns = 1000
    nPid = 5
    pids = ''

    dst = '../publication/06_KG_energyAbsorption/images/plot/doe_all.pdf'  # with tn
    dst = '../publication/06_KG_energyAbsorption/images/plot/doe_all_dt.pdf'
    sims = '.*fp3.*,.*fod_.*,.*fo5.*'
    oem = oems.oems('CEVT')
    df1 = oem.cypher().out_dataframe(
        ns=int(ns), nPID=int(nPid), nOrd=ords, regs=sims, regp=pids)

    df1 = df1.sort_values(by=[c_grp])
    df1['IE'] = df1['IE'] / 1000  # kNmm

    df1 = df1.replace('fo5', 'foI')
    df1 = df1.replace('fod', 'foU')
    df1 = df1.replace('fp3', 'ffo')

    fig1 = px.scatter_matrix(
        df1, ["ti", "dt", "tn", "IE"], color=c_grp,
        hover_name="PID",
        labels={
            c_grp: "Load Cases",
            "dt": u"\u0394 t",
            "ti": "t<sub>i</sub>",
            "tn": "t<sub>n</sub>",
            "IE": "IE<sub>max</sub>",
        },
    )
    fig1.update_layout(
        width=600, height=600,
        font=dict(size=20),
        legend=dict(
            yanchor='top', y=0.99, xanchor='right', x=0.95
        )
    )
    fig1.update_traces(
        marker_size=4,
        diagonal_visible=False,
        showupperhalf=False,
    )
    fig1.show()
    fig1.write_image(dst)


def plot_doe_lc_all_dt_tn(rls):
    import seaborn as sns
    lcs = ['fp3', 'fo5', 'fod_']
    sns.set(color_codes=False)
    c_grp = 'c_lc'
    ords = 5
    ns = 1000
    nPid = 5
    pids = ''

    dst = '../publication/06_KG_energyAbsorption/images/plot/doe_{}_dt_tn.pdf'.format(
        rls.replace('.*', ''))  # with tn
    sims = '{0}.*fp3.*,{0}.*fod_.*,{0}.*fo5.*'.format(rls)
    oem = oems.oems('CEVT')
    df1 = oem.cypher().out_dataframe(
        ns=int(ns), nPID=int(nPid), nOrd=ords, regs=sims, regp=pids)

    df1 = df1.sort_values(by=[c_grp])
    df1['IE'] = df1['IE'] / 1000  # kNmm

    fig1 = px.scatter_matrix(
        df1, ["ti", "tn", "dt"], color=c_grp,
        hover_name="PID",
        labels={
            c_grp: "Load Cases",
            "tn": "t<sub>n</sub>",
            "dt": u"\u0394 t",
            "ti": "t<sub>i</sub>",
        },
    )
    fig1.update_layout(
        width=600, height=600,
        font=dict(size=14),
        legend=dict(
            yanchor='top', y=0.99, xanchor='right', x=0.95
        )
    )
    fig1.update_traces(
        marker_size=4,
        showupperhalf=False,
        diagonal_visible=False
    )
    fig1.show()
    # fig1.write_image(dst)


def plot_power_law():
    oem = oems.oems('CEVT')
    c_grp = 'c_grPID'
    c_grp = 'c_grOrd'
    ords = 20
    ns = 1000
    nPid = 20
    sims = ['.*_stv0_.*', '.*_stv0_.*fp3.*',
            '.*_stv0_.*fo5.*', '.*_stv0_.*fod_.*']
    sims = ['.*_stv0_.*']
    pids = ''

    for s in sims:
        dst = '../publication/06_KG_energyAbsorption/images/plot/powerlaw{0}.pdf'.format(
            s.replace('.*', ''))
        df1 = oem.cypher().out_dataframe(
            ns=int(ns), nPID=int(nPid), nOrd=ords, regs=s, regp=pids)
        df1 = df1.drop_duplicates('PID', keep='first')
        df1 = df1.sort_values(by=['count'], ascending=False)

        df_top = df1[df1['count'] > 1]
        df_top['PID'] = df_top['PID'].astype(str)

        h_data = {
            "sim": True,
            "c_grPID": False,
            "IE": ':.2f',
            "ti": ':.3f',
            "dt": ':.2f',
            "tn": ':.2f'}
        fig1 = px.bar(df_top, x='PID', y='count',
                      # color=c_grp,
                      custom_data=["pic"],
                      hover_name="PID",
                      hover_data=h_data,
                      labels={
                          'count': 'DES Degree'
                      },)
        fig1.update_layout(
            # width=300, height=200,
            width=700, height=200,
            font_size=18,
            legend=dict(
                yanchor='top', y=0.8, xanchor='right', x=0.92),
            margin=dict(t=0, r=0, l=0, b=0, pad=5),
            showlegend=False,
            xaxis_showticklabels=False
        )

        fig1.write_image(dst)
        # fig1.show()
        # break


def plot_sym_part():
    import seaborn as sns
    lcs = ['fp3']
    sns.set(color_codes=False)
    c_grp = 'PID'
    ords = 3
    ns = 100
    nPid = 5
    pids = '10021520, 10020420, 18620120, 18620080'

    dst = '../publication/06_KG_energyAbsorption/images/plot/fp3_sym_part_3d.svg'
    sims = '.*stv0_.*fp3.*'
    oem = oems.oems('CEVT')
    df1 = oem.cypher().out_dataframe(
        ns=int(ns), nPID=int(nPid), nOrd=ords, regs=sims, regp=pids)

    df1['PID'] = df1['PID'].astype(str)
    df1['IE'] = df1['IE'] / 1000  # kNmm

    df1 = df1.sort_values(by=[c_grp])
    fig1 = px.scatter_3d(
        df1, x="ti", y="dt", z="IE", color=c_grp, custom_data=["pic"],
        hover_name="PID")  # , template="plotly_white")

    fig1.update_layout(
        width=600, height=600,
        scene_camera=dict(
            eye=dict(x=1.8, y=1.8, z=0.3)),
        scene=dict(
            xaxis_title="t<sub>i</sub> [ms]",
            yaxis_title=u'\u0394 t [ms]',
            zaxis_title='IE<sub>max</sub> [kNmm]',
            xaxis_nticks=3,
            yaxis_nticks=3,
            zaxis_nticks=3,
            # xaxis_linecolor = 'black',
            # yaxis_linecolor = 'black',
            # zaxis_linecolor = 'black',
            # xaxis_gridcolor = 'lightgray',
            # yaxis_gridcolor = 'lightgray',
            # zaxis_gridcolor = 'lightgray',
        ),
        font_size=16,
        legend=dict(
            yanchor='top', y=0.8, xanchor='right', x=0.92),
        margin=dict(t=0, r=0, l=0, b=0, pad=5),
    )
    fig1.update_traces(
        marker_size=6,
    )

    fig1.write_image(dst)
    fig1.show()


def plot_3d_fp3_stv0():
    import seaborn as sns
    sns.set(color_codes=False)
    c_grp = 'PID'
    ords = 1000
    ns = 1000
    nPid = 1000
    pids = '10021520, 10020420, 18620080, 18620120, 55131400, 55132410, 55132590,55021040, 55021060, 18620070, 18620110'
    sims = '.*stv0_.*fp3.*'
    oem = oems.oems('CEVT')

    dst = '../publication/06_KG_energyAbsorption/images/plot/fp3_stv0_3d.svg'
    df1 = oem.cypher().out_dataframe(
        ns=int(ns), nPID=int(nPid), nOrd=ords, regs=sims, regp=pids)

    df1['PID'] = df1['PID'].astype(str)
    df1['IE'] = df1['IE'] / 1000  # kNmm

    df1 = df1.sort_values(by=[c_grp])
    fig1 = px.scatter_3d(
        df1, x="ti", y="tn", z="IE", color=c_grp, custom_data=["pic"],
        hover_name="PID")  # , template="plotly_white")

    fig1.update_layout(
        width=600, height=600,
        scene_camera=dict(
            # eye=dict(x=-1.4, y=-1.8, z=1.8)),
            eye=dict(x=-1.4, y=1.8, z=0.8)),
        scene=dict(
            xaxis_title='t<sub>i</sub> [ms]',
            yaxis_title='t<sub>n</sub> [ms]',
            zaxis_title='IE<sub>max</sub> [kNmm]',
            xaxis_nticks=3,
            yaxis_nticks=3,
            zaxis_nticks=3,
            # xaxis_linecolor = 'black',
            # yaxis_linecolor = 'black',
            # zaxis_linecolor = 'black',
            # xaxis_gridcolor = 'lightgray',
            # yaxis_gridcolor = 'lightgray',
            # zaxis_gridcolor = 'lightgray',

        ),
        font_size=16,
        legend=dict(
            yanchor='top', y=0.95, xanchor='right', x=0.97),
        margin=dict(t=0, r=0, l=0, b=0, pad=5),
    )
    fig1.update_traces(
        marker_size=6,
    )

    fig1.write_image(dst)
    fig1.show()


def plot_3d_fp3_stv03():
    import seaborn as sns
    lcs = ['fp3']
    sns.set(color_codes=False)
    c_grp = 'PID'
    ords = 10
    ns = 100
    nPid = 10
    pids = '10021870, 10021320, 18620090, 18620070, 55131440, 55131150,55131220, 55131430, 55132390'
    pids = '18620090, 18620070, 10021870, 10021320, 55131440, 55131220, 55021060, 55021040'
    sims = '.*stv03_.*fp3.*'
    oem = oems.oems('CEVT')

    dst = '../publication/06_KG_energyAbsorption/images/plot/fp3_stv03_3d.svg'
    df1 = oem.cypher().out_dataframe(
        ns=int(ns), nPID=int(nPid), nOrd=ords, regs=sims, regp=pids)

    df1['PID'] = df1['PID'].astype(str)
    df1['IE'] = df1['IE'] / 1000  # kNmm

    df1 = df1.sort_values(by=[c_grp])
    fig1 = px.scatter_3d(
        df1, x="ti", y="tn", z="IE", color=c_grp, custom_data=["pic"],
        hover_name="PID")  # , template="plotly_white")

    fig1.update_layout(
        width=600, height=600,
        scene_camera=dict(
            eye=dict(x=-2.25, y=0.63, z=0.72)),
        scene=dict(
            xaxis_title='t<sub>i</sub> [ms]',
            yaxis_title='t<sub>n</sub> [ms]',
            zaxis_title='IE<sub>max</sub> [kNmm]',
            xaxis_nticks=3,
            yaxis_nticks=3,
            zaxis_nticks=3,
            # xaxis_linecolor = 'black',
            # yaxis_linecolor = 'black',
            # zaxis_linecolor = 'black',
            # xaxis_gridcolor = 'lightgray',
            # yaxis_gridcolor = 'lightgray',
            # zaxis_gridcolor = 'lightgray',

        ),
        font_size=16,
        legend=dict(
            yanchor='top', y=0.92, xanchor='right', x=0.99),
        margin=dict(t=0, r=0, l=0, b=0, pad=5),
    )
    fig1.update_traces(
        marker_size=6,
    )

    fig1.write_image(dst)
    fig1.show()


if __name__ == '__main__':

    global OEM
# PAG
    OEM = 'PAG'
 # mining
    # plot_ti_selection()
    # plot_ti_selection_sum()
    # plot_range_nrg_embd('../src/pic/range_nrg_embd.png')
    # plot_range_nrg_embd_2d_PAG('../src/pic/range_nrg_embd_2d_pag.png')
    # plot_range_nrg_embd_zoom('../src/pic/range_nrg_embd_zoom.png')
 # Manuscript
    # plot_ti_selection_sum()
    # plot_nrg_embd_similarity_PAG1('../publication/06_KG_energyAbsorption/images/plot/curve_sim_pag_{}.pdf')
    # plot_nrg_embd_similarity_PAG2('../publication/06_KG_energyAbsorption/images/plot/curve_sim_pag_{}.pdf')  # plot_ti_selection()

# CEVT
    OEM = 'CEVT'

 # mining
    # plot_CEVT().dna()#
    # plot_CEVT().sym('stv0', 'fp3')
    # plot_CEVT().out_dataframe('stv0', 'fp3')
    # plot_embd_binout_cevt()
    # plot_doe_lc_all_dt_tn('.*stv0_')
    # plot_doe_lc_all_dt_tn('.*stcr')
    # plot_doe_lc_all_dt_tn('.*')

    # plot_doe_runs('cm1e_m1_005_fp3__002,cm1e_m1_046_fp3__001,cm1e_m1_021_fp3__001')
    # plot_doe_runs('cm1e_m1_005_fp3__002,cm1e_m1_046_fp3__001,cm1e_m1_006_fp3__001')
    # plot_doe_runs('cm1e_m1_002_fod__001, cm1e_m1_003_fod__001,cm1e_m1_005_fod__002')
    # plot_doe_runs('cm1e_stcr_251_fp3__001,cm1e_stcr_308_fp3__001,cm1e_stcr_293_fp3__001')
    # plot_doe_runs('cm1e_stv03_226_fod__001,cm1e_stv03_330_fod__001, cm1e_stv03_004_fod__001')  # hard to judge, IE compensade with dt with npid=3
    # plot_doe_runs('cm1e_stv03_226_fod__001,cm1e_stv03_330_fod__001, cm1e_stv03_227_fod__001')  # hard to judge, IE compensade with dt with np=20
    # plot_doe_runs('cm1e_stv03_005_fod__001,cm1e_stv03_009_fod__001, cm1e_stv03_227_fod__001')  # hard to judge, IE compensade with dt with np=20
    # plot_nrg_embd_similarity_cevt2('../publication/06_KG_energyAbsorption/images/plot/curve_sim_cevt_{}.pdf')
 # Manuscript
    # plot_doe_lc_ord()
    # plot_doe_lc_pid()
    # plot_doe_lc_pid_1fig()
    # plot_doe_ord_1fig()
    # plot_doe_rls_1fig()
    # plot_doe_rls()
    # plot_doe_lc_all()
    # plot_power_law()
    # plot_sym_part()
    # plot_3d_fp3_stv0()
    # plot_3d_fp3_stv03()
    # plot_nrg_embd_similarity_cevt1('../publication/06_KG_energyAbsorption/images/plot/curve_sim_cevt_{}.pdf')
    # hard to judge, IE compensade with dt with np=20
    # plot_doe_runs(
    #     'cm1e_stcr_354_fo5__001,cm1e_stcr_387_fo5__001,cm1e_stcr_090_fo5__001', nPid=15)

    # on with the yaxis_range
    plot_doe_runs_HHLL(
        'cm1e_stcr_354_fo5__001,cm1e_stcr_387_fo5__001,cm1e_stcr_017_fo5__001', nPid=20)
    plot_doe_runs_HHLL(
        'cm1e_stcr_004_fo5__001,cm1e_stcr_007_fo5__001,cm1e_stcr_287_fo5__001', nPid=20)
    plot_doe_runs_HHLL(
        'cm1e_stcr_004_fo5__001,cm1e_stcr_007_fo5__001,cm1e_stcr_354_fo5__001', nPid=20)

    # off with the yaxis_range
    # plot_doe_runs_HHLL(
    # 'cm1e_stcr_354_fo5__001,cm1e_stcr_387_fo5__001,cm1e_stcr_237_fo5__001', nPid = 20)


# # YARIS
    OEM = 'YARIS'
 # mining
    # plot_nrg_embd_2d_yaris_sub_debug_tn()
    # plot_nrg_embd_2d_yaris_sub_9part()

 # Manuscript
    # plot_nrg_embd_2d_yaris_sub()
    # plot_nrg_embd_2d_yaris_sub_rev()

# plt.show()
