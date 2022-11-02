from neo4j import GraphDatabase
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import annotate_hoover_scatter as sc_h
from random import randint

sys.path.append('..')
try:
    import KG_feed_OEM_data as KG
    import oems
except NameError:
    raise

# # oem = 'Porsche'
# oem = 'CEVT'
# oem = 'YARIS'
# print(oem)
# if oem == 'CEVT':
#     # KG.neo4j_bolt('7687', 'ivory')
#     # uri = "neo4j://ivory:7687"
#     uri = "bolt://ivory:7687"
#     driver = GraphDatabase.driver(uri, auth=("neo4j", "ivory123"))
#     KG.neo4j_bolt('7687', 'localhost')

# elif oem == 'Porsche':
#     uri = "neo4j://localhost:3687"
#     driver = GraphDatabase.driver(uri, auth=("neo4j", "ivory123"))
#     KG.neo4j_bolt('3687', 'ivory')
# if oem == 'YARIS':
#     #     uri = "neo4j://localhost:3687"
#     #     driver = GraphDatabase.driver(uri, auth=("neo4j", "ivory123"))
#     KG.neo4j_bolt('3687', 'localhost')


def nrg_cypher(list, func, opr='=~'):

    txt = "MATCH (n:Sim) WHERE n.sim_name {0} $name RETURN ".format(opr)
    for f in func:
        for item in list:
            txt += ("{0}(n.sim_{1}),".format(f, item))
    return(txt[:-1])


def nrg_overview(name, opt, name_list):
    with driver.session() as session:

        func = ["max", "min", "stDevP", "avg"]
        result = session.run(nrg_cypher(name_list, func), name=name)
        # return
        for record in result:
            nrg_scales = record.values()
            ss = len(name_list)
            stDev = nrg_scales[2 * ss:3 * ss]
            avg = nrg_scales[3 * ss:]

            return(
                np.asarray([avg[o] for o in opt]),
                np.asarray([stDev[o] for o in opt]), name_list)


def plt_nrg(prop):
    with driver.session() as session:
        result = session.run(nrg_cypher(), name='ROB_VOWA.*')

        for record in result:
            (record.values())


def PAG_overview(opt, title, names):
    '''
    plot the overview of energy features for feed_normalization
    opt     list of numbers which refer to feature extraction dictionary
    title   plot header
    names   list, includes features names (features dictionary header)
    '''

    sim_batches = [
        '505', '721', '506', '722',
        '507', '726', '508', '735',
        '715', '736', '716', '737',
        '717', '739', '718', '740',
        '719', '741', '720', '742']
    # sim_batches = ['739', '742', '716', '740', '715', '508']
    # sim_batches = ['739', '742', '716']

    avg = [[] for o in opt]
    std = [[] for o in opt]
    for b in sim_batches:
        # print('-------------------------------')
        # print('   {}   '.format(b))
        # print('-------------------------------')
        [avgi, stdi, name] = nrg_overview('.*' + b + '.*', opt, names)
        for o, i in enumerate(opt):
            avg[o].append(avgi[o])
            # input("Press Enter to continue...")
            std[o].append(stdi[o])

    for o, i in enumerate(opt):
        plt.errorbar(sim_batches, avg[o], std[o], label=name[i])
    plt.legend(fontsize=20)
    plt.title(title, fontsize=20)
    plt.show()


def nrg_normalized(nameReg, fts, norm_opt):
    with driver.session() as session:
        func = ["max", "min"]
        result_norm = session.run(nrg_cypher(norm_opt, func), name=nameReg)
        for r in result_norm:
            ss = len(norm_opt)
            MAX = np.asarray(r.values()[0:ss])
            # MIN = np.asarray(r.values()[ss:])
        MAX[2] = MAX[2] - MAX[1]
        fts[:, 2] = (fts[:, 2] - fts[:, 1])  # dt
        fts_nrm = (fts) / (MAX)
        return(fts_nrm)


def feed_normalization(nrmList, simList, norm_opt, ft_opt, pids_sel=None):
    sims_nrg = []
    sims_pid = []
    sims_pid0 = []
    for s in simList:
        sim = KG.Sim.nodes.get_or_none(sim_name=s)
        if pids_sel:
            fts = sim.embed_nrg_sel(ft_opt, pids_sel)
            if not len(fts) == 0:
                pids = fts[:, -1]
                fts = fts[:, :-1]
                diff_len = len(pids_sel) - len(pids)
            else:
                continue
            if diff_len > 0:
                for di in range(0, diff_len):
                    pids = np.append(pids, 0)
                    fts = np.append(fts, [[0, 0, 0]], axis=0)
        else:
            fts = sim.embed_nrg(ft_opt)
            if not len(fts) == 0:
                pids = fts[:, -1]
                fts = fts[:, :-1]
            else:
                continue
        # print(s)
        # print(fts.shape)
        if not norm_opt == []:
            fts_nrm = nrg_normalized(nrmList, fts, norm_opt)
        else:
            fts_nrm = fts

        sims_nrg.append(fts_nrm)
        sims_pid.append(np.array(pids))

    sims_nrg = np.array(sims_nrg)
    sims_pid = np.array(sims_pid)

    return(sims_nrg, sims_pid)


def plt_nrg_embed_old(sims_fts, pids, id=None, plt3=None, m=10):

    if plt3:
        fig, (ax1, ax3, ax2) = plt.subplots(1, 3)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2)

    for i, s in enumerate(sims_fts):
        x = s[0:m, 0]  # IE
        y = s[0:m, 1]  # ti
        z = s[0:m, 2]  # dt

        ax1.plot(y, x, 'o')
        ax1.set(xlabel='$t_0$', ylabel='$IE_{max}$')
        ax2.plot(y, z, 'o')
        ax2.set(xlabel='$t_0$', ylabel='$\\Delta t$')
        if plt3:
            ax3.plot(z, x, 'o')
            ax3.set(xlabel='$\\Delta t$', ylabel='$IE_{max}$')

        if id:
            for j, txt in enumerate(pids[i]):
                TXT = str(int(txt)).split('20000')[-1]  # + '_' + str(i)
                ax1.annotate(TXT, (y[j], x[j]))
                ax2.annotate(TXT, (y[j], z[j]))
                if not z[j] == 0:
                    ax3.annotate(TXT, (z[j], x[j]))
                if j >= m - 1:
                    break
    plt.show()


def get_color(n):
    colors = []
    for i in range(n):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    return(colors)


def plt_nrg_embed(sims_fts, pids, names, id=None, plt3=None,
                  m=10, grp='Yes', leg=None, grpPid=None, grpOrd=None,
                  fig=None, axs=None, cN=None):

    if sum(plt3) == 3:
        fig, axs = plt.subplots(1, 3)
    elif sum(plt3) == 2:
        fig, axs = plt.subplots(1, 2)
    else:
        if not axs:
            fig, axs = plt.subplots()
            axs = [axs]

    pidU = np.unique(pids[:, :m])
    c = ['b', 'r', 'g', 'c', 'm', 'y', 'k', '#4AFFB5']
    # c = ['#5A5E94', '#47736A', '#C394E6', '#C96833', '#E73D50', '#263BE8', '#66139F', '#C0E594', '#F9C2C8', '#A3DECE', '#2063FE', '#6F65A5', '#0425E5', '#0A5F1F', '#743A0B', '#4BB0E1', '#D2A23C', '#297B72', '#616D4E', '#744593', '#89E58B', '#51B7D2', '#7FCD26', '#35CD1C', '#445C37', '#C06CA7', '#07A530', '#7A174D', '#5A8EEF', '#4F3698']

    if len(pidU) > len(c):
        if grpOrd:
            if m > len(c):
                c = get_color(len(pidU))
        else:
            c = get_color(len(pidU))
    if grp:
        if grpPid:
            axs, fig = plt_pid_group(sims_fts, pids, fig, axs, m, c, plt3)
        # else: # grpsim
        #     axs, fig =

    if cN:
        c = cN
    for i, s in enumerate(sims_fts):
        x = s[0:m, 0]  # IE
        y = s[0:m, 1]  # ti
        z = s[0:m, 2]  # dt

        pid_names = [str(int(x)) for x in pids[i][0:m]]
        # plt.show()
        ci = 0
        ii = i % len(c)

        if grpPid:
            cii = []
            for p in pids[i, :m]:
                for ui, pu in enumerate(pidU):
                    if p == pu:
                        cii.append(c[ui])
        elif grpOrd:
            cii = c[:len(pid_names)]
        else:  # groupSim:
            cii = c[ii]

        if plt3[0] == 1:
            # axs[ci].plot(y, x)
            fig, axs[ci] = sc_h.scatter_hoover(y, x, pid_names, cii, fig, axs[ci],
                                               names[i], leg=leg)
            axs[ci].set(xlabel='$t_i$', ylabel='$IE_{max}$')
            axs[ci].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ci += 1
        if plt3[1] == 1:
            # axs[ci].plot(z, x)
            fig, axs[ci] = sc_h.scatter_hoover(z, x, pid_names, cii, fig, axs[ci],
                                               names[i], leg=leg)
            axs[ci].set(xlabel='$t_n$', ylabel='$IE_{max}$')
            axs[ci].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ci += 1
        if plt3[2] == 1:
            # axs[ci].plot(y, z)
            fig, axs[ci] = sc_h.scatter_hoover(y, z, pid_names, cii, fig, axs[ci],
                                               names[i], leg=leg)
            axs[ci].set(xlabel='$t_i$', ylabel='$t_n$')
            ci += 1

        fig.set_size_inches(5, 5)
    return(fig, axs)


def plt_nrg_embed_simple(sims_fts, pids, names, id=None, plt3=None,
                         m=10, grp='Yes', leg=None, grpPid=None, grpOrd=None,
                         fig=None, axs=None, cN=None, marker=None):

    fig, axs = plt.subplots()
    axs = [axs]

    pidU = np.unique(pids[:, :m])

    c = cN
    sN = len(sims_fts) + 1
    for i, s in enumerate(sims_fts):
        x = s[0:m, 0]  # IE kN-mm
        y = s[0:m, 1]  # ti ms
        z = s[0:m, 2]  # dt ms

        pid_names = [str(int(x)) for x in pids[i][0:m]]
        ci = 0
        ii = i % len(c)

        for pi, p in enumerate(pids[i]):
            if marker:
                mp = marker[str(int(p))]
            else:
                mp = 'o'
            axs[ci].scatter(z[pi], x[pi], c=c[ii], s=(
                sN - 1 - i) * 100, marker=mp)
            axs[ci].set(xlabel='$t_n\ [ms]$', ylabel='$IE_{max}\\ [kNmm]$')
            axs[ci].ticklabel_format(
                axis='y', style='sci', scilimits=(0, 0), useLocale=':f')
        ci += 1

        fig.set_size_inches(5, 5)
    return(fig, axs)


def plt_nrg_embed_3d(sims_fts, pids, names, id=None, m=10, grp='Yes', cN=None,
                     grpSim=None, leg=None, grpPid=None):

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    pidU = np.unique(pids[:, :m])
    c = ['b', 'r', 'g', 'c', 'm', 'y', 'k', '#4AFFB5']
    if len(pidU) > len(c):
        c = get_color(len(pidU))
    if cN:
        c = cN

    if grp:
        axs, fig = plt_pid_group(sims_fts, pids, fig, ax, m, c)

    K = 0
    for i, s in enumerate(sims_fts):
        x = s[0:m, 0]  # IE
        y = s[0:m, 1]  # ti
        z = s[0:m, 2]  # dt/tn

        # if grp:
        #     ax.plot(y, x, z)
        pid_names = [str(int(x)) for x in pids[i][0:m]]
        if grpSim:
            ii = i % len(c)
            fig, ax = sc_h.scatter_hoover(
                z, y, pid_names, c[ii], fig, ax, names[i], z=x, leg=leg)
        elif grpPid:
            cii = []
            for p in pids[i, :m]:
                for ui, pu in enumerate(pidU):
                    if p == pu:
                        cii.append(c[ui])

            fig, ax = sc_h.scatter_hoover(
                z, y, pid_names, cii, fig, ax, names[i], z=x, leg=leg)
        else:
            for k, item in enumerate(x):
                fig, ax = sc_h.scatter_hoover(
                    z[k], y[k], [pid_names[k]], cN[K], fig, ax, names[i], z=x[k])
                K += 1

        ax.set(ylabel='$t_i$', zlabel='$IE_{max}$', xlabel='$t_n$/$\Delta t$')
        ax.ticklabel_format(axis='z', style='sci', scilimits=(0, 0))

    fig.set_size_inches(18.5, 10.5)
    # plt.show()
    return(fig, ax)


def fltr_nrg(limits, sims, pids):
    sims_fltr, pids_fltr = [], []

    for ii, sd in enumerate(sims):
        id_fltr = (
            np.where(
                (sd[:, 0] < limits[0])  # IE
                & (sd[:, 1] > limits[1])  # ti
                & (sd[:, 2] > limits[2])))  # dt
        sims_fltr.append(sd[id_fltr])
        # not pids are not sorted based on max, it is ok for plotting, but m
        # highst energy doesn't work, you sould use filter with where
        pids_fltr.append(pids[ii][id_fltr])

    pids_fltr = np.array(pids_fltr)
    sims_fltr = np.array(sims_fltr)
    return(sims_fltr, pids_fltr)


def nrg_diff(sims_nrg, sims_pid, fltr=None):

    sims_pid_ord = np.zeros(sims_pid.shape)
    sims_nrg_ord = np.zeros(sims_nrg.shape)
    for i, s in enumerate(sims_nrg):
        a, b = zip(*sorted(zip(sims_pid[i], s)))
        sims_pid_ord[i], sims_nrg_ord[i] = a, b

    if (sims_pid_ord[0][:] == sims_pid_ord[1:][:]).all():
        sims_nrg_diff = abs(sims_nrg_ord[1:] - sims_nrg_ord[0])
        sims_nrg_diff_srt, sims_pid_srt = sort_nrg(
            sims_nrg_diff[:], sims_pid_ord[1:])

        if fltr:
            limits_fltr = [3e6, 0, 0]
            sims_nrg_diff_fltr, sims_pid_fltr = fltr_nrg(
                limits_fltr,
                sims_nrg_diff_srt, sims_pid_srt)
            return(sims_nrg_diff_fltr, sims_pid_fltr)

        return(sims_nrg_diff_srt, sims_pid_srt)


def sort_nrg(sims_nrg, sims_pid, comp=0):
    sims_nrg_srt = sims_nrg
    sims_pid_srt = sims_pid
    for s, sim in enumerate(sims_nrg):
        ind = sim[:, comp].argsort()[::-1]
        sims_pid_srt[s] = sims_pid[s][ind]
        sims_nrg_srt[s] = sim[ind]

    return(sims_nrg_srt, sims_pid_srt)


def plt_pid_group(sims_nrg, sims_pid, fig, ax, m, c, plt3=None):
    unique = np.unique(sims_pid[:, :m])
    for p, pid in enumerate(unique):
        sims = sims_nrg[np.where(sims_pid == pid)]
        x = sims[:, 0]
        y = sims[:, 1]
        z = sims[:, 2]

        ci = 0
        if not plt3:
            ax.plot(y, x, z, label=int(pid))
        else:
            if plt3[0] == 1:
                ax[ci].plot(y, x, c=c[p], label=int(pid), linewidth=0.1)
                ci += 1
            if plt3[1] == 1:
                ax[ci].plot(z, x, c=c[p], label=int(pid), linewidth=0.1)
                ci += 1
            if plt3[2] == 1:
                ax[ci].plot(y, z, c=c[p], label=int(pid), linewidth=0.1)
                ci += 1
        leg = plt.legend(loc='right', bbox_to_anchor=(2, 0.5))

        for line in leg.get_lines():
            line.set_linewidth(8)
    return(ax, fig)


def merge_pids(sims_nrg, sims_pid, m):
    intxn = sims_pid[:, :m]
    unique, counts = np.unique(intxn, return_counts=True)
    pids_matric = []
    for s, pids in enumerate(sims_pid):
        pids_matric.append(np.isin(unique, pids[:m]) * 1)
        # s_nrg = np.isin(pids)

    pids_matric = np.array(pids_matric)
    # print(np.where(sims_pid == (unique*pids_matric)))
    return(unique, pids_matric)


def make_simList(batchList, desList, limit=5):

    simList = []
    txt = '''
        MATCH (n:Sim) WHERE n.sim_name =~ $name
        RETURN distinct n.sim_name
        '''
    with driver.session() as session:
        for b in batchList:
            for d in desList:
                if not d == '.*':
                    name = '.*{0}.*{1:04d}'.format(b, d)
                else:
                    name = '.*{0}.*{1}'.format(b, d)
                if limit:
                    txt_l = txt + 'limit {}'.format(limit)
                    result = session.run(txt_l, name=name)

                result = session.run(txt, name=name)

                for record in result:
                    simList.append(record.values()[0])
    return(simList)


def out_dataframe(sims_nrg, sims_pid, simListAbb):
    df = pd.DataFrame()
    print(sims_nrg)
    print(sims_pid)
    pidU = np.unique(sims_pid)
    for i, pi in enumerate(pidU):
        id = np.where(sims_pid == pi)
        for j, sj in enumerate(simListAbb):
            row = np.append(sims_nrg[id][j], [pi])
            dfi = pd.DataFrame(sims_nrg[id], columns=['IE', 'ti', 'tn'])
            df = df.append(dfi)
        # print(pi)
    df = df.reset_index(drop=True)
    # print(df)
    # print(sims_nrg.shape)


if __name__ == '__main__':
    oem = 'YARIS'

    if oem == 'CEVT':
        uri = "neo4j://ivory:7687"
        driver = GraphDatabase.driver(uri, auth=("neo4j", "ivory123"))
        KG.neo4j_bolt('7687', 'ivory')

        nrmList = '.*'
        pids_sel = [18620110, 18620090, 18620080, 18620120, 55021060, 55021040,
                    18620070, 18620130, 10020210]

    elif oem == 'PAG':
        uri = "neo4j://localhost:3687"
        driver = GraphDatabase.driver(uri, auth=("neo4j", "ivory123"))
        KG.neo4j_bolt('3687', 'ivory')

        nrmList = 'ROB_VOWA_505.*'
        pids_sel = [20002000, 20001400, 20004400, 20001800]
        pids_sel += [20004900, 20005200, 20005300]
        # pids_sel = [70110101]
        # pids_sel = [88100114, 88100104]
        pids_sel += [
            20002600, 74500851, 74516431, 88000004,
            88000010, 88001008, 88001010]
        # pids_sel = [20002000, 20002600]
        pids_sel = [
            20002000, 20001400, 20004400]  # 20001800, 20001100, 20002600, 20004500
        pids_sel = [20002000]

        sim_batches = [
            '505', '721', '506', '722',
            '507', '726', '508', '735',
            '715', '736', '716', '737',
            '717', '739', '718', '740',
            '719', '741', '720', '742']
        # simList = [
        #     'ROB_VOWA_717_Design0028',
        #     'ROB_VOWA_722_Design0017',
        #     'ROB_VOWA_726_Design0017',
        # ]
        simList = make_simList(sim_batches[0], ['.*'], 5)

    # --------------------------------
    norm_list = [
        "ie_tot_max",  # 0
        "ie_prt_max",  # 1
        "ke_tot_max",  # 2

        "ie_ti_ll",  # 3
        "ti_ll_pp",  # 4
        "ie_ti_grad",  # 5
        "ti_grad_pp",  # 6

        "ie_tn_pct",  # 7
        "tn_pct_max",  # 8
        "ie_tn_max",  # 9
        "tn_max_max"]  # 10

    feat_list = [
        "nrg_max",  # 0

        "ti_grad",  # 1
        "ti_ll",  # 2

        "tn_pct",  # 3
        "tn_max"]  # 4

    norm_opt = [norm_list[i] for i in [2, 4, 10]]
    norm_opt = []
    ft_opt = [feat_list[i] for i in [0, 1, 3]]

    # simList = make_simList(['.*'], ['.*'], 4)
    # simList = simList[:2]
    # --------------------------------
    # sims_nrg, sims_pid = feed_normalization(
    #     nrmList, simList, norm_opt, ft_opt, pids_sel=None)
    # print(sims_pid)
    # sims_nrg, sims_pid = sort_nrg(sims_nrg, sims_pid, comp=2)
    # plt_nrg_embed(
    #     sims_nrg[:], sims_pid[:], simList, m=20, plt3=[1, 1, 1], grp=None)
    # leg='on'
    # plt_nrg_embed_3d(sims_nrg[:], sims_pid[:], simList, m=10, grp=None)
    # --------------------------------
    # pids_m, pids_matric = merge_pids(sims_nrg, sims_pid, m=5)
    # --------------------------------
    # sims_nrg_diff, sims_pid_diff = nrg_diff(sims_nrg, sims_pid)  # fltr=[3e6, 0, 0]
    # pids_m, pids_matric = merge_pids(sims_nrg_diff, sims_pid_diff, m=5)
    # print(pids_m*pids_matric)
    # plt_nrg_embed(
    #     sims_nrg_diff[:], sims_pid_diff[:], simList, m=5, plt3=[1, 1, 1])

    driver.close()
