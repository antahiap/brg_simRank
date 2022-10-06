import enum
from operator import index
from re import T
import _paths
import glob
import math
import time
import random

import numpy as np
import pandas as pd
import networkx as nx
from neomodel import db
import plotly.express as px
import plotly.graph_objects as go

import oems
import KG_feed_OEM_data as kg
import graph_mining as gm


# class query:
#     def __init__(self, oem):
# self.driver = oem.driver


class TestSimRank():

    def make_diff_mtrx(self, sims, siml, sims_d3plt):

        nsim = len(sims)
        diff = np.zeros((nsim, nsim))
        simDisp = {}
        for si, dispi in sims_d3plt.items():
            for sj, dispj in sims_d3plt.items():

                ii = siml[si]
                jj = siml[sj]

                MSE = np.square(np.subtract(dispi, dispj)).mean()
                RMSE = math.sqrt(MSE)
                diff[ii, jj] = RMSE
                tag = '{}-{}'.format(ii+1, jj+1)
                simDisp[tag] = RMSE
        simDisp_sort = dict(sorted(simDisp.items(), key=lambda item: item[1]))

        return(simDisp_sort)

    def test_sim_grnd_truth_single_time(self):
        '''
        use nodes deformation diff as the similarity measure
        and compare it with simrank++
        '''

        OEM = 'YARIS'
        oem = oems.oems(OEM)
        d3plt = kg.DataD3plot('dispENVS')
        sims = glob.glob(oem.data_path)

        sims_d3plt = {}
        siml = {}
        for i, s in enumerate(sims):
            sim = kg.CaeSim(OEM)
            sim.dataYARIS(s)
            disp = d3plt.read_disp(s, states={-1})
            sims_d3plt[sim.abb] = disp
            siml[sim.abb] = i

        simDisp = self.make_diff_mtrx(sims, siml, sims_d3plt)

        for pair, dist in simDisp.items():
            print(pair, dist)

    def test_sim_grnd_truth_all_time(self):
        '''
        use nodes deformation diff as the similarity measure
        and compare it with simrank++ for all the time steps
        '''
        ts = time.clock()
        OEM = 'YARIS'
        oem = oems.oems(OEM)
        d3plt = kg.DataD3plot('dispENVS')
        sims = glob.glob(oem.data_path)

        sims_d3plt = {}
        siml = {}
        for i, s in enumerate(sims):
            sim = kg.CaeSim(OEM)
            sim.dataYARIS(s)
            disp = d3plt.read_disp(s)
            sims_d3plt[sim.abb] = disp
            siml[sim.abb] = i

        simDisp = self.make_diff_mtrx(sims, siml, sims_d3plt)

        print(time.clock() - ts)
        for pair, dist in simDisp.items():
            print(pair, dist)

    def test_sim_grnd_truth_selective_part(self):
        '''
        use nodes deformation diff as the similarity measure
        and compare it with simrank++ for last time step and
        only 5 energetic part
        '''
        ts = time.clock()
        OEM = 'YARIS'
        oem = oems.oems(OEM)
        d3plt = kg.DataD3plot('dispENVS')
        sims = glob.glob(oem.data_path)

        parts = [2000000,  2000001,  2000002,  2000501,  2000502]

        sims_d3plt = {}
        siml = {}
        for i, s in enumerate(sims):
            sim = kg.CaeSim(OEM)
            sim.dataYARIS(s)
            disp = d3plt.read_disp(s, states={-1}, part_ids=parts)
            sims_d3plt[sim.abb] = disp
            siml[sim.abb] = i

        simDisp = self.make_diff_mtrx(sims, siml, sims_d3plt)

        print(time.clock() - ts)
        for pair, dist in simDisp.items():
            print(pair, dist)

    def test_read_dsplcmnt_yaris(self):
        '''
        read displacement with lasso package for specific oem
        '''

        OEM = 'YARIS'
        oem = oems.oems(OEM)
        d3plt = kg.DataD3plot('dispENVS')
        s = glob.glob(oem.data_path)[0]

        disp = d3plt.read_disp(s)

    def test_sim_GT_IE(self):
        '''
        get energy features and evaluate distance based on IE and P_e
        '''

        OEM = 'YARIS'
        oem = oems.oems(OEM)
        query = oem.query(oem)

        nFrmt = '"CCSA_submodel_000{}"'
        lmt = 30
        sims = [4, 5, 6, 7]

        simIE = {}
        for n0 in sims:
            for n1 in sims:
                if n0 == n1:
                    continue
                sim0 = nFrmt.format(n0)
                df0 = query.nrg_fts(sim0, lmt)
                df0 = df0.sort_values(by=['PID'])

                sim1 = nFrmt.format(n1)
                df1 = query.nrg_fts(sim1, lmt)
                df1 = df1.sort_values(by=['PID'])

                d01_IE = (df1.IE-df0.IE)/1000
                MSE = np.square(d01_IE).mean()
                d01_Pe = (df1.IE/(df1.tn - df1.ti) -
                          df0.IE/(df0.tn - df0.ti))/1e6
                MSE = np.square(d01_Pe).mean()
                RMSE = math.sqrt(MSE)
                tag = '{}-{}'.format(n0-3, n1-3)
                simIE[tag] = RMSE

        simIE_sort = dict(sorted(simIE.items(), key=lambda item: item[1]))

        for pair, dist in simIE_sort.items():
            print(pair, dist)

    def add_ref_all(self, fig, points, df):

        def add_ref(fig, val):
            fig.add_trace(go.Scatter(
                mode='markers',
                x=[val[0]], y=[val[1]],
                name=val[2],
                marker=dict(
                    color=val[3],
                    size=12)
            ), col=val[4], row=1)

            return(fig)

        # fig = add_ref(fig, [1.7, 1.7, '11', 'blue', 1])
        # fig = add_ref(fig, [2.1, 1.7, '12', 'yellow', 1])
        # fig = add_ref(fig, [1.7, 1.5, '13', 'red', 1])
        # fig = add_ref(fig, [1.5, 1.7, '14', 'green', 1])
        # fig = add_ref(fig, [1.7, 2.1, '15', '#9A0eea', 1])

        # fig = add_ref(fig, [1.9, 1.9, '21', 'blue', 1])
        # fig = add_ref(fig, [2.3, 1.9, '22', 'yellow', 1])
        # fig = add_ref(fig, [1.9, 1.7, '23', 'red', 1])
        # fig = add_ref(fig, [1.7, 1.9, '24', 'green', 1])
        # fig = add_ref(fig, [1.9, 2.3, '25', '#9A0eea', 1])
        m = 0
        for p in points:
            color = points[p]
            if color == 'blue':
                m += 1
            r = df.loc[df.id == int(p)]
            x = r.tL_u.values[0]
            y = r.tR_u.values[0]
            fig = add_ref(fig, [x, y, str(p), color, m])
        return(fig)

    def simRank_pair(
        self, nFrmt, nFrmtM, simExt, key_ref, oem,
        wscl=1e7,  # 1e7,  # 1e9,
        evd=False,
        sprd=False,
        M2='""'
    ):
        simExt = simExt.reset_index(drop=True)
        cypherTxt = self.query.simrank_G(
            nFrmt, nFrmtM, self.pidMax, simM2=M2, simList=simExt.name.to_list())
        # wscl scaling the weight
        G = gm.get_graph(cypherTxt, '', w=wscl, driver=oem.driver)
        _, _, spp, top = gm.simRankpp(
            G, self.pidMax, 0.0, wscl=wscl, sprd=sprd, evd=evd,
            importance_factor=0.9)

        sName = nx.get_node_attributes(G, 'name')
        keys = [k for k, v in sName.items() if v in key_ref]
        spp_filter = spp[keys]

        # sim_tst = [nodeId for nodeId,
        #            name in sName.items() if name in ['3023', '6051']]
        # for s in sim_tst:
        #     input(spp_filter[:, s])
        # input(spp_filter[:, sim_tst].T)

        slct_subm = np.argmax(spp_filter, axis=0)
        slct_subm_2 = np.argsort(
            spp_filter, kind='mergesort', axis=0)[-2, :]

        simExt['lc pair'] = ''
        simExt['similarity'] = 0.0
        for i, v in enumerate(slct_subm_2):
            sNi = sName[i]
            try:
                r = simExt[simExt.id == int(sNi)].index[0]
            except:
                continue
            spp_v = spp_filter[v, i]
            if spp_v > 0.59:
                simExt.at[r, 'lc pair'] = sName[keys[v]]
                simExt.at[r, 'similarity'] = spp_v

        # simExt['lc pair'] = ''
        # simExt['similarity'] = 0.0
        # for ii, r in simExt.iterrows():
        #     id = r.id
        #     index_sim = [nodeId for nodeId,
        #                  name in sName.items() if name == str(id)]
        #     if not index_sim == []:
        #         si = index_sim[0]
        #         input(slct_subm[si])
        #         sim_pair = slct_subm[index_sim][0]
        #         sim_pair2 = slct_subm_2[index_sim][0]
        #         spp_v = spp_filter[sim_pair, index_sim][0]
        #         if spp_v > 0.4:
        #             simExt.at[ii, 'lc pair'] = sName[keys[sim_pair]]
        #             simExt.at[ii, 'similarity'] = spp_v
        #         simExt.at[ii, 'lc pair 2'] = sName[keys[sim_pair2]]
        #         simExt.at[ii, 'similarity2'] = spp_filter[sim_pair2,
        #                                                   index_sim][0]
        #     else:
        #         simExt.at[ii, 'lc pair'] = 'miss result'
        #         simExt.at[ii, 'similarity'] = 0.0
        #         print('MISSING REF MODEL')
        #         print(id)
        return(simExt)

    def simRank_perRef(
        self, nFrmt, nFrmtM, simExt, key_ref, oem,
        n=5,
        wscl=1e7,  # 1e7,  # 1e9,
        evd=False,
        sprd=False,
        M2='""'
    ):
        simExt = simExt.reset_index(drop=True)
        cypherTxt = self.query.simrank_G(
            nFrmt, nFrmtM, self.pidMax, simM2=M2, simList=simExt.name.to_list())
        # wscl scaling the weight
        G = gm.get_graph(cypherTxt, '', w=wscl, driver=oem.driver)

        _, _, spp, top = gm.simRankpp(
            G, self.pidMax, 0.0, wscl=wscl, sprd=sprd, evd=evd,
            importance_factor=0.9)

        sName = nx.get_node_attributes(G, 'name')
        keys = [k for k, v in sName.items() if v in key_ref]
        spp_filter = spp[keys]

        # slct_subm = np.argmax(spp_filter, axis=0)
        slct_subm = np.argsort(
            spp_filter, axis=1)[:, -n-1:-1]  # [:, -7:]  # , kind='mergesort'

        simExt['lc pair'] = ''
        simExt['similarity'] = 0.0

        sims = np.unique(slct_subm)
        # input(n*len(keys))
        # n0 = n
        # while len(sims) < n0*len(keys):
        #     n += 1
        #     sims = np.unique(slct_subm[:, -n-1:-1])

        count = [0 for k in keys]
        for si in sims:
            sNi = sName[si]

            rows = np.where(slct_subm == si)[0]
            max_index = np.argmax(spp_filter[rows, si])
            ri = rows[max_index]
            ki = keys[ri]
            pair = sName[ki]

            try:
                r = simExt[simExt.id == int(sNi)].index[0]
            except:
                continue
            spp_v = spp_filter[ri, si]
            simExt.at[r, 'lc pair'] = pair
            simExt.at[r, 'similarity'] = spp_v

        return(simExt)

    def test_simRank_extnd(self):

        OEM = 'YARIS'
        oem = oems.oems(OEM)
        self.query = oem.query(oem)
        self.pidMax = 4

        # nFrmt = '"CCSA_submodel_10.*"'
        # nFrmtM = '"CCSA_submodel_00[2].*"'
        # '0021': 'blue', '0022': 'yellow', '0023': 'red', '0024': 'green', '0025': '#9A0eea',
        # key_ref1 = ['0021', '0024', '0023', '0022', '0025']
        # random.shuffle(key_ref1)
        # simExt10 = pd.read_pickle("extnd_mdl/sims_extnd_01.pkl")
        # simExt10['data_id'] = 1
        # simExt1 = self.simRank_pair(nFrmt, nFrmtM, simExt10, key_ref1)

        # nFrmt = '"CCSA_submodel_2.*"'
        # nFrmtM = '"CCSA_submodel_00[1].*"'
        # ref_2 = {
        #     '2001': 'blue', '2008': 'yellow', '2009': 'red', '2064': 'green', '2065': '#9A0eea'}
        # key_ref2 = [k for k, v in ref_2.items()]
        # random.shuffle(key_ref2)
        # simExt20 = pd.read_pickle("extnd_mdl/sims_extnd_02.pkl")
        # simExt20['data_id'] = 2
        # simExt2 = self.simRank_pair(nFrmt, nFrmtM, simExt20, key_ref2, oem)

        ref_3 = {
            '3003': 'blue', '3006': 'red', '3007': 'green', '3064': 'yellow', '3065': '#9A0eea'}
        nFrmt = '"CCSA_submodel_3.*"'
        nFrmtM = '"CCSA_submodel_00[2].*"'
        key_ref3 = [k for k, v in ref_3.items()]
        random.shuffle(key_ref3)
        simExt30 = pd.read_pickle("extnd_mdl/sims_extnd_03.pkl")
        simExt30['data_id'] = 3

        simExt30_fltr = simExt30.loc[(
            simExt30.tL_u < 5) & (simExt30.tR_u < 5)]
        simExt30_fltr = simExt30_fltr.loc[(
            simExt30_fltr.tL_u > 1.6) | (simExt30_fltr.tR_u > 1.6)]

        simExt3 = self.simRank_pair(nFrmt, nFrmtM,
                                    simExt30_fltr, key_ref3, oem)  # , M2='"CCSA_submodel_002.*"')

        ref_5 = {
            '5000': 'blue',
            '5006': 'red', '5007': 'green', '5050': 'yellow', '5051': '#9A0eea'}
        nFrmt = '"CCSA_submodel_5.*"'
        nFrmtM = '"CCSA_submodel_5.*"'
        key_ref5 = [k for k, v in ref_5.items()]
        random.shuffle(key_ref3)
        simExt50 = pd.read_pickle("extnd_mdl/sims_extnd_05.pkl")
        simExt50['data_id'] = 5

        simExt5 = self.simRank_pair(nFrmt, nFrmtM,
                                    simExt50, key_ref5, oem)  # , M2='"CCSA_submodel_002.*"')

        # nFrmt = '"CCSA_submodel_40.*"'
        # nFrmtM = '"CCSA_submodel_00[12].*"'
        # key_ref4 = ['0011', '0015', '0013', '0014', '0012']
        # random.shuffle(key_ref4)
        # simExt40 = pd.read_pickle("extnd_mdl/sims_extnd_04.pkl")
        # simExt40['data_id'] = 4

        # simExt40_fltr = simExt40.loc[(
        #     simExt40.tL_u < 2.6) & (simExt40.tR_u < 2.6)]

        # simExt4 = self.simRank_pair(nFrmt, nFrmtM, simExt40_fltr, key_ref4)

        # nFrmt = '"CCSA_submodel_[34].*"'
        # nFrmtM = '"CCSA_submodel_00[12].*"'
        # key_ref5 = key_ref3 + key_ref4
        # simExt5 = pd.concat([simExt30, simExt40])
        # simExt5 = simExt5.reset_index(drop=True)
        # simExt5 = simExt5.assign(dt_ui='0.4-0.6')
        # simExt5 = self.simRank_pair(nFrmt, nFrmtM, simExt5, key_ref5)

        color_discrete_map = {**ref_5}  # **ref_2, **ref_3, }  # , **ref_5}
        # , simExt3])  # simExt1, simExt2,, simExt5
        simExt = pd.concat([simExt5])  # simExt2, simExt3,  # ])
        symbol_map = ['arrow-left', 'arrow-right', 'square', 'circle']
        simExt = simExt.sort_values(by=['dt_ui', 'lc pair'])

        fig = px.scatter(simExt, x='tL_u', y='tR_u',
                         color='lc pair',  # size='markerSize',
                         hover_data=["id", 'similarity', 'tL_i', 'tR_i',
                                     'similarity2', 'lc pair 2'],
                         facet_col="data_id",  # dt_ui",
                         symbol='data_id',
                         color_discrete_map=color_discrete_map,
                         symbol_sequence=symbol_map,
                         labels={"lc pair": "sim pair"},
                         width=600, height=500)

        fig = self.add_ref_all(fig, color_discrete_map, simExt)

        fig.show()

    def test_simRank_extnd_IE(self):
        '''
        get energy features and evaluate distance based on IE and P_e
        '''

        def sort_IE_sim(query, nFrmt, nFrmt1, sims, simExt, lmt):

            simList = simExt.id.tolist()
            simPair, similarity = [], []
            for n1 in simList:
                d = simExt.loc[simExt.id == n1]
                tL, tR = d.tL_u, d.tR_u
                sub4 = []
                for n0 in sims:
                    sim0 = nFrmt.format(n0)
                    df0 = query.nrg_fts(sim0, lmt)
                    df0 = df0.sort_values(by=['PID'])

                    sim1 = nFrmt1.format(n1)
                    df1 = query.nrg_fts(sim1, lmt)
                    try:
                        df1 = df1.sort_values(by=['PID'])
                    except KeyError:
                        continue

                    if not df1.PID.tolist() == df0.PID.tolist():
                        print('ERR IN MATCHIN PIDs')
                    d01_IE = (df1.IE-df0.IE)/1000
                    MSE = np.square(d01_IE).mean()
                    # d01_Pe = (df1.IE/(df1.tn - df1.ti) -
                    #           df0.IE/(df0.tn - df0.ti))/1e6
                    # MSE = np.square(d01_Pe).mean()
                    RMSE = math.sqrt(MSE)

                    sub4.append(RMSE)
                if not sub4 == []:
                    s = np.argmin(sub4)     # most similar with least distance
                    simP, simV = str(sims[s]), sub4[s]
                else:
                    simP, simV = 0, 0

                simPair.append(simP)
                similarity.append(simV)

            simExt['lc pair'] = simPair
            simExt['similarity'] = similarity

            return(simExt)

        OEM = 'YARIS'
        oem = oems.oems(OEM)
        query = oem.query(oem)

        nFrmt = '"CCSA_submodel_00{}"'
        nFrmt1 = '"CCSA_submodel_{}"'
        lmt = 28
        read = True  # False  #
        outPath = "./sims_extnd_IE_rmse_45_mixed.pkl"

        if read:
            # sims_1 = [11, 12, 13, 14, 15]
            # random.shuffle(sims_1)
            # simExt_1 = pd.read_pickle("extnd_mdl/sims_extnd_04.pkl")
            # simExt_1['data_id'] = 3
            # simExt_1 = sort_IE_sim(query, nFrmt, nFrmt1, sims_1, simExt_1, lmt)

            sims_2_j = {
                '3001': 'blue', '3020': 'red', '3021': 'green', '3060': 'yellow', '3061': '#9A0eea'}
            sims_2 = [int(k) for k, v in sims_2_j.items()]
            random.shuffle(sims_2)
            simExt_2 = pd.read_pickle("extnd_mdl/sims_extnd_03.pkl")
            simExt_2['data_id'] = 4
            simExt_2 = sort_IE_sim(query, nFrmt1, nFrmt1,
                                   sims_2, simExt_2, lmt)

            sims_3_j = {
                '5001': 'blue', '5020': 'red', '5021': 'green', '5060': 'yellow', '5061': '#9A0eea'}
            sims_3 = [int(k) for k, v in sims_3_j.items()]
            random.shuffle(sims_3)
            simExt_3 = pd.read_pickle("extnd_mdl/sims_extnd_05.pkl")
            simExt_3['data_id'] = 5
            simExt_3 = sort_IE_sim(query, nFrmt1, nFrmt1,
                                   sims_3, simExt_3, lmt)

            color_discrete_map = {**sims_2_j, **sims_3_j}

            # simExt_12 = pd.concat([simExt_1, simExt_2])
            # simExt_12 = simExt_12.assign(dt_ui='0.4-0.6')
            # sims_3 = sims_1 + sims_2
            # simExt_12 = simExt_12.reset_index(drop=True)
            # simExt_3 = sort_IE_sim(query, nFrmt, nFrmt1,
            #    sims_3, simExt_12, lmt)

            simExt = pd.concat([simExt_2, simExt_3])  # , simExt_2, simExt_3])
            simExt.to_pickle(outPath)

        simExt = pd.read_pickle(outPath)
        simExt = simExt.sort_values(by=['lc pair', 'data_id'])

        # color_discrete_map = {'11': 'blue', '12': 'gold', '13': 'red', '14': 'green', '15': '#9A0eea', 0: 'gray',
        #                       '21': 'blue', '22': 'gold', '23': 'red', '24': 'green', '25': '#9A0eea'}
        symbol_map = ['arrow-left', 'arrow-right']

        fig = px.scatter(simExt, x='tL_u', y='tR_u',
                         color='lc pair',  # size='markerSize',
                         hover_data=["id", 'similarity'], facet_col="data_id",
                         symbol='data_id',
                         color_discrete_map=color_discrete_map,
                         symbol_sequence=symbol_map,
                         labels={"lc pair": "sim pair"},
                         width=600, height=500)

        fig = self.add_ref_all(fig, color_discrete_map, simExt)
        fig.show()

    def test_simRank_diagonal(self):
        OEM = 'YARIS'
        oem = oems.oems(OEM)
        query = oem.query(oem)

        nFrmt = '"CCSA_submodel_.*"'
        pidMax = 5
        wscl = 1e9
        evd = True
        sprd = True
        cypherTxt = query.simrank_G(nFrmt, pidMax)

        # wscl scaling the weight
        G = gm.get_graph(cypherTxt, '', w=wscl)
        _, _, spp, top = gm.simRankpp(
            G, pidMax, 0.0, wscl=wscl, sprd=sprd, evd=evd)

        sName = nx.get_node_attributes(G, 'name')
        key_ref = [
            '0023', '0021', '0024', '0022', '0025']
        keys = [k for k, v in sName.items() if v in key_ref]
        input(keys)

        spp_filter = spp[keys]  # [:, top]
        print(spp_filter)

    def test_simRank_singleSim(self):
        OEM = 'YARIS'
        oem = oems.oems(OEM)
        query = oem.query(oem)

        sim = '3001'
        nFrmt = '"CCSA_submodel_{}"'.format(sim)
        simRef = '"CCSA_submodel_00[12].*"'
        pidMax = 5
        wscl = 1e9  # 1e7
        evd = False
        sprd = False
        cypherTxt = query.simrank_G(nFrmt, simRef, pidMax)

        # wscl scaling the weight
        G = gm.get_graph(cypherTxt, '', w=wscl)
        _, _, spp, top = gm.simRankpp(
            G, pidMax, 0.0, wscl=wscl, sprd=sprd, evd=evd,
            importance_factor=0.9)
        print(spp)

        sName = nx.get_node_attributes(G, 'name')
        print(sName)
        keys = [k for k, v in sName.items() if v == sim][0]

        s_i = spp[:, keys]
        print(s_i)
        ord = np.argsort(s_i).argsort(kind="heapsort")
        print(ord)
        print(sName)

        for i in range(len(s_i)-1, 0, -1):
            ii = np.where(ord == i)[0][0]
            print(sName[ii], s_i[ii])

    def test_simRank_extend_pltAll(self):
        OEM = 'YARIS'
        oem = oems.oems(OEM)
        self.query = oem.query(oem)
        self.pidMax = 5

        ref = {
            '6003': 'blue', '6016': 'red', '6017': 'green', '6064': 'yellow', '6065': '#9A0eea'}

        datasets = [6, 5]  # , 5, 3]  # 1, 2, 3, 4, 5, 6]
        ds_str = ''.join([str(d) for d in datasets])

        nFrmt = '"CCSA_submodel_[{}].*"'.format(ds_str)
        nFrmtM = '""'
        key_ref = [k for k, v in ref.items()]
        random.shuffle(key_ref)

        data_path = "extnd_mdl/sims_extnd_0{}.pkl"

        simExt = pd.DataFrame()
        for i in datasets:
            simExt_i = pd.read_pickle(data_path.format(i))
            simExt_i['data_id'] = i
            simExt = pd.concat([simExt, simExt_i])

        # simExt = simExt.drop_duplicates(subset=['tL_u', 'tR_u'])

        simExt_sR = self.simRank_pair(nFrmt, nFrmtM,
                                      simExt, key_ref, oem)  # , M2='"CCSA_submodel_002.*"')
        # simExt_sR = simExt_sR.loc[(
        #     simExt_sR.tL_u <= 1.9) & (simExt_sR.tR_u <= 1.9)]
        # input(simExt_sR)

        color_discrete_map = {**ref}
        symbol_map = ['arrow-left', 'arrow-right', 'square', 'circle']
        fig = px.scatter(simExt_sR, x='tL_u', y='tR_u',
                         color='lc pair',  # size='markerSize',
                         hover_data=["id", 'similarity', 'tL_i', 'tR_i'],
                         #  'similarity2', 'lc pair 2'],
                         facet_col="dt_ui",  # "data_id",  #
                         #  symbol='data_id',
                         color_discrete_map=color_discrete_map,
                         #  symbol_sequence=symbol_map,
                         labels={"lc pair": "sim pair"},
                         width=1000, height=900)

        # fig.append_trace(
        #     go.Scatter(
        #         x=[1.85, 2.6], y=[2.1, 3.6],
        #         mode='lines',
        #         line={'dash': 'dash', 'color': '#9A0eea'},
        #         showlegend=False
        #     ), 1, 1)
        # fig.append_trace(
        # go.Scatter(
        #     x=[2.1, 3.6], y=[1.85, 2.6],
        #     mode='lines',
        #     line={'dash': 'dash', 'color': 'yellow'},
        #     showlegend=False
        # ), 1, 1)

        fig = self.add_ref_all(fig, color_discrete_map, simExt_sR)

        fig.show()

    def test_simRank_extend_perRef(self):
        OEM = 'YARIS'
        oem = oems.oems(OEM)
        self.query = oem.query(oem)
        self.pidMax = 5

        ref = {
            '6003': 'blue', '6030': 'red', '6031': 'green', '6060': 'yellow', '6061': '#9A0eea'}
        # '5005': 'blue', '1016': 'red', '1017': 'green', '1064': 'yellow', '1065': '#9A0eea'}
        # '6005': 'blue', '6010': 'red',  '6011': 'green',
        # '6064': 'yellow', '6065': '#9A0eea'}

        datasets = [6]  # , 5, 3]  # 1, 2, 3, 4, 5, 6]
        ds_str = ''.join([str(d) for d in datasets])

        nFrmt = '"CCSA_submodel_[{}].*"'.format(ds_str)
        nFrmtM = '""'
        key_ref = [k for k, v in ref.items()]
        random.shuffle(key_ref)

        data_path = "extnd_mdl/sims_extnd_0{}.pkl"

        simExt = pd.DataFrame()
        for i in datasets:
            simExt_i = pd.read_pickle(data_path.format(i))
            simExt_i['data_id'] = i
            simExt = pd.concat([simExt, simExt_i])

        # simExt = simExt.drop_duplicates(subset=['tL_u', 'tR_u'])

        simExt_sR = self.simRank_perRef(nFrmt, nFrmtM,
                                        simExt, key_ref, oem, n=5)  # , M2='"CCSA_submodel_002.*"')
        # simExt_sR = simExt_sR.loc[(
        #     simExt_sR.tL_u <= 1.9) & (simExt_sR.tR_u <= 1.9)]

        color_discrete_map = {**ref, '': 'white'}
        symbol_map = ['arrow-left', 'arrow-right', 'square', 'circle']
        fig = px.scatter(simExt_sR, x='tL_u', y='tR_u',
                         color='lc pair',  # size='markerSize',
                         hover_data=["id", 'similarity', 'tL_i', 'tR_i'],
                         facet_col="dt_ui",  # "data_id",  #
                         symbol='data_id',
                         color_discrete_map=color_discrete_map,
                         #  symbol_sequence=symbol_map,
                         labels={"lc pair": "sim pair"},
                         width=1000, height=900)

        fig = self.add_ref_all(fig, ref, simExt_sR)

        fig.show()

    def test_pick_color(self):

        a = np.array([
            [1, 0.1, 0.2, 0.3, 0.4],
            [0.1, 1, 0.11, 0.21, 0.54],
            [0.2, 0.11, 1, 0.7, 0.8],
            [0.3, 0.21, 0.7, 1, 0.9],
            [0.4, 0.54, 0.8, 0.9, 1]
        ])

        key = [1, 3]

        a_key = a[key]
        print(a_key)
        print(np.argmax(a_key, axis=0))
        # a_color =


if __name__ == '__main__':

    tst = TestSimRank()
    # tst.test_sim_grnd_truth_selective_part()
    # tst.test_sim_grnd_truth_all_time()
    # tst.test_sim_grnd_truth_single_time()
    # tst.test_read_dsplcmnt_yaris()

    # tst.test_sim_GT_IE()
    # tst.test_simRank_singleSim()
    # tst.test_simRank_extnd()
    # tst.test_simRank_extend_pltAll()
    tst.test_simRank_extend_perRef()
    # tst.test_simRank_extnd_IE()

    # tst.test_pick_color()
    # tst.test_simRank_diagonal()
