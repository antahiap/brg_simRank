import enum
from re import T
import _paths
import glob
import math
import time

import numpy as np
import pandas as pd

import oems
import KG_feed_OEM_data as kg


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


if __name__ == '__main__':

    tst = TestSimRank()
    # tst.test_sim_grnd_truth_selective_part()
    tst.test_sim_grnd_truth_all_time()
    # tst.test_sim_grnd_truth_single_time()
    # tst.test_read_dsplcmnt_yaris()
