import _paths
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import oems
import KG_feed_OEM_data as kg
import inv_simRnk_extnd as invS


def plt_ke():

    dst = '../publication/20_thesis/Figures/KE_cmprsn.pdf'

    MEDIUM_SIZE = 10
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    full_model = '/home/ndv/stud/data/YARIS/full_front/CCSA_loadpath/CCSA_loadpath_0001/'
    sub_model = '/home/ndv/stud/data/YARIS/full_front/CCSA_submodel/CCSA_submodel_0003/'
    sub_model_500 = '/home/ndv/stud/data/YARIS/full_front/CCSA_submodel/CCSA_submodel_0004/'

    fig1 = plt.figure(1)
    path = [full_model, sub_model, sub_model_500]
    c = ['0.6', 'r', 'b', 'c', 'm', 'y', 'k']
    for i, pi in enumerate(path):
        binout = kg.get_binout(pi)
        t = binout.read('glstat', 'time')
        ids = binout.read('glstat', 'ids')
        curv = binout.read('glstat', 'kinetic_energy')

        plt.plot(t*1000, curv/1e6, c[i])

    plt.legend([
        'complete vehicle', 'submodel\nwith no added mass', 'submodel\n500 kg added mass'
    ], prop={'size': MEDIUM_SIZE})

    plt.ylabel('$KE\ [MNmm]$')
    plt.xlabel('$t \ [ms]$')
    fig1.set_size_inches(3.7, 2)
    plt.subplots_adjust(left=0.2, right=0.97, top=0.97, bottom=0.22)
    # ax1.set_xticks(range(0, 17, 2))
    # ax1.set_yticks(range(0, 21, 5))
    fig1.savefig(dst)
    plt.show()


if __name__ == '__main__':
    plt_ke()
