import _paths
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

import oems
import KG_feed_OEM_data as kg
import inv_simRnk_extnd as invS
import re

import os
from matplotlib.transforms import TransformedBbox
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox
from matplotlib.legend_handler import HandlerBase
from matplotlib._png import read_png

from matplotlib.offsetbox import OffsetImage, AnnotationBbox


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


def plt_ie_parts_all(sim, fig):

    dst = '../publication/20_thesis/Figures/plt_ie_parts_{}_all.png'.format(
        sim)

    MEDIUM_SIZE = 10
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    pathB = '/home/ndv/stud/data/YARIS/full_front/CCSA_submodel/crash_modes/CCSA_submodel_000{}/'.format(
        sim)

    # fig1 = plt.figure(fig)
    fig1, ax = plt.subplots()

    binout = kg.get_binout(pathB)
    t = binout.read('glstat', 'time')
    curv = binout.read('glstat', 'internal_energy')
    s = plt.plot(t*1000, curv/1e6, '.2')

    curv_i = binout.read('matsum', 'internal_energy')
    ids = binout.read('matsum', 'ids')
    t = binout.read('matsum', 'time')

    curvAbs = np.absolute(curv_i.T)
    idsMax = np.argsort([max(x) for x in curvAbs])[::-1]
    npart = 18
    c = list(reversed(sns.color_palette("Spectral_r", npart+2).as_hex()))

    # GET THE NAMES
    names = binout.read('matsum', 'legend')
    names2 = []

    # get part names
    for x in re.findall('.{70}', names):
        try:
            names2.append(x.split()[0])
        except IndexError:
            names2.append('')

    leg = ['KE']
    HNDL = {}
    plt_line = [s[0]]
    for ei, i in enumerate(idsMax):
        ci = curv_i.T[i]
        pi = plt.plot(t*1000, ci/1e6, c[ei])
        s = plt.scatter(130, 0, c='1')

        # ab = AnnotationBbox(
        #     getImage(
        #         "../dash-nrg/assets/YARIS/CCSA_submodel_000{}_{}_iso0.png".format(sim, ids[i])),
        #     (t[-1]*1000*(ei+1)/10, ci[-1]/1e6),
        #     frameon=False)
        # ax.add_artist(ab)

        leg.append(ids[i])  # names2[i])
        if ei > npart:
            break

        # setup the handler instance for the scattered data
        custom_handler1 = ImageHandler()
        custom_handler1.set_image("../dash-nrg/assets/YARIS/CCSA_submodel_000{}_{}_iso0.png".format(sim, ids[i]),
                                  image_stretch=(10, 10))  # this is for grace hopper
        HNDL[s] = custom_handler1
        plt_line.append(s)

        # add the legend for the scattered data, mapping the
        # scattered points to the custom handler

    fig1.set_size_inches(4, 5)

    l = plt.legend(plt_line,
                   leg,
                   handler_map=HNDL,
                   labelspacing=0.7,
                   prop={'size': MEDIUM_SIZE},
                   loc='upper right',
                   bbox_to_anchor=(1, .95),
                   ncol=2,
                   facecolor='0.7',
                   frameon=True)
    c = ['.2'] + c
    for ei, text in enumerate(l.get_texts()):
        text.set_color(c[ei])

    # l = plt.legend(leg, prop={'size': MEDIUM_SIZE}, facecolor='0.6')

    plt.ylabel('$KE\ [MNmm]$')
    plt.xlabel('$t \ [ms]$')
    plt.subplots_adjust(left=0.12, right=0.97, top=0.97, bottom=0.1)
    # ax1.set_xticks(range(0, 17, 2))
    # ax1.set_yticks(range(0, 21, 5))
    fig1.savefig(dst)


def plt_ie_parts(sim, fig):

    dst = '../publication/20_thesis/Figures/plt_ie_parts_{}.png'.format(
        sim)

    MEDIUM_SIZE = 10
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels

    pathB = '/home/ndv/stud/data/YARIS/full_front/CCSA_submodel/crash_modes/CCSA_submodel_60{}/'.format(
        sim)

    # fig1 = plt.figure(fig)
    fig1, ax = plt.subplots()

    binout = kg.get_binout(pathB)
    t = binout.read('glstat', 'time')
    curv = binout.read('glstat', 'internal_energy')
    s = plt.plot(t*1000, curv/1e6, '.2')

    curv_i = binout.read('matsum', 'internal_energy')
    ids = binout.read('matsum', 'ids')
    t = binout.read('matsum', 'time')

    curvAbs = np.absolute(curv_i.T)
    idsMax = np.argsort([max(x) for x in curvAbs])[::-1]
    npart = 5
    c = list(reversed(sns.color_palette("Spectral_r", npart+2).as_hex()))

    # GET THE NAMES
    names = binout.read('matsum', 'legend')
    names2 = []

    # get part names
    for x in re.findall('.{70}', names):
        try:
            names2.append(x.split()[0])
        except IndexError:
            names2.append('')

    leg = ['KE']
    HNDL = {}
    plt_line = [s[0]]
    for ei, i in enumerate(idsMax):
        ci = curv_i.T[i]
        pi = plt.plot(t*1000, ci/1e6, c[ei])
        s = plt.scatter(130, 0, c='1')

        ab = AnnotationBbox(
            getImage(
                "../dash-nrg/assets/YARIS/CCSA_submodel_0004_{}_iso0.png".format(ids[i])),
            (t[-1]*1000*(ei+1)/10, ci[-1]/1e6),
            frameon=False)
        ax.add_artist(ab)

        leg.append(ids[i])  # names2[i])
        if ei > npart:
            break

        # setup the handler instance for the scattered data
        custom_handler1 = ImageHandler()
        custom_handler1.set_image("../dash-nrg/assets/YARIS/CCSA_submodel_0004_{}_iso0.png".format(ids[i]),
                                  image_stretch=(10, 10))  # this is for grace hopper
        HNDL[s] = custom_handler1
        plt_line.append(s)

        # add the legend for the scattered data, mapping the
        # scattered points to the custom handler

    fig1.set_size_inches(4, 3.5)

    # l = plt.legend(plt_line,
    #                leg,
    #                handler_map=HNDL,
    #                labelspacing=0.7,
    #                prop={'size': MEDIUM_SIZE},
    #                loc='upper right',
    #                bbox_to_anchor=(1, .95),
    #                ncol=2,
    #                facecolor='0.7',
    #                frameon=True)
    # c = ['.2'] + c
    # for ei, text in enumerate(l.get_texts()):
    #     text.set_color(c[ei])

    l = plt.legend(leg, prop={'size': MEDIUM_SIZE},
                   loc='upper right',
                   bbox_to_anchor=(1, .95), facecolor='0.7',
                   frameon=True)

    plt.ylabel('$KE\ [MNmm]$')
    plt.xlabel('$t \ [ms]$')
    plt.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.15)
    # ax1.set_xticks(range(0, 17, 2))
    # ax1.set_yticks(range(0, 21, 5))
    fig1.savefig(dst)


class ImageHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):

        # enlarge the image by these margins
        sx, sy = self.image_stretch

        # create a bounding box to house the image
        bb = Bbox.from_bounds(xdescent - sx,
                              ydescent - sy,
                              width + sx,
                              height + sy)

        tbb = TransformedBbox(bb, trans)
        image = BboxImage(tbb)
        image.set_data(self.image_data)

        self.update_prop(image, orig_handle, legend)

        return [image]

    def set_image(self, image_path, image_stretch=(0, 0)):
        self.image_data = read_png(image_path)
        self.image_stretch = image_stretch


def getImage(path):
    return OffsetImage(plt.imread(path, format="png"), zoom=.05)


def test_leg_pic():

    # random data
    x = np.random.randn(100)
    y = np.random.randn(100)
    y2 = np.random.randn(100)

    # plot two series of scatter data
    s = plt.scatter(x, y, c='b')
    s2 = plt.scatter(x, y2, c='r')

    # setup the handler instance for the scattered data
    custom_handler1 = ImageHandler()
    custom_handler1.set_image("../dash-nrg/assets/YARIS/CCSA_submodel_0004_3000005_iso0.png",
                              image_stretch=(10, 10))  # this is for grace hopper
    custom_handler2 = ImageHandler()
    custom_handler2.set_image("../dash-nrg/assets/YARIS/CCSA_submodel_0004_3000503_iso0.png",
                              image_stretch=(10, 10))  # this is for grace hopper

    # add the legend for the scattered data, mapping the
    # scattered points to the custom handler
    plt.legend([s, s2],
               ['Scatters 1', 'Scatters 2'],
               handler_map={s: custom_handler1, s2: custom_handler2},
               labelspacing=2,
               frameon=False)


if __name__ == '__main__':
    # plt_ke()
    # plt_ie_parts_all(4, 1)
    plt_ie_parts('03', 1)
    plt_ie_parts('30', 2)
    plt_ie_parts('31', 3)
    # test_leg_pic()

    plt.show()
