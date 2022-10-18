import _paths
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import oems
import inv_simRnk_extnd as invS


def add_ref(fig, val, legend=True):
    fig.add_trace(go.Scatter(
        mode='markers',
        x=[val[0]], y=[val[1]],
        name=val[2],
        showlegend=legend,
        marker=dict(
            color=val[3],
            size=12)
    ), col=val[4], row=1)

    return(fig)


def plt_dataset():
    dst = '../publication/06_KG_energyAbsorption/submition/nrg_extnd_6000.pdf'

    ref = {
        '3': 'blue', '30': 'red', '31': 'lime', '60': 'gold', '61': '#9A0eea'}

    data_path = "extnd_mdl/sims_extnd_06.pkl"
    sims = pd.read_pickle(data_path)
    sims.id = sims.id - 6000
    fig = px.scatter(sims, x='tL_u', y='tR_u',
                     labels={
                         "lc pair": "sim pair",
                         'tL_u': 'T<sub>2</sub> LHS [mm]',
                         'tR_u': 'T<sub>2</sub> RHS [mm]',
                     },
                     width=440, height=400)
    fig.update_traces(
        marker_size=7,
        marker_color='gray'
    )
    fig.update_layout(
        font_size=12
    )
    fig.update_yaxes(tick0=1.5, dtick=0.2)
    fig.update_xaxes(tick0=1.5, dtick=0.2)

    for p in ref:
        color = ref[p]
        r = sims.loc[sims.id == int(p)]
        x = r.tL_u.values[0]
        y = r.tR_u.values[0]
        fig = add_ref(fig, [x, y, str(p), color, 1])

    print(dst)
    fig.write_image(dst)
    fig.show()


def plt_simrankpp():

    OEM = 'YARIS'
    oem = oems.oems(OEM)

    nSim, pidMax, wTag = 7, 25, 'P_e'
    wscl, sprd, evd, C = 1e9, 'trgt', 2, .8

    dst = '../publication/06_KG_energyAbsorption/submition/nrg_extnd_simrankpp_6000.pdf'

    nFrmt = '"CCSA_submodel_[6].*"'
    nFrmtM = '""'
    ref = {
        '6003': 'blue', '6030': 'red', '6031': 'lime', '6060': 'gold', '6061': '#9A0eea'}

    data_path = "extnd_mdl/sims_extnd_06.pkl"
    sims = pd.read_pickle(data_path)

    ext = invS.ExtndSimRank()
    ext.query = oem.query(oem)
    sim_sR = ext.simRank_perRef(nFrmt, nFrmtM,
                                sims, ref, oem,
                                nSim,
                                pidMax, wTag, wscl,
                                sprd, evd, C)

    tags = []
    for r in sim_sR['lc pair']:
        if not r == '':
            tags.append(str(int(r)-6000))
        else:
            tags.append(r)
    sim_sR['lc pair'] = tags
    ref1 = {
        '3': 'blue', '30': 'red', '31': 'lime', '60': 'gold', '61': '#9A0eea'}

    color_discrete_map = {**ref1, '': 'silver'}
    fig = px.scatter(sim_sR, x='tL_u', y='tR_u',
                     color='lc pair',
                     color_discrete_map=color_discrete_map,
                     #  symbol_sequence=['arrow-left'],
                     labels={
                         "lc pair": "",
                         'tL_u': 'T<sub>2</sub> LHS [mm]',
                         'tR_u': 'T<sub>2</sub> RHS [mm]',
                         '6003': '3'
                     },
                     width=460, height=400)
    fig.update_traces(
        marker_size=7,
    )
    fig.update_layout(
        font_size=14,
        margin=dict(t=0, r=0, l=0, b=0, pad=5),
    )
    fig.update_yaxes(tick0=1.5, dtick=0.2)
    fig.update_xaxes(tick0=1.5, dtick=0.2)

    # add reference marker
    for p in ref:
        color = ref[p]
        r = sims.loc[sims.id == int(p)]
        x = r.tL_u.values[0]
        y = r.tR_u.values[0]
        fig = add_ref(fig, [x, y, str(p), color, 1], legend=False)

    print(dst)
    fig.write_image(dst)
    fig.show()


def plt_rank_Pe():

    OEM = 'YARIS'
    oem = oems.oems(OEM)
    nFrmt1 = '""'
    nFrmt = '"CCSA_submodel_{}"'
    lmt = 28
    nSim = 7

    dst = '../publication/06_KG_energyAbsorption/submition/nrg_extnd_Pe_6000.pdf'

    ref = {
        '6003': 'blue', '6030': 'red', '6031': 'lime', '6060': 'gold', '6061': '#9A0eea'}
    keys = [int(k) for k, v in ref.items()]

    data_path = "extnd_mdl/sims_extnd_06.pkl"
    sims = pd.read_pickle(data_path)

    ext = invS.ExtndSimRank()
    ext.query = oem.query(oem)
    sim_rIE = ext.sort_IE_sim(
        nFrmt, nFrmt, keys, sims, lmt, nSim=7)

    tags = []
    for r in sim_rIE['lc pair']:
        if not r == '':
            tags.append(str(int(r)-6000))
        else:
            tags.append(r)
    sim_rIE['lc pair'] = tags
    ref1 = {
        '3': 'blue', '30': 'red', '31': 'lime', '60': 'gold', '61': '#9A0eea'}

    ref1 = {**ref1, '': 'silver'}
    sim_rIE = sim_rIE.sort_values(by=['lc pair'])
    fig = px.scatter(sim_rIE, x='tL_u', y='tR_u',
                     color='lc pair',
                     color_discrete_map=ref1,
                     labels={"lc pair": ""},
                     width=400, height=400)
    fig.update_traces(
        marker_size=7,
        showlegend=False
    )
    fig.update_layout(
        font_size=14,
        margin=dict(t=0, r=0, l=0, b=0, pad=5),
    )
    fig.update_yaxes(tick0=1.5, dtick=0.2)
    fig.update_xaxes(tick0=1.5, dtick=0.2)

    for p in ref:
        color = ref[p]
        r = sims.loc[sims.id == int(p)]
        x = r.tL_u.values[0]
        y = r.tR_u.values[0]
        fig = add_ref(fig, [x, y, str(p), color, 1], legend=False)
    print(dst)
    fig.write_image(dst)
    fig.show()


if __name__ == '__main__':

    # plt_dataset()
    plt_simrankpp()
    # plt_rank_Pe()
