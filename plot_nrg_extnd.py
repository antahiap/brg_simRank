import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


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


def plt_dataset():
    dst = '../publication/06_KG_energyAbsorption/submition/nrg_extnd_6000.pdf'

    ref = {
        '6003': 'blue', '6030': 'red', '6031': 'lime', '6060': 'yellow', '6061': '#9A0eea'}

    data_path = "extnd_mdl/sims_extnd_06.pkl"
    sims = pd.read_pickle(data_path)
    fig = px.scatter(sims, x='tL_u', y='tR_u',
                     #  color='lc pair',  # size='markerSize',
                     #  hover_data=["id", 'similarity', 'tL_i', 'tR_i'],
                     #  facet_col="dt_ui",  # "data_id",  #
                     #  symbol='data_id',
                     #  color_discrete_map=ref,
                     #  #  symbol_sequence=symbol_map,
                     labels={
                         "lc pair": "sim pair",
                         'tL_u': 'T<sub>2</sub> LHS [mm]',
                         'tR_u': 'T<sub>2</sub> RHS [mm]',
                     },
                     width=450, height=400)
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


if __name__ == '__main__':

    plt_dataset()
