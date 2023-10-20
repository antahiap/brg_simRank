import matplotlib.pyplot as plt
# import numpy as np


def scatter_hoover(x, y, names, c, fig, ax, sim_name, z=None, leg=None, marker='o'):
    # x = np.random.rand(15)
    # y = np.random.rand(15)
    # names = np.array(list("ABCDEFGHIJKLMNO"))
    # c = np.random.randint(1,c,size=len(x))
    # norm = plt.Normalize(1,4)
    # cmap = plt.cm.RdYlGn

    # fig,ax = plt.subplots()
    # sc = plt.scatter(x,y,c=c, s=10, cmap=cmap, norm=norm)
    if z is not None:
        sc = ax.scatter(x, y, z, c=c, label=sim_name)
        if leg:
            plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5))
    else:
        sc = ax.scatter(x, y, c=c, s=20, label= sim_name, marker=marker)
        if leg:
            plt.legend()

    annot = ax.annotate("", xy=(0, 0), xytext=(-10, 10),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        # arrowprops=dict(arrowstyle="->")
                        )
    annot.set_visible(False)

    def update_annot(ind):

        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        # text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))),
        #                     " ".join([names[n] for n in ind["ind"]]))
        text = "{}\n{}".format(" ".join([names[n] for n in ind["ind"]])
                                , sim_name)
        # Annotate the 2nd position with another image (a Grace Hopper portrait)
        annot.set_text(text)
        # print(text)
        # annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
        annot.get_bbox_patch().set_alpha(0.4)
        # p5 = ax.plot(pos, "r--")


    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    return(fig, ax)
    # plt.show()

# if __name__ == '__main__':
#     xx = np.random.rand(15)
#     yy = np.random.rand(15)
#     name = np.array(list("ABCDEFGHIJKLMNO"))
#     scatter_hoover(xx, yy, name)
