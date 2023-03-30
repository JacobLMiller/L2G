import pylab as plt
import numpy as np
import pickle 

# pickleFile = open("../data/03_19_2.pkl","rb")
# data = pickle.load(pickleFile)
# pickleFile.close()

pickleFile = open("../data/update_tsnet.pkl","rb")
tsdata = pickle.load(pickleFile)
pickleFile.close()

# data = data | tsdata

pickleFile = open("../data/03_28_unioned.pkl","rb")
data = pickle.load(pickleFile)
pickleFile.close()

for key in tsdata['tsne']:
    if key in data['tsne'] and tsdata['tsne'][key]["NE"] is not None:
        data['tsne'][key]["NE"] = tsdata['tsne'][key]["NE"][0]
        data['tsne'][key]["m1"] = tsdata['tsne'][key]["m1"][0] 
        data['tsne'][key]["stress"] = tsdata['tsne'][key]["stress"][0]       



def update_colors(colors,row,i,jet,neutral):
    colors[i*2,:] = neutral
    colors[i*2,1] = jet(row[0])
    colors[i*2,2] = jet(row[1])
    colors[i*2,4] = jet(row[2])
    colors[i*2,5] = jet(row[3])

    ind = i*2+1
    colors[ind,:] = neutral
    colors[ind,0] = jet(row[4])
    colors[ind,3] = jet(row[5])
    colors[ind,6] = jet(row[6])
    return colors


def draw_table(metric="NE"):
    tab_vals = list()

    graphs = list(data['l2g'].keys())

    cmap = plt.get_cmap("gist_yarg")
    # cmap = plt.get_cmap("RdYlGn_r")
    colors = np.zeros( (len(graphs) * 2,7,4) )

    i = 0
    row_to_cell = {
        0: np.array([0,1]),
        1: np.array([0,2]),
        2: np.array([0,4]), 
        3: np.array([0,5]),
        4: np.array([1,0]),                       
        5: np.array([1,3]),
        6: np.array([1,5]),
    }
    mins = list()
    for graph in graphs:

        if data['l2g'][graph][metric] is None: continue
        tab_vals.append(list())
        tab_vals.append(list())

        tsne = data['tsne'][graph][metric]
        umap = data['umap'][graph][metric]
        mds = data["mds"][graph][metric]
        l2g = data['l2g'][graph][metric]
        l2g1, l2g2, l2g3, l2g4 = l2g[2], l2g[3], l2g[4], l2g[6]

        row_data = [l2g1,l2g2,l2g3,l2g4,tsne,umap,mds]
        normal = plt.Normalize(min(row_data),max(row_data))
        jet = lambda x: cmap(normal(x))
        colors = update_colors(colors,row_data,i,jet,cmap(0.5))

        min_loc = row_to_cell[np.argmin(row_data)].copy()
        min_loc[0] += (i*2)
        mins.append(min_loc)

        fmt = lambda s: str(s)[:6]
        tab_vals[-1] = [fmt(tsne), "", "", fmt(umap), "", "", fmt(mds)]
        tab_vals[-2] = ["", fmt(l2g1), fmt(l2g2), "", fmt(l2g3), fmt(l2g4), ""]
        i += 1


        

    

    # rows = [g.split(".")[0] for g in graphs if "connected_watts_1000" not in g]
    rows = list()
    for g in graphs:
        # if "connected_watts_1000" in g: continue
        rows.append(g.split(".")[0])
        rows.append("")

    columns = ["tsNET","k=16", "k=32", "umap", "k=64", "k=100","MDS"]


    # cell_data = np.array(tab_vals)
    # jet = plt.get_cmap("gist_yarg")
    # colors = np.zeros( (cell_data.shape[0],cell_data.shape[1],4) )
    # for i in range(cell_data.shape[0]):
    #     col_data = cell_data[i]
    #     normal = plt.Normalize(np.min(col_data),np.max(col_data))
    #     colors[i,:] = jet(normal(col_data))

    # mins = np.argmin(cell_data,axis=1)
    # cell_data = [[str(s)[:6] for s in row] for row in cell_data]

    # if metric == "m1":
    #     bad = [rows.index("12square"),
    #         rows.index("powerlaw300"),
    #         rows.index("494_bus"),
    #         rows.index("btree9"),
    #         rows.index("EVA")]
        # for i in bad:
        #     cell_data[i] = ["N/A" for _ in cell_data[i]]
        #     colors[i,:] = jet(0.5)

    colors[:,:,3] = 0.55
    fig,ax = plt.subplots()
    ax.axis("off")
    ax.axis("tight")
    print(len(rows))
    table = ax.table( 
        tab_vals,
        rowLabels=rows,
        colLabels=columns,
        cellColours=colors,
        loc="center",
        colWidths=[0.1 for _ in columns]
    )
    
    for cell in mins:
        table[cell[0]+1,cell[1]].set_text_props(weight="bold")

    # fig.tight_layout()
    fig.set_figheight(15)
    fig.set_figwidth(15)
    table.scale(1,0.55)

    # table.auto_set_font_size(False)
    # table.set_fontsize(24)
    # table.scale(2,2)

    plt.savefig(f"figures/{metric}_table_grey.pdf")
    

if __name__ == "__main__":
    draw_table('NE')
    draw_table('stress')
    draw_table('m1')