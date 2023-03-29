import pylab as plt
import numpy as np
import pickle 

pickleFile = open("../data/03_19_2.pkl","rb")
data = pickle.load(pickleFile)
pickleFile.close()

pickleFile = open("../data/update_tsnet.pkl","rb")
tsdata = pickle.load(pickleFile)
pickleFile.close()

data = data | tsdata


def draw_table(metric="NE"):
    tab_vals = list()

    graphs = list(data['l2g'].keys())
    for graph in graphs:
        if "connected_watts_1000" in graph: continue
        # if metric not in data['tsne'][graph]: print(graph)
        if data['l2g'][graph][metric] is None: continue
        tab_vals.append(list())
        row = tab_vals[-1]

        if graph not in data['tsne']:
            row.append(1)
        elif "EVA" in graph:
            row.append(1)
        elif data['tsne'][graph][metric] is not None: 
            row.append(data["tsne"][graph][metric][0])
        else: row.append(1)
        row.append(data['l2g'][graph][metric][2])
        row.append(data['l2g'][graph][metric][3])
        row.append(data['umap'][graph][metric][0])
        row.append(data['l2g'][graph][metric][4])
        row.append(data['l2g'][graph][metric][6])        
        row.append(data["mds"][graph][metric][0])

    rows = [g.split(".")[0] for g in graphs if "connected_watts_1000" not in g]
    columns = ["tsNET","k=16", "k=32", "umap", "k=64", "k=100","MDS"]

    print(len(rows))

    cell_data = np.array(tab_vals)
    jet = plt.get_cmap("RdYlGn_r")
    colors = np.zeros( (cell_data.shape[0],cell_data.shape[1],4) )
    for i in range(cell_data.shape[0]):
        col_data = cell_data[i]
        normal = plt.Normalize(np.min(col_data),np.max(col_data))
        colors[i,:] = jet(normal(col_data))

    cell_data = [[str(s)[:6] for s in row] for row in cell_data]

    if metric == "m1":
        bad = [rows.index("12square"),
            rows.index("powerlaw300"),
            rows.index("494_bus"),
            rows.index("btree9"),
            rows.index("EVA")]
        for i in bad:
            cell_data[i] = ["N/A" for _ in cell_data[i]]
            colors[i,:] = jet(0.5)

    fig,ax = plt.subplots()
    ax.axis("off")
    ax.axis("tight")
    table = ax.table( 
        cell_data,
        rowLabels=rows,
        colLabels=columns,
        cellColours=colors,
        loc="center",
        colWidths=[0.1 for _ in columns]
    )
    fig.tight_layout()

    # table.auto_set_font_size(False)
    # table.set_fontsize(24)
    # table.scale(2,2)

    plt.savefig(f"figures/{metric}_table.pdf")
    

if __name__ == "__main__":
    draw_table('NE')
    draw_table('stress')
    draw_table('m1')