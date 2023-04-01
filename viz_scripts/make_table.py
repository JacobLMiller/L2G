import pylab as plt
import numpy as np
import pickle 

fileObj = open('../data/03_28_unioned.pkl', 'rb')
data = pickle.load(fileObj)
fileObj.close()

pickleFile = open("../data/update_tsnet.pkl","rb")
tsdata = pickle.load(pickleFile)
pickleFile.close()

for key in tsdata['tsne']:
    if key in data['tsne'] and tsdata['tsne'][key]["NE"] is not None:
        data['tsne'][key]["NE"] = tsdata['tsne'][key]["NE"][0]
        data['tsne'][key]["m1"] = tsdata['tsne'][key]["m1"][0] 
        data['tsne'][key]["stress"] = tsdata['tsne'][key]["stress"][0]    

def get_tab_vals(metric,is_l2g=True):
    tab_vals = list()
    if is_l2g:
        graphs = list(data['l2g'].keys())
        for graph in graphs:
            if data['l2g'][graph][metric] is None: continue
            tab_vals.append(list())
            row = tab_vals[-1]

            # row.append(data['l2g'][graph][metric][1])
            row.append(data['l2g'][graph][metric][2])
            row.append(data['l2g'][graph][metric][3])
            row.append(data['l2g'][graph][metric][4])
            row.append(data['l2g'][graph][metric][5])
            row.append(data['l2g'][graph][metric][7])
            row.append(data['l2g'][graph][metric][9])
            row.append(data['l2g'][graph][metric][9])
            row.append(data['tsne'][graph][metric])
            row.append(data['umap'][graph][metric])
            row.append(data['mds'][graph][metric])

        return tab_vals


def draw_table(metric="NE",columns=list(),is_l2g=True,skip=5):
    tab_vals = get_tab_vals(metric,is_l2g)

    graphs = list(data['l2g'].keys())    
    
    rows = [g.split(".")[0] for g in graphs]


    cell_data = np.array(tab_vals)
    jet = plt.get_cmap("gist_yarg")
    # jet = plt.get_cmap("RdYlGn_r")
    colors = np.zeros( (cell_data.shape[0],cell_data.shape[1],4) )
    for i in range(cell_data.shape[0]):
        col_data = cell_data[i]
        l2g_data = cell_data[i,0:skip]
        normal = plt.Normalize(np.min(l2g_data),np.max(l2g_data))
        colors[i,0:skip] = jet(normal(l2g_data))

        other_data = cell_data[i,skip+1:]
        normal = plt.Normalize(np.min(other_data),np.max(other_data))
        colors[i,skip+1:] = jet(normal(other_data))

        normal = plt.Normalize(np.min(col_data),np.max(col_data))
        colors[i,:] = jet(normal(col_data))

    mins = np.argmin(cell_data,axis=1)
    cell_data = [[str(s)[:6] for s in row] for row in cell_data]

    cell_data = np.array(cell_data,dtype=str)

    if metric == "m1":
        noclusters = [
            "powerlaw300",
            "price_1000",
            "btree9",
            "CSphd",
            "EVA"
        ]
        noclusters = [rows.index(s) for s in noclusters]
        cell_data[noclusters,:] = "N/A"
        colors[noclusters,:] = jet(0.)

    colors[:,skip] = np.array([1.,1.,1.,0.])
    
    cell_data[:,skip] = ""

    colors[:,:,3] = 0.55
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
    
    for i,j in enumerate(mins):
        table[i+1,j].set_text_props(weight="bold")

    fig.tight_layout()

    # table.auto_set_font_size(False)
    # table.set_fontsize(24)
    # table.scale(2,2)

    plt.savefig(f"figures/{metric}_table.pdf")
    

if __name__ == "__main__":
    columns = [ "k=32", "k=64", "k=85", "k=100","k=150","k=200", "", "tsnet", "umap", "mds"]

    draw_table('NE',columns,skip=6)
    draw_table('stress',columns,skip=6)
    draw_table('m1',columns,skip=6)