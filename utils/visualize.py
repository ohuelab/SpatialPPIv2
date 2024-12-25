from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np


def visualize_contact_map(coords, distance_threshold=8):
    distanceMatrix = distance_matrix(coords, coords)
    sns.heatmap(distanceMatrix<distance_threshold, cmap='viridis')


def visualize_data(data):
    alen = data.data_shape[0] + data.data_shape[1]
    map = torch.zeros([alen, alen])
    for j, i in enumerate(data.edge_index.T):
        map[i[0], i[1]] = data.edge_attr[j]
    sns.heatmap(map, cmap='viridis')


def visualize_attentions(input_data, attention_weights, vmin, vmax):
    aws = [[i.cpu().numpy() for i in aw] for aw in attention_weights]
    rows = 4
    cols = 4
    f, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
    alen = input_data.data_shape[0] + input_data.data_shape[1]
    overall = np.zeros([alen, alen])

    for r in range(rows):
        row_sum = np.zeros([alen, alen])
        for c in range(cols):
            ax=axes[r][c]
            edge_index = aws[r][0]
            edge_attr = aws[r][1][:, c]
            
            map = np.zeros([alen, alen])
            for j, i in enumerate(edge_index.T):
                map[i[0], i[1]] = edge_attr[j]
            # if r == 3 and c == 3:
            #     sns.heatmap(map, cmap='viridis', vmin=vmin, vmax=vmax, ax=ax, cbar=True, cbar_ax=f.add_axes([0.93, 0.15, 0.01, 0.7]))
            # else:
            sns.heatmap(map, cmap='viridis', vmin=vmin, vmax=vmax, ax=ax, cbar=False)
            ax.title.set_text(f'GAT[{r}] Head[{c}]')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")
            overall += map
            row_sum += map
    plt.show()
