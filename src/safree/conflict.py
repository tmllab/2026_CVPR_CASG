import torch
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
import torch
import matplotlib.pyplot as plt

def visualize_directions(directions, labels, title):
    """
    3D visualization of concept directions with distinct colors and arrow orientation.
    """
    if not os.path.exists("fig"):
        os.makedirs("fig")
    directions = torch.stack(directions) if isinstance(directions, (list, tuple)) else directions

    # PCA to 3D if needed
    if directions.shape[-1] > 3:
        pca = PCA(n_components=3)
        directions_3d = pca.fit_transform(directions.cpu().numpy())
    else:
        directions_3d = directions.cpu().numpy()

    # norm
    norms = np.linalg.norm(directions_3d, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    directions_3d = directions_3d / norms

    # colors
    cmap = plt.cm.get_cmap("tab10") 
    colors = [cmap(i % 10) for i in range(len(directions_3d))]

    # setup figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='z', labelsize=18)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]["linewidth"] = 0
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    light_gray = (0.7, 0.7, 0.7)
    ax.xaxis.line.set_color(light_gray)
    ax.yaxis.line.set_color(light_gray)
    ax.zaxis.line.set_color(light_gray)
    
    # plot
    for i, vec in enumerate(directions_3d):
        ax.quiver(
            0, 0, 0, vec[0], vec[1], vec[2],
            length=1.0, normalize=True,
            arrow_length_ratio=0.2,
            color=colors[i],
            label=labels[i] if labels else f"Direction {i}"
        )
        ax.scatter(vec[0], vec[1], vec[2], color=colors[i], s=50, edgecolor='k')
    
    ax.set_title(f"{title}", fontsize=14)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0))

    lim = 1.00   # 1.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    plt.tight_layout()
    plt.savefig(f"fig/{title}.pdf", dpi=300, bbox_inches="tight", transparent=True)
    plt.show()
    plt.close()
    
    print(f"3D direction visualization saved to fig/{title}.pdf")
  
def vis_direction_conflict(prompt_embs, proj_embs_list, overall_proj_embeds, guidance_type):
    safety_direction_list = []
    for proj_embs in proj_embs_list:
        safety_direction_list.append(
            (proj_embs - prompt_embs).flatten()
        )
    safety_direction_list.append(
        (overall_proj_embeds - prompt_embs).flatten()
    )
    
    # plot
    harmful_label = ['hate', 'harassment', 'violence', 'self-harm', 'sexual', 'shocking', 'illegal', 'overall']
    title = f"{guidance_type}_safety_direction_3d"
    visualize_directions(safety_direction_list, harmful_label, title)

def vis_direction_attenuation(g_overall, g_k, guidance_type):
    """
    Visualizes how the overall vector at each timestep decomposes into projections along given direction vectors.
    
    Args:
        g_overall (torch.Tensor): Overall vector [D]
        g_k (List[torch.Tensor] or torch.Tensor): Direction vectors [K, D]
        guidance_type (str): Guidance type for the plot title and filename
    """
    g_overall = g_overall.flatten()
    g_k = [gk.flatten() for gk in g_k]
    
    K = len(g_k)
    device = g_overall.device

    projection_strengths = torch.zeros((K), device=device)


    for k, gki in enumerate(g_k):
        # normalize direction vectors
        gk = gki / (gki.norm() + 1e-8)

        # compute projections and their signed magnitudes
        proj_vec = (torch.dot(g_overall, gk)) * gk
        signed_proj = torch.sign(torch.dot(g_overall, gk)) * proj_vec.norm()
        projection_strengths[k] = signed_proj / (g_overall.norm() + 1e-8)

    projection_strengths_np = projection_strengths.detach().cpu().numpy()
        
    if not os.path.exists("fig"):
        os.makedirs("fig")
    # plt.figure(figsize=(max(10, T * 0.4), 2 + 0.1 * K))
    plt.figure(figsize=(10, 2.5))
    ax = sns.heatmap(
        # conver to 2D
        projection_strengths_np.reshape(1, -1),
        cmap="YlGnBu",
        xticklabels=["hate", "harassment", "violence", "self-harm", "sexual", "shocking", "illegal"],
        annot=False
    )
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Harmful Category", fontsize=12)
    ax.tick_params(axis='x', which='major', labelsize=14)
    ax.tick_params(axis='y', which='major', labelsize=14)
    # ax.set_title(f"Aggregated Directional Attenuation Heatmap ({guidance_type})", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"fig/{guidance_type}_direction_attenuation_heatmap.pdf", dpi=300)
    plt.close()

    print(f"Direction attenuation heatmap saved to fig/{guidance_type}_direction_attenuation_heatmap.pdf")
    # write data to txt
    if not os.path.exists("log"):
        os.makedirs("log")
    np.savetxt(f"log/{guidance_type}_direction_attenuation_data.txt", projection_strengths_np, fmt="%.6f")
    
