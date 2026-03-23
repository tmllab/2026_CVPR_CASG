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

def vis_direction_conflict(uncon_pred, prompt_pred, harmful_pred_list, overall_harmful_pred, guidance_type, step):
    safety_direction_list = []
    for harmful_pred in harmful_pred_list:
        safety_direction_list.append(
            (prompt_pred - harmful_pred).flatten()
        )
    safety_direction_list.append(
        (prompt_pred - overall_harmful_pred).flatten()
    )
    
    # plot
    harmful_label = ['hate', 'harassment', 'violence', 'self-harm', 'sexual', 'shocking', 'illegal', 'overall']
    title = f"{guidance_type}_safety_direction_3d_step{step}"
    visualize_directions(safety_direction_list, harmful_label, title)

def vis_direction_attenuation(g_overall_list, g_k_list, guidance_type):
    """
    Visualizes how the overall vector at each timestep decomposes into projections along given direction vectors.
    
    Args:
        g_overall_list (List[torch.Tensor]): Overall vectors at each timestep [T, D]
        g_k_list (List[List[torch.Tensor]] or torch.Tensor): Direction vectors per timestep [T, K, D]
        guidance_type (str): Guidance type for the plot title and filename
    """
    # select key timesteps
    # key_timesteps = list(range(0, len(g_overall_list), max(1, len(g_overall_list)//10)))
    key_timesteps = [ i for i in range(0, len(g_overall_list), 2) ]
    g_k_list = [g_k_list[t] for t in key_timesteps]
    g_overall_list = [g_overall_list[t] for t in key_timesteps]
    
    # select key categories
    concept_labels = ["hate", "harassment", "violence", "self-harm", "sexual", "disturbing", "illegal"]
    key_categories = [i for i in range(7)]  # e.g., hate, sexual, illegal
    g_k_list = [[g_k_list[t][k] for k in key_categories] for t in range(len(g_k_list))]
    
    T = len(g_overall_list)
    K = len(g_k_list[0])
    device = g_overall_list[0].device

    projection_strengths = torch.zeros((T, K), device=device)

    for t in range(T):
        # get direction vectors at timestep t
        g_overall = g_overall_list[t]
        gks = g_k_list[t]

        # normalize direction vectors
        gks_normed = [gk / (gk.norm() + 1e-8) for gk in gks]

        # compute projections and their signed magnitudes
        for k, gk in enumerate(gks_normed):
            proj_vec = (torch.dot(g_overall, gk)) * gk
            signed_proj = torch.sign(torch.dot(g_overall, gk)) * proj_vec.norm()
            projection_strengths[t, k] = signed_proj / (g_overall.norm() + 1e-8)

    projection_strengths_np = projection_strengths.detach().cpu().numpy()
    
    if not os.path.exists("fig"):
        os.makedirs("fig")
    # plt.figure(figsize=(max(10, T * 0.4), 2 + 0.1 * K))
    plt.figure(figsize=(10, 3.5))
    ax = sns.heatmap(
        projection_strengths_np.T,
        cmap="YlGnBu",
        # cbar_kws={'label': 'Category-wise Directional Retention (larger means more retention)'},
        xticklabels=key_timesteps,
        yticklabels=[concept_labels[k] for k in key_categories],
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

@torch.no_grad()
def select_strongest_harmful_direction(
    noise_pred_uncond: torch.Tensor,
    noise_pred_text: torch.Tensor,
    noise_pred_harmful_list: list[torch.Tensor],
    eps: float = 1e-12,
):
    """
    selects the strongest harmful direction among all harmful directions based on their similarity with the prompt direction.
    returns:
        g_neg: the strongest harmful direction
        idx_max: the index of the strongest harmful direction
        scores: the scores of all harmful directions (similarity with the prompt direction)
    """
    delta = (noise_pred_text - noise_pred_uncond).flatten()
    if len(noise_pred_harmful_list) == 0:
        return torch.zeros_like(delta), None, torch.tensor([])

    H = torch.stack([(h - noise_pred_uncond).flatten() for h in noise_pred_harmful_list])  # [N, D]
    norms = H.norm(dim=1) + eps
    
    scores = (H @ delta) / norms
    idx_max = torch.argmax(scores)
    g_neg = H[idx_max]
    
    return g_neg, idx_max.item(), scores
