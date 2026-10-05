import os
import matplotlib.animation as animation
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def animate_network(SIMU, save_path, save_frames_dir):
    
    COLOR_PATIENT = "#4A90E2"
    COLOR_IMPATIENT = "#D9534F"
    COLOR_ACTED = "#F39C12"
    COLOR_EDGE = "#BDC3C7"
    COLOR_SIGNAL = "#FFD700"

    net = SIMU.network
    G = net.G
    pos = net.pos

    ordered_agents = [agent for withdrawal_round in SIMU.withdrawn_order for agent in withdrawal_round]
    ordered_nodes = [net.agent_to_node[agent] for agent in ordered_agents]

    # Create frames directory if requested
    if save_frames_dir:
        os.makedirs(save_frames_dir, exist_ok=True)

    # Figure setup with extra right margin space for legend
    fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
    fig.subplots_adjust(right=0.75)  # Reserve 25% of the right side for legend
    ax.axis("off")
    ax.set_aspect("equal")

    # Draw static edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=COLOR_EDGE, style="dashed", width=1.5, alpha=0.8)

    # Initial node colors
    node_color_map = {}
    for node in G.nodes():
        agent = net.agents_by_node[node]
        node_color_map[node] = (COLOR_PATIENT if agent.is_patient else COLOR_IMPATIENT)

    nodes_list = list(G.nodes())
    node_x = [pos[n][0] for n in nodes_list]
    node_y = [pos[n][1] for n in nodes_list]
    colors = [node_color_map[n] for n in nodes_list]

    nodes_scatter = ax.scatter(node_x, node_y, c=colors, s=380, zorder=3, edgecolors="#2C3E50", linewidths=1.2)

    labels = ({node: i for i, node in enumerate(G.nodes())} if net.network_type == "grid" else {node: str(node) for node in G.nodes()})
    for n, (x, y) in pos.items():
        ax.text(x, y, labels[n], color="white", fontsize=10, fontweight="bold", ha="center", va="center", zorder=4,)

    ax.set_title("Depositor Network Interaction", loc="left", fontweight="bold", pad=20, fontsize=16)

    signals_scatter = ax.scatter([], [], c=COLOR_SIGNAL, s=10, zorder=5, edgecolors="#D4AC0D")

    STEPS_PER_ACTION = 10
    total_frames = len(ordered_nodes) * STEPS_PER_ACTION

    # Legend placement fix
    legend_handles = [mpatches.Patch(color=COLOR_PATIENT, ec="#2C3E50", label=r"Patient Depositor ($\omega_i = 1$)"),
                      mpatches.Patch(color=COLOR_IMPATIENT, ec="#5B1208", label=r"Impatient Depositor ($\omega_i = 0$)"),
                      mpatches.Patch(color=COLOR_ACTED, ec="#B9770E", label=r"Acted / Withdrawn Node"),
                      mlines.Line2D([],[], color=COLOR_SIGNAL, marker="o", ls="", ms=8, label="Interaction Signal")]

    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.03, 0.5),  facecolor="#F8F9F9", edgecolor="#BDC3C7", fontsize=10, borderpad=0.8, labelspacing=1.0)

    def update(frame):
        node_idx = frame // STEPS_PER_ACTION
        step = frame % STEPS_PER_ACTION

        current_node = ordered_nodes[node_idx]
        neighbors = list(G.neighbors(current_node))

        # Node color updates
        current_colors = [COLOR_ACTED if n in ordered_nodes[: node_idx + 1] else node_color_map[n] for n in nodes_list]
        nodes_scatter.set_facecolors(current_colors)

        # Pulse signal update
        if step < STEPS_PER_ACTION - 1 and len(neighbors) > 0:
            alpha = (step + 1) / STEPS_PER_ACTION
            start_pos = np.array(pos[current_node])

            sig_x, sig_y = [], []
            for target in neighbors:
                end_pos = np.array(pos[target])
                curr_pos = start_pos + alpha * (end_pos - start_pos)
                sig_x.append(curr_pos[0])
                sig_y.append(curr_pos[1])

            signals_scatter.set_offsets(np.c_[sig_x, sig_y])
        else:
            signals_scatter.set_offsets(np.empty((0, 2)))

        # Save individual frame as image
        if save_frames_dir:
            frame_filename = os.path.join(save_frames_dir, f"frame_{frame:04d}.png")
            fig.savefig(frame_filename, dpi=150, bbox_inches="tight")

        return nodes_scatter, signals_scatter

    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=100, blit=False)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ani.save(save_path, writer="ffmpeg", fps=10, dpi=150)

    plt.show()
    return ani