import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from datetime import datetime
import numpy as np
import random


class Network:
    def __init__(self, environment, network_type="random", p=0.2, number_of_clusters=2, p_inside=0.7, p_outside=0.1, seed=None):

        self.agents = environment.agents
        self.N = environment.N
        self.Lambda = environment.Lambda

        self.network_type = network_type

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if (network_type == "grid") and (int(np.sqrt(self.N))**2 != self.N):
            self.network_type = 'random'
            print('Grid type must be an squared number of individuals')

        if self.network_type == "grid":
            self.G = self._grid_network()

        elif self.network_type == "clustered":
            self.number_of_clusters = number_of_clusters
            self.p_inside = p_inside
            self.p_outside = p_outside
            self.G = self._clustered_network()
            

        else: # elif: network_type == "random":
            self.network_type = 'random'
            self.p = p
            self.G = nx.gnp_random_graph(self.N, p, seed=seed)

        # Ambos almacenan la misma información
        self.agents_by_node = {}
        for node, agent in zip(self.G.nodes(), self.agents):
            self.agents_by_node[node] = agent

        for node, agent in self.agents_by_node.items():
            agent.neighbors = [self.agents_by_node[neighbor_node] for neighbor_node in self.G.neighbors(node)]

        self.agent_to_node = {agent: node for node, agent in self.agents_by_node.items()}
        
    def _grid_network(self):       
        G = nx.grid_2d_graph(int(np.sqrt(self.N)), int(np.sqrt(self.N)))
    
        # Add diagonal neighbourhood connections
        for x in range(int(np.sqrt(self.N)) - 1):
            for y in range(int(np.sqrt(self.N)) - 1):
                G.add_edge((x, y),(x + 1, y + 1))
                G.add_edge((x, y + 1),(x + 1, y))
    
        return G

    def _clustered_network(self):
        # 1. División equilibrada de tamaños de clusters
        base_size = self.N // self.number_of_clusters
        sizes = [base_size] * self.number_of_clusters
        sizes[-1] += self.N - sum(sizes)  # Ajuste para el remanente de N
    
        # 2. Construcción de la matriz de probabilidades de enlace (SBM)
        # P_ij = p_inside si i == j, de lo contrario p_outside
        probs = [
            [self.p_inside if i == j else self.p_outside 
             for j in range(self.number_of_clusters)]
            for i in range(self.number_of_clusters)
        ]
    
        # 3. Generación estocástica garantizando que no existan nodos aislados
        while True:
            G = nx.stochastic_block_model(sizes, probs, seed=random.randint(0, 1000000))
            
            # Verificar que el grafo sea completamente conectado o no tenga nodos de grado 0
            if all(deg >= 1 for _, deg in G.degree()):
                break
    
        # 4. Asignar atributo de cluster a los nodos (útil para métricas y visualización)
        start_idx = 0
        for cluster_id, size in enumerate(sizes):
            for node in range(start_idx, start_idx + size):
                G.nodes[node]['cluster'] = cluster_id
            start_idx += size
    
        return G

    def graph(self, save = False):
        COLOR_PATIENT = "#4A90E2"
        COLOR_IMPATIENT = "#D9534F"
        COLOR_EDGE = "#BDC3C7"
    
        patient_nodes = []
        impatient_nodes = []
    
        for node in self.G.nodes():
            agent = self.agents_by_node[node]
    
            if agent.is_patient:
                patient_nodes.append(node)
            else:
                impatient_nodes.append(node)
    

        if self.network_type == "grid":
            self.pos = {node: node for node in self.G.nodes()}
        elif self.network_type == "clustered":
            self.pos = nx.spring_layout(self.G, k=0.15, iterations=100, seed=42)
        else:
            self.pos = nx.spring_layout(self.G, seed=42)

        fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
        
        nx.draw_networkx_edges(self.G, self.pos, ax=ax, edge_color=COLOR_EDGE, style="dashed", width=1.5, alpha=0.8)
        nx.draw_networkx_nodes(self.G, self.pos, ax=ax, nodelist=patient_nodes, node_color=COLOR_PATIENT, edgecolors="#2C3E50", node_size=380, linewidths=1.2)
        nx.draw_networkx_nodes(self.G, self.pos, ax=ax, nodelist=impatient_nodes, node_color=COLOR_IMPATIENT, edgecolors="#5B1208", node_size=380, linewidths=1.2)

        if self.network_type == "grid":
            labels = {node: i for i, node in enumerate(self.G.nodes())}
            nx.draw_networkx_labels(self.G, self.pos, ax=ax, labels=labels, font_size=11, font_color="white", font_weight="bold")
        else:
            nx.draw_networkx_labels(self.G, self.pos, ax=ax, font_size=11, font_color="white", font_weight="bold")

        
        ax.set_title("Depositor Interaction Network Topology", loc="left", fontweight="bold", pad=20, fontsize=18)
        if self.network_type == 'clustered':
            ax.text(0, 1, rf"Model Parameters: $N = {self.N}$, $\lambda = {self.Lambda}$, {self.network_type.capitalize()} Network ($C = $ {self.number_of_clusters}, $p_i =$ {self.p_inside}, $p_o =$ {self.p_outside})",transform=ax.transAxes, fontsize=15, ha="left", va="bottom", color="#555555")
        elif self.network_type == 'grid':
            ax.text(0, 1, rf"Model Parameters: $N = {self.N}$, $\lambda = {self.Lambda}$, {self.network_type.capitalize()} Network ({int(self.N**0.5)}x{int(self.N**0.5)})",transform=ax.transAxes, fontsize=15, ha="left", va="bottom", color="#555555")
        else:
            ax.text(0, 1, rf"Model Parameters: $N = {self.N}$, $\lambda = {self.Lambda}$, {self.network_type.capitalize()} Network ($p = $ {self.p})",transform=ax.transAxes, fontsize=15, ha="left", va="bottom", color="#555555")
            
        ax.axis("off")
        ax.set_aspect("equal")

        legend_handles = [mpatches.Patch(color=COLOR_PATIENT, ec="#2C3E50", label=r"Patient Depositor ($\omega_i = 1$)"), mpatches.Patch(color=COLOR_IMPATIENT, ec="#5B1208", label=r"Impatient Depositor ($\omega_i = 0$)"), mlines.Line2D([], [], color=COLOR_EDGE, lw=1.8, ls="--", label=r"Interaction Channels ($g_{ij}$)")]

        ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1, frameon=False, fontsize=13, labelspacing=1.2, handlelength=2.2, handleheight=1.5)
        
        plt.tight_layout()
        if save:
            plt.savefig(f'outputs/MODEL/FIG{datetime.now().strftime("%d-%m-%y.%H.%M.%S")}.png', dpi=300, bbox_inches='tight', transparent=True)
        plt.show()

    def mean_degree(self):
        return sum(dict(self.G.degree()).values()) / self.N

    def density(self):
        return nx.density(self.G)
