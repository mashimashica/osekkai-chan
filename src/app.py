import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
# import japanize_matplotlib
from packaging import version as Version

class Agent:
    def __init__(self, is_osekkai=False, is_needy=False, is_supported=False):
        self.is_osekkai = is_osekkai
        self.is_needy = is_needy
        self.is_supported = is_supported

class Model:
    def __init__(self, n_agents, osekkai_rate, needy_rate, support_rate, needy_transition_rate):
        self.agents = []
        self.network = self.create_random_network(n_agents)
        self.needy_transition_rate = needy_transition_rate
        
        for _ in range(n_agents):
            is_osekkai = np.random.random() < osekkai_rate
            is_needy = np.random.random() < needy_rate
            is_supported = is_needy and np.random.random() < support_rate
            self.agents.append(Agent(is_osekkai, is_needy, is_supported))

    def create_random_network(self, n_agents):
        G = nx.Graph()
        G.add_nodes_from(range(n_agents))
        for i in range(n_agents):
            n_edges = np.random.randint(1, 11)
            potential_neighbors = list(set(range(n_agents)) - set([i]) - set(G.neighbors(i)))
            if len(potential_neighbors) < n_edges:
                n_edges = len(potential_neighbors)
            new_neighbors = np.random.choice(potential_neighbors, n_edges, replace=False)
            G.add_edges_from([(i, j) for j in new_neighbors])
        return G

    def step(self):
        for i, agent in enumerate(self.agents):
            if not agent.is_needy and not agent.is_osekkai:
                if np.random.random() < self.needy_transition_rate:
                    agent.is_needy = True
                    agent.is_supported = False
            
            if agent.is_osekkai:
                neighbors = list(self.network.neighbors(i))
                if neighbors:
                    target = np.random.choice(neighbors)
                    target_agent = self.agents[target]
                    if target_agent.is_needy and not target_agent.is_supported:
                        if np.random.random() < 0.5:
                            target_agent.is_supported = True
            
            if agent.is_supported:
                if np.random.random() < 0.2:
                    agent.is_needy = False
                    agent.is_supported = False
                    if np.random.random() < 0.5:
                        agent.is_osekkai = True

    def run(self, steps):
        history = []
        for _ in range(steps):
            self.step()
            stats = self.get_stats()
            history.append(stats)
        return history

    def get_stats(self):
        n_osekkai = sum(1 for agent in self.agents if agent.is_osekkai)
        n_needy = sum(1 for agent in self.agents if agent.is_needy)
        n_supported = sum(1 for agent in self.agents if agent.is_supported)
        total_agents = len(self.agents)
        support_rate = n_supported / n_needy if n_needy > 0 else 0
        return n_osekkai, n_needy, n_supported, support_rate

    def visualize_network(self):
        colors = []
        for agent in self.agents:
            if agent.is_osekkai:
                colors.append('red')
            elif agent.is_needy:
                if agent.is_supported:
                    colors.append('green')
                else:
                    colors.append('blue')
            else:
                colors.append('gray')

        fig, ax = plt.subplots(figsize=(12, 8))
        pos = nx.spring_layout(self.network)
        nx.draw(self.network, pos, node_color=colors, with_labels=False, node_size=30, ax=ax)

        ax.set_title('エージェントネットワークの可視化')
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='おせっかいさん',
                                      markerfacecolor='red', markersize=10),
                           plt.Line2D([0], [0], marker='o', color='w', label='支援を受けている困窮者',
                                      markerfacecolor='green', markersize=10),
                           plt.Line2D([0], [0], marker='o', color='w', label='支援を受けていない困窮者',
                                      markerfacecolor='blue', markersize=10),
                           plt.Line2D([0], [0], marker='o', color='w', label='その他',
                                      markerfacecolor='gray', markersize=10)]
        ax.legend(handles=legend_elements, loc='upper right')
        plt.tight_layout()
        return fig

def run_simulation(n_agents, osekkai_rate, needy_rate, support_rate, needy_transition_rate, steps):
    model = Model(n_agents, osekkai_rate, needy_rate, support_rate, needy_transition_rate)
    history = model.run(steps)
    return model, history

def visualize_results(history):
    osekkai, needy, supported, support_rates = zip(*history)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    ax1.plot(osekkai, label='おせっかいさん')
    ax1.plot(needy, label='困窮者')
    ax1.plot(supported, label='支援を受けている人')
    ax1.set_xlabel('ステップ')
    ax1.set_ylabel('エージェント数')
    ax1.set_title('エージェントの状態の変化')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(support_rates, label='支援率', color='purple')
    ax2.set_xlabel('ステップ')
    ax2.set_ylabel('支援率')
    ax2.set_title('困窮者のうち支援を受けている人の割合')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    return fig

st.title('エージェントベースモデル: おせっかいさんと困窮者')

n_agents = st.slider('エージェント数', 100, 10000, 1000, 100)
osekkai_rate = st.slider('おせっかい率', 0.001, 0.1, 0.005, 0.001)
needy_rate = st.slider('困窮者率', 0.05, 0.3, 0.157, 0.01)
support_rate = st.slider('初期支援率', 0.1, 0.5, 0.2, 0.1)
needy_transition_rate = st.slider('困窮化率', 0.0001, 0.01, 0.001, 0.0001)
steps = st.slider('シミュレーションステップ', 10, 500, 100, 10)

if st.button('シミュレーション実行'):
    model, history = run_simulation(n_agents, osekkai_rate, needy_rate, support_rate, needy_transition_rate, steps)
    
    st.pyplot(model.visualize_network())
    st.pyplot(visualize_results(history))
    
    final_support_rate = history[-1][3]
    st.write(f"シミュレーション終了時の支援率: {final_support_rate:.2%}")