import streamlit as st
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from packaging import version as Version
from PIL import Image
import time

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
            max_new_edges = min(10 - G.degree(i), 10)  # 既存のエッジ数を考慮して、新しく追加できるエッジの最大数を計算
            if max_new_edges <= 0:
                continue
            n_edges = np.random.randint(1, max_new_edges + 1)
            potential_neighbors = list(set(range(n_agents)) - set([i]) - set(G.neighbors(i)))
            potential_neighbors = [j for j in potential_neighbors if G.degree(j) < 10]  # 次数が10未満のノードのみを選択
            if len(potential_neighbors) < n_edges:
                n_edges = len(potential_neighbors)
            new_neighbors = np.random.choice(potential_neighbors, n_edges, replace=False)
            G.add_edges_from([(i, j) for j in new_neighbors])
        return G

    def step(self):
        for i, agent in enumerate(self.agents):
            if not agent.is_needy:
                if np.random.random() < self.needy_transition_rate:
                    agent.is_needy = True
                    agent.is_supported = False
                    agent.is_osekkai = False
            
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


def run_simulation(n_agents, osekkai_rate, needy_rate, support_rate, needy_transition_rate, steps):
    model = Model(n_agents, osekkai_rate, needy_rate, support_rate, needy_transition_rate)
    history = model.run(steps)
    return model, history

def visualize_rates_by_degree(model):
    degrees = dict(model.network.degree())
    osekkai_counts = {}
    needy_counts = {}
    supported_counts = {}
    total_counts = {}

    for node, degree in degrees.items():
        agent = model.agents[node]
        if degree not in total_counts:
            osekkai_counts[degree] = 0
            needy_counts[degree] = 0
            supported_counts[degree] = 0
            total_counts[degree] = 0
        
        if agent.is_osekkai:
            osekkai_counts[degree] += 1
        if agent.is_needy:
            needy_counts[degree] += 1
        if agent.is_supported:
            supported_counts[degree] += 1
        total_counts[degree] += 1

    degrees = list(total_counts.keys())
    osekkai_rates = [osekkai_counts[d] / total_counts[d] if total_counts[d] > 0 else 0 for d in degrees]
    needy_rates = [needy_counts[d] / total_counts[d] if total_counts[d] > 0 else 0 for d in degrees]
    supported_rates = [supported_counts[d] / total_counts[d] if total_counts[d] > 0 else 0 for d in degrees]
    normal_rates = [1 - (osekkai_rates[i] + needy_rates[i] + supported_rates[i]) for i in range(len(degrees))]

    # 色を統一
    colors = {'おせっ貝ちゃん': 'red', '困窮者': 'blue', '支援を受けている人': 'green', '通常': 'gray'}
    
    fig = go.Figure(data=[
        go.Bar(name='通常', x=degrees, y=normal_rates, marker_color=colors['通常']),
        go.Bar(name='おせっかい', x=degrees, y=osekkai_rates, marker_color=colors['おせっ貝ちゃん']),
        go.Bar(name='貧困', x=degrees, y=needy_rates, marker_color=colors['困窮者']),
        go.Bar(name='被支援', x=degrees, y=supported_rates, marker_color=colors['支援を受けている人'])
    ])

    fig.update_layout(
        title='ノード次数ごとの各状態の割合',
        xaxis_title='ノード次数',
        yaxis_title='割合',
        yaxis_tickformat='.0%',
        barmode='stack',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    return fig


def visualize_comprehensive_simulation(history, n_agents):
    osekkai, needy, supported, support_rates = zip(*history)
    steps = list(range(len(history)))
    
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=('エージェントの状態の推移', 'おせっ貝ちゃんと困窮者の割合'),
                        vertical_spacing=0.2)
    
    # 色を統一
    colors = {'おせっ貝ちゃん': 'red', '困窮者': 'blue', '支援を受けている人': 'green', '通常': 'gray'}
    
    # エージェントの状態の推移
    fig.add_trace(go.Scatter(x=steps, y=osekkai, mode='lines', name='おせっ貝ちゃん', line=dict(color=colors['おせっ貝ちゃん'], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=steps, y=needy, mode='lines', name='困窮者', line=dict(color=colors['困窮者'], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=steps, y=supported, mode='lines', name='支援を受けている人', line=dict(color=colors['支援を受けている人'], width=2)), row=1, col=1)
    
    # おせっ貝ちゃんと困窮者の割合
    osekkai_ratio = [o / n_agents for o in osekkai]
    needy_ratio = [n / n_agents for n in needy]
    fig.add_trace(go.Scatter(x=steps, y=osekkai_ratio, mode='lines', name='おせっ貝ちゃんの割合', line=dict(color=colors['おせっ貝ちゃん'], width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=steps, y=needy_ratio, mode='lines', name='困窮者の割合', line=dict(color=colors['困窮者'], width=2)), row=2, col=1)
    
    # レイアウトの設定
    fig.update_layout(
        height=800,
        title={
            'text': 'おせっ貝ちゃん増殖シミュレーション結果',
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        ),
        margin=dict(l=60, r=60, t=100, b=60)
    )
    
    fig.update_xaxes(title_text='ステップ')
    fig.update_yaxes(title_text='エージェント数', row=1, col=1)
    fig.update_yaxes(title_text='割合', row=2, col=1, tickformat='.1%')
    
    # サブプロットのタイトルのフォントサイズを調整
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=14)
    
    return fig



# アプリケーションのタイトルの前に画像を配置
col1, col2, col3 = st.columns([1,2,1])

with col1:
    st.write("")

with col2:
    image = Image.open('static/images/osekkai-chan.png')
    st.image(image, width=150)  # 幅を150ピクセルに設定

with col3:
    st.write("")

st.title('おせっ貝ちゃん増殖シミュレーション')
# GitHubリポジトリへのリンクを追加
st.markdown("[GitHub リポジトリ](https://github.com/mashimashica/osekkai-chan)")

st.markdown("""
このシミュレーションでは、社会におけるおせっかいさん（支援者）と困窮者の相互作用をモデル化しています。
各パラメータを調整して、どのように支援の広がり方が変化するかを観察できます。

- **エージェント数**: シミュレーション内の総人数
- **おせっかい率**: 初期状態でおせっかいさんである確率
- **困窮者率**: 初期状態で困窮者である確率
- **初期支援率**: 初期状態で支援を受けている困窮者の割合
- **困窮化率**: 毎ステップで新たに困窮者になる確率
- **シミュレーションステップ**: シミュレーションを実行する期間

パラメータを設定し、「シミュレーション実行」ボタンをクリックしてシミュレーションを開始してください。
""")

with st.expander("シミュレーションの詳細説明を開く"):
    st.markdown("""
    ### シミュレーションの詳細

    このシミュレーションは、社会におけるおせっかい（支援）行動の伝播と、その効果を模倣しています。

    #### エージェントの種類
    1. **おせっかいさん**：他者を支援する意欲のある個人
    2. **困窮者**：支援を必要としている個人
    3. **一般の人**：特別な状態にない個人

    #### シミュレーションの流れ
    1. 初期状態では、設定されたパラメータに基づいてエージェントが配置されます。
    2. 各ステップで以下のプロセスが実行されます：
       - 一般の人が一定確率で困窮者になる
       - おせっかいさんが隣接する困窮者を支援する
       - 支援を受けた困窮者が回復し、おせっかいさんになる可能性がある

    #### パラメータの意味
    - **エージェント数**：シミュレーション内の総人数。多いほど複雑な相互作用が観察できます。
    - **おせっかい率**：初期状態でおせっかいさんである確率。高いほど支援が広がりやすくなります。
    - **困窮者率**：初期状態で困窮者である確率。社会の初期状態の困難度を表します。
    - **初期支援率**：初期状態で支援を受けている困窮者の割合。初期の支援体制の充実度を表します。
    - **困窮化率**：毎ステップで新たに困窮者になる確率。社会の不安定さを表します。
    - **シミュレーションステップ**：シミュレーションを実行する期間。長いほど長期的な傾向が観察できます。

#### 結果の解釈
- **時系列グラフ**：各タイプのエージェント数の変化とおせっ貝ちゃんと困窮者の割合の推移を示しています。
- **ノード次数ごとの状態割合**：ネットワーク内のノード（エージェント）の次数（つながりの数）ごとに、各状態（通常、おせっかい、困窮者、被支援）の割合を示しています。
- **最終統計情報**：シミュレーション終了時点での各状態のエージェント数、割合、支援率、およびおせっ貝ちゃんの影響力を示しています。

    このシミュレーションを通じて、小さな支援行動が社会全体にどのように影響を与えるかを観察し、
    効果的な支援システムの構築に向けたヒントを得ることができます。
    """)


n_agents = st.slider('エージェント数', 100, 10000, 5000, 100)
osekkai_rate = st.slider('おせっかい率', 0.001, 0.1, 0.001, 0.001)
needy_rate = st.slider('困窮者率', 0.05, 0.3, 0.157, 0.01)
support_rate = st.slider('初期支援率', 0.1, 0.5, 0.2, 0.1)
needy_transition_rate = st.slider('困窮化率', 0.0001, 0.01, 0.001, 0.0001)
steps = st.slider('シミュレーションステップ', 10, 500, 100, 10)


# Streamlitアプリケーションの更新部分
if st.button('シミュレーション実行'):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(10):
        status_text.text(f"おせっ貝ちゃん増殖中{'.' * (i % 4)}")
        progress_bar.progress((i + 1) * 10)
        time.sleep(0.2)
    model, history = run_simulation(n_agents, osekkai_rate, needy_rate, support_rate, needy_transition_rate, steps)

    status_text.empty()
    progress_bar.empty()    
    # 新しい包括的可視化関数を使用
    st.plotly_chart(visualize_comprehensive_simulation(history, n_agents), use_container_width=True)
    
    # ネットワーク図は以前と同じ
    st.plotly_chart(visualize_rates_by_degree(model), use_container_width=True)
    
    # 最終的な統計情報を表示
    final_osekkai, final_needy, final_supported, final_support_rate = history[-1]
    st.write(f"シミュレーション終了時の統計:")
    st.write(f"- おせっ貝ちゃんの数: {final_osekkai} ({final_osekkai/n_agents:.2%})")
    st.write(f"- 困窮者の数: {final_needy} ({final_needy/n_agents:.2%})")
    st.write(f"- 支援を受けている人の数: {final_supported}")
    st.write(f"- 支援率: {final_support_rate:.2%}")
    st.write(f"- おせっ貝ちゃんの影響力: {final_supported/final_osekkai if final_osekkai > 0 else 0:.2f} 人/おせっ貝ちゃん")