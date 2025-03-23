import numpy as np
import matplotlib.pyplot as plt

def create_training_plot(scores: list, trial_num: str = '01'):
    """
    function to create a training plot and save it.

    Params:
    ======
    scores: scores from training run
    trial_num: string to better organize result files. Indicates the trial number.
    """
    # Compute rolling average (moving average) using numpy's convolv function
    window_size = 100
    rolling_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')

    # Plot scores and rolling average
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(scores)), scores, label="Scores", alpha=0.5)
    plt.plot(np.arange(window_size-1, len(scores)), rolling_avg, label="100-Episode Rolling Avg", color='orange', linewidth=2)
    plt.axhline(y=13, color='r', linestyle='--', linewidth=2, label="Target Score")
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('Training Progress')
    plt.legend()
    plt.savefig(f'results/training_scores/training_scores_trial_{trial_num}.png', bbox_inches='tight', transparent=True, facecolor='white')
    plt.show()

import networkx as nx

def create_visual_network_architecture_png(state_size: int= 37, fc1_units: int = 64, fc2_units: int = 64, action_size: int = 4):
    """
    function to create a plot of the utilized network architecture.

    Params:
    ======
    state_size (int): Dimension of each state
    action_size (int): Dimension of each action
    fc1_units (int): Number of nodes in first hidden layer
    fc2_units (int): Number of nodes in second hidden layer
    """

    layers = {
        "Input Layer": state_size,
        "Hidden Layer 1": fc1_units, 
        "Hidden Layer 2": fc2_units,
        "Output Layer": action_size  
    }

    # init graph
    G = nx.DiGraph()
    layer_positions = {}
    x_offset = 0

    for layer, num_nodes in layers.items():
        for i in range(min(num_nodes, 10)):
            node_name = f"{layer}_{i}"
            G.add_node(node_name, layer=layer)
            layer_positions[node_name] = (x_offset, -i)
        x_offset += 2

    # edges
    layer_names = list(layers.keys())
    for i in range(len(layer_names) - 1):
        current_layer = layer_names[i]
        next_layer = layer_names[i + 1]

        for j in range(min(layers[current_layer], 10)):
            for k in range(min(layers[next_layer], 10)):
                G.add_edge(f"{current_layer}_{j}", f"{next_layer}_{k}")

    # graph
    plt.figure(figsize=(15, 12)) 
    pos = {node: (x, y) for node, (x, y) in layer_positions.items()}
    nx.draw(G, pos, with_labels=False, node_size=500, node_color="blue", edge_color="black", arrows=False)

    # layer labels
    for layer, x in zip(layers.keys(), range(0, len(layers) * 2, 2)):
        plt.text(x, 1, layer, fontsize=12, ha="center", fontweight="bold")

    plt.title("Architecture of the QNetwork")
    plt.savefig("results/further_viz/model_architecture.png", bbox_inches="tight")
    plt.show()
