import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

file_path = "./data/matrices/mat1.csv" 
adjacency_matrix = pd.read_csv(file_path, header=None).values

G = nx.Graph(adjacency_matrix)

pos = nx.spring_layout(G, seed=42)  # Set the layout algorithm with a fixed seed for reproducibility

node_size = 800
node_color = 'skyblue'
edge_color = '#888888'

nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color)

nx.draw_networkx_edges(G, pos, width=1.0, edge_color=edge_color)

nx.draw_networkx_labels(G, pos, font_size=10, font_color='black', font_weight='bold')


plt.axis('off')
plt.show()
