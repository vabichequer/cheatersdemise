import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()

# Nodes

G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_node(4)

# Edges

G.add_edge(1,2)

pos = pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args="-Gnodesep=5")

# Draw

nx.draw(G, pos, with_labels = True)

labels = {}
labels[1, 2] = '12'

nx.draw_networkx_edge_labels(G, pos, labels, font_color='red')

# Plot
plt.axis('off')

plt.show()

