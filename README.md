import heapq
import networkx as nx
import matplotlib.pyplot as plt

def Oracle(graph, start, goal):
    priority_queue = [(0, start)]
    visited = set()
    parent_map = {}
    g_scores = {node: float('inf') for node in graph}
    g_scores[start] = 0

    while priority_queue:
        _, node = heapq.heappop(priority_queue)

        if node in visited:
            continue

        visited.add(node)

        if node == goal:
            break

        for neighbor, neighbor_cost in graph[node].items():
            if neighbor not in visited:
                tentative_g_score = g_scores[node] + neighbor_cost
                if tentative_g_score < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g_score
                    priority_queue.append((tentative_g_score + heuristic(neighbor, goal), neighbor))
                    parent_map[neighbor] = node

    if goal not in visited:
        return None, float('inf')
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = parent_map[node]
    path.append(start)
    path.reverse()

    return path, g_scores[goal]

def heuristic(node, goal):
    heuristic_values = {
        'A': 9, 'B': 7, 'C': 5, 'D': 4, 'E': 3, 'F': 6, 'G': 2, 'H': 4, 'I': 2, 'J': 0
    }
    return heuristic_values[node]

graph = {
    'A': {'B': 2, 'C': 4, 'D': 1},
    'B': {'C': 3, 'E': 2},
    'C': {'D': 2, 'E': 5},
    'D': {'E': 3, 'F': 6},
    'E': {'F': 2, 'G': 1},
    'F': {'G': 3, 'H': 4},
    'G': {'I': 2},
    'H': {'I': 3, 'J': 5},
    'I': {'J': 1},
    'J': {}
}

start_node = 'A'
goal_node = 'J'

shortest_path, shortest_cost = oracle(graph, start_node, goal_node)

if shortest_path:
    print(f"Shortest Path from {start_node} to {goal_node}: {shortest_path}")
    print(f"Total Cost: {shortest_cost}")

    G = nx.Graph()

    for node, neighbors in graph.items():
        G.add_node(node, heuristic=heuristic(node, goal_node))
        for neighbor, cost in neighbors.items():
            G.add_edge(node, neighbor, weight=cost)

    shortest_path_edges = [(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)]

    edge_labels = {(node, neighbor): cost for node, neighbor, cost in G.edges(data='weight')}

    fig, ax = plt.subplots()

    pos = nx.spring_layout(G)
    nx.draw(G, pos, ax=ax, with_labels=True, node_size=500, node_color='lightblue')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    nx.draw_networkx_edges(G, pos, edgelist=shortest_path_edges, edge_color='red', width=2)

    node_labels = {node: f"{node}\nHeuristic: {G.nodes[node]['heuristic']}" for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color='black')

    plt.show()
else:
    print(f"No path found from {start_node} to {goal_node}")
