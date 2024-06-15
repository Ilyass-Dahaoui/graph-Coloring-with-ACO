import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt

class Ant:
    def __init__(self, alpha=1, beta=1):
        self.graph = None
        self.colors = {}
        self.start = None
        self.visited = []
        self.unvisited = []
        self.alpha = alpha
        self.beta = beta
        self.distance = 0
        self.number_colisions = 0
        self.colors_available = []
        self.colors_assigned = {}

    def initialize(self, g, colors, start=None):
        self.colors_available = sorted(colors.copy())

        keys = [n for n in g_nodes_int]
        self.colors_assigned = {key: None for key in keys}

        if start is None:
            self.start = random.choice(g_nodes_int)
        else:
            self.start = start

        self.visited = []
        self.unvisited = g_nodes_int.copy()

        if len(self.visited) == 0:
            self.assign_color(self.start, self.colors_available[0])
        return self

    def assign_color(self, node, color):
        self.colors_assigned[node] = color
        self.visited.append(node)
        self.unvisited.remove(node)

    def colorize(self):
        len_unvisited = len(self.unvisited)
        tabu_colors = []
        for i in range(len_unvisited):
            next_node = self.next_candidate()
            tabu_colors = []
            for j in range(number_nodes):
                if adj_matrix[next_node, j] == 1:
                    tabu_colors.append(self.colors_assigned[j])
            for k in self.colors_available:
                if k not in tabu_colors:
                    self.assign_color(next_node, k)
                    break
        self.distance = len(set(self.colors_assigned.values()))

    def dsat(self, node=None):
        if node is None:
            node = self.start
        col_neighbors = []
        for j in range(number_nodes):
            if adj_matrix[node, j] == 1:
                col_neighbors.append(self.colors_assigned[j])
        return len(set(col_neighbors))

    def si(self, node, adj_node):
        return phero_matrix[node, adj_node]

    def next_candidate(self):
        if len(self.unvisited) == 0:
            candidate = None
        elif len(self.unvisited) == 1:
            candidate = self.unvisited[0]
        else:
            max_value = 0
            heuristic_values = []
            candidates = []
            candidates_available = []
            for j in self.unvisited:
                heuristic_values.append((self.si(self.start, j) ** self.alpha) * (self.dsat(j) ** self.beta))
                candidates.append(j)
            max_value = max(heuristic_values)
            for i in range(len(candidates)):
                if heuristic_values[i] >= max_value:
                    candidates_available.append(candidates[i])
            candidate = random.choice(candidates_available)
        self.start = candidate
        return candidate

    def pheromone_trail(self):
        phero_trail = np.zeros((number_nodes, number_nodes), float)
        for i in g_nodes_int:
            for j in g_nodes_int:
                if self.colors_assigned[i] == self.colors_assigned[j]:
                    phero_trail[i, j] = 1
        return phero_trail

    def colisions(self):
        colisions = 0
        for key in self.colors_assigned:
            node = key
            col = self.colors_assigned[key]
            for j in range(number_nodes):
                if adj_matrix[node, j] == 1 and self.colors_assigned[j] == col:
                    colisions = colisions + 1
        return colisions


def create_graph(path):
    global number_nodes
    g = nx.Graph()
    f = open(path)
    n = int(f.readline())
    for i in range(n):
        graph_edge_list = f.readline().split()
        graph_edge_list[0] = int(graph_edge_list[0])
        graph_edge_list[1] = int(graph_edge_list[1])
        g.add_edge(graph_edge_list[0], graph_edge_list[1])
    return g


def draw_graph_with_colors(g, col_val):
    pos = nx.spring_layout(g)
    values = [col_val.get(node, 'blue') for node in g.nodes()]
    nx.draw(g, pos, with_labels=True, node_color=values, edge_color='black', width=1, alpha=0.7)


def init_colors(g):
    colors = []
    grundy = len(nx.degree_histogram(g))
    for c in range(grundy):
        colors.append(c)
    return colors


def init_pheromones(g):
    phero_matrix = np.ones((number_nodes, number_nodes), float)
    for node in g:
        for adj_node in g.neighbors(node):
            phero_matrix[node, adj_node] = 0
    return phero_matrix


def adjacency_matrix(g):
    adj_matrix = np.zeros((number_nodes, number_nodes), int)
    for node in g_nodes_int:
        for adj_node in g.neighbors(node):
            adj_matrix[node, adj_node] = 1
    return adj_matrix


def create_colony():
    ants = []
    ants.extend([Ant().initialize(g, colors) for _ in range(number_ants)])
    return ants


def apply_decay():
    for node in g_nodes_int:
        for adj_node in g_nodes_int:
            phero_matrix[node, adj_node] = phero_matrix[node, adj_node] * (1 - phero_decay)


def update_elite():
    global phero_matrix
    best_dist = 0
    elite_ant = None
    for ant in ants:
        if best_dist == 0:
            best_dist = ant.distance
            elite_ant = ant
        elif ant.distance < best_dist:
            best_dist = ant.distance
            elite_ant = ant
    elite_phero_matrix = elite_ant.pheromone_trail()
    phero_matrix = phero_matrix + elite_phero_matrix
    return elite_ant.distance, elite_ant.colors_assigned


def solve(input_graph, num_ants=10, iter=10, a=1, b=3, decay=0.8):
    global g
    global number_nodes
    global g_nodes_int
    global number_ants
    global alpha
    global beta
    global phero_decay
    global adj_matrix
    global phero_matrix
    global colors
    global ants

    g = input_graph
    number_ants = num_ants
    number_iterations = iter
    alpha = a
    beta = b
    phero_decay = decay

    final_solution = {}
    final_costs = 0
    iterations_needed = 0

    number_nodes = nx.number_of_nodes(g)
    g_nodes_int = [node for node in sorted(g.nodes)]
    adj_matrix = adjacency_matrix(g)
    colors = init_colors(g)
    phero_matrix = init_pheromones(g)

    for i in range(number_iterations):
        ants = create_colony()
        for ant in ants:
            ant.colorize()
        apply_decay()
        elite_dist, elite_sol = update_elite()
        if final_costs == 0:
            final_costs = elite_dist
            final_solution = elite_sol
            iterations_needed = i + 1
        elif elite_dist < final_costs:
            final_costs = elite_dist
            final_solution = elite_sol
            iterations_needed = i + 1

    return final_costs, final_solution, iterations_needed


G = nx.Graph()
G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 4), (0, 3), (1, 2), (1, 4), (3, 5), (5, 6)])

cost, solution, iterations = solve(G, num_ants=10, iter=100, a=1, b=3, decay=0.8)
print("Coût de la solution optimale:", cost)
print("Solution optimale (coloriage des nœuds):", solution)
print("Nombre d'itérations nécessaires:", iterations)

col_val = {}
for node, color in solution.items():
    col_val[node] = colors[color]

draw_graph_with_colors(G, col_val)
plt.title('Coloration optimale du graphe')
plt.show()
