import networkx as nx
import networkit as nk
import matplotlib.pyplot as plt
import collections
import numpy as np
from collections import defaultdict
import random
import seaborn as sns


class SIRModel:
    def __init__(self, G,
                num_of_initial_infected=5, rate_of_infection_spread=0.2,
                num_of_initial_removed=0, removal_chance=0.1,
                num_of_initial_strong_protected=5, rate_of_strong_protection_spread=0.05,
                num_of_initial_weak_protected=10, rate_of_weak_protection_spread=0.2,
                num_of_initial_weak_protected_but_infected=0,
                num_of_initial_strong_protected_but_infected=0,
                num_of_initial_immune=0, recovery_chance=0.01, immunity_chance=0.1,
                weak_protection_failure_chance=0.5, protection_removal_chance=0.1,
                protection_initialization_chance=0.01):

        self.G = G
        self.rate_of_infection_spread = rate_of_infection_spread
        self.removal_chance = removal_chance
        self.rate_of_strong_protection_spread = rate_of_strong_protection_spread
        self.rate_of_weak_protection_spread = rate_of_weak_protection_spread
        self.recovery_chance = recovery_chance
        self.immunity_chance = immunity_chance
        self.weak_protection_failure_chance = weak_protection_failure_chance
        self.protection_removal_chance = protection_removal_chance
        self.protection_initialization_chance = protection_initialization_chance

        # Initialize node states
        # All nodes are susceptible at the beginning
        self.states = {node: 0 for node in G.nodes()}

        # Set initial infected nodes
        if num_of_initial_infected > 0:
            initial_infected = random.choices(list(G.nodes()), k=num_of_initial_infected)
            for node in initial_infected:
                self.states[node] = 1

        # Set initial removed nodes
        if num_of_initial_removed > 0:
            initial_removed = random.choices(list(G.nodes()), k=num_of_initial_removed)
            for node in initial_removed:
                self.states[node] = 2

        # Set initial strong protected nodes
        if num_of_initial_strong_protected > 0:
            initial_strong_protected = random.choices(list(G.nodes()), k=num_of_initial_strong_protected)
            for node in initial_strong_protected:
                self.states[node] = 3

        # Set initial weak protected nodes
        if num_of_initial_weak_protected > 0:
            initial_weak_protected = random.choices(list(G.nodes()), k=num_of_initial_weak_protected)
            for node in initial_weak_protected:
                self.states[node] = 4

        # Set initial strong protected but infected nodes
        if num_of_initial_strong_protected_but_infected > 0:
            initial_strong_protected_but_infected = random.choices(list(G.nodes()), k=num_of_initial_strong_protected_but_infected)
            for node in initial_strong_protected_but_infected:
                self.states[node] = 5

        # Set initial weak protected but infected nodes
        if num_of_initial_weak_protected_but_infected > 0:
            initial_weak_protected_but_infected = random.choices(list(G.nodes()), k=num_of_initial_weak_protected_but_infected)
            for node in initial_weak_protected_but_infected:
                self.states[node] = 6

        # Set initial immune nodes
        if num_of_initial_immune > 0:
            initial_immune = random.choices(list(G.nodes()), k=num_of_initial_immune)
            for node in initial_immune:
                self.states[node] = 7

        # Store history for analysis
        self.history = []
        self.store_state()

    def store_state(self):
        """Store current state counts."""
        s = sum(1 for state in self.states.values() if state == 0)
        i = sum(1 for state in self.states.values() if state == 1)
        r = sum(1 for state in self.states.values() if state == 2)
        sp = sum(1 for state in self.states.values() if state == 3)
        wp = sum(1 for state in self.states.values() if state == 4)
        spi = sum(1 for state in self.states.values() if state == 5)
        wpi = sum(1 for state in self.states.values() if state == 6)
        im = sum(1 for state in self.states.values() if state == 7)
        self.history.append({'S': s, 'I': i, 'R': r, 'SP': sp, 'WP': wp, 'SPI': spi, 'WPI': wpi, 'IM': im})

    def step(self):
        """Simulate one time step."""
        new_susceptibles = []
        new_infections = []
        new_removals = []
        new_strong_protections = []
        new_weak_protections = []
        new_strong_protections_but_infected = []
        new_weak_protections_but_infected = []
        new_immunities = []

        # Going over the network, spread infection, protection, and check for removal
        for node in self.G.nodes():
            # If node is infected, try to infect each susceptible neighbors
            if self.states[node] == 1:
                for neighbor in self.G.neighbors(node):
                    if self.states[neighbor] == 0: # If neighbor is susceptible
                        if random.random() < self.rate_of_infection_spread: # Infection attempt
                            new_infections.append(neighbor)
                    if self.states[neighbor] == 4: # If neighbor is weak protected
                        if random.random() < self.weak_protection_failure_chance: # Protection break attempt
                            if random.random() < self.rate_of_infection_spread: # Infection attempt
                                new_infections.append(neighbor)

            # If node is strongly protected, try to spread the news of protection
            if (self.states[node] == 3) or (self.states[node] == 5):
                for neighbor in self.G.neighbors(node):
                    if random.random() < self.rate_of_strong_protection_spread: # protection attempt
                        if self.states[neighbor] == 0: # If neighbor is susceptible
                            new_strong_protections.append(neighbor)
                        elif self.states[neighbor] == 1: # If neighbor is infected
                            new_strong_protections_but_infected.append(neighbor)

            # If node is weakly protected, try to spread the news of protection
            if (self.states[node] == 4) or (self.states[node] == 6):
                for neighbor in self.G.neighbors(node):
                    if random.random() < self.rate_of_weak_protection_spread: # protection attempt
                        if self.states[neighbor] == 0: # If neighbor is susceptible
                            new_weak_protections.append(neighbor)
                        elif self.states[neighbor] == 1: # If neighbor is infected
                            new_weak_protections_but_infected.append(neighbor)

            # If protected, attempt to remove protection
            if (self.states[node] == 3) or (self.states[node] == 4):
                if random.random() < self.protection_removal_chance:
                    new_susceptibles.append(node)

            # If protected but infected, attempt to remove protection
            if (self.states[node] == 5) or (self.states[node] == 6):
                if random.random() < self.protection_removal_chance:
                    new_infections.append(node)

            # If susceptible, attempt to become strongly protected
            if self.states[node] == 0:
                if random.random() < self.protection_initialization_chance:
                    new_strong_protections.append(node)

            # If infected, attempt to become strongly protected
            if self.states[node] == 1:
                if random.random() < self.protection_initialization_chance:
                    new_strong_protections_but_infected.append(node)

            # If node is infected, attempt recovery (with chance of immunity) or removal
            if (self.states[node] == 1) or (self.states[node] == 5) or (self.states[node] == 6):
                if random.random() < self.recovery_chance:
                    if random.random() < self.immunity_chance:
                        new_immunities.append(node)
                    else:
                        new_susceptibles.append(node)
                elif random.random() < self.removal_chance:
                    new_removals.append(node)

        # Update states
        for node in new_susceptibles:
            self.states[node] = 0
        for node in new_infections:
            self.states[node] = 1
        for node in new_removals:
            self.states[node] = 2
        for node in new_strong_protections:
            self.states[node] = 3
        for node in new_weak_protections:
            self.states[node] = 4
        for node in new_strong_protections_but_infected:
            self.states[node] = 5
        for node in new_weak_protections_but_infected:
            self.states[node] = 6
        for node in new_immunities:
            self.states[node] = 7

        self.store_state()

        return len(new_susceptibles), len(new_infections), len(new_removals), \
                len(new_strong_protections), len(new_weak_protections), \
                len(new_strong_protections_but_infected), len(new_weak_protections_but_infected), \
                len(new_immunities)

    def run(self, max_steps=100):
        """
        Run simulation for specified number of steps.
        Returns: history - list of dicts (Number of S, I, R, WP, SP, WPI, SPI, IM nodes at each time step)
        """
        for _ in range(max_steps):
            new_susceptibles, new_infections, new_removals, new_strong_protections, new_weak_protections, \
            new_strong_protections_but_infected, new_weak_protections_but_infected, new_immunities = self.step()

        return self.history


def filterTSV(inputFile, outputFile):
    with open(inputFile, 'r') as f:
        with open(outputFile, 'w') as o:
            for line in f:
                line = line.strip()
                fields = line.split('\t')
                o.write('\t'.join(fields[1:3]) + '\n')


def exportGraph(G, outputFile):
    with open(outputFile, 'w') as f:
        for edge in G.edges():
            f.write('\t'.join(edge) + '\n')


def getLargestConnectedComponent(G):
    components = nx.connected_components(G)
    largestComponent = max(components, key=len)
    print('Nodes in giant component:', len(largestComponent))
    return G.subgraph(largestComponent)


def convert_nx_to_nk(nx_graph):
    nk_graph = nk.Graph(nx_graph.number_of_nodes())
    node_map = {node: idx for idx, node in enumerate(nx_graph.nodes())}
    reverse_map = {idx: node for node, idx in node_map.items()}
    for u, v in nx_graph.edges():
        nk_graph.addEdge(node_map[u], node_map[v])
    return nk_graph, node_map, reverse_map


def get_highest_centrality_nodes(G, centrality_type='degree', k=1):
    if isinstance(G, nx.Graph):
        nk_graph, node_map, reverse_map = convert_nx_to_nk(G)
    else:
        nk_graph = G
        reverse_map = {i: i for i in range(nk_graph.numberOfNodes())}
    if centrality_type == 'degree':
        centrality = nk.centrality.DegreeCentrality(nk_graph)
    elif centrality_type == 'closeness':
        centrality = nk.centrality.Closeness(nk_graph, True, False)
    elif centrality_type == 'betweenness':
        centrality = nk.centrality.Betweenness(nk_graph)
    else:
        raise ValueError("Unsupported centrality type")
    centrality.run()
    scores = centrality.scores()
    top_k = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
    return [(reverse_map[node], score) for node, score in top_k]


def getNodeStats(G):
    degrees = dict(G.degree())
    max_degree_node = max(degrees, key=degrees.get)
    max_degree = degrees[max_degree_node]
    print(f"Node with the largest degree: {max_degree_node} (Degree: {max_degree})")

    pageRank = dict(nx.pagerank(G))
    max_pageRank_node = max(pageRank, key=pageRank.get)
    max_pageRank = pageRank[max_pageRank_node]
    print(f"Node with the largest PageRank: {max_pageRank_node} (PageRank: {max_pageRank})")

    number_of_nodes_to_display = 5
    top_degree_nodes = get_highest_centrality_nodes(G, centrality_type='degree', k=number_of_nodes_to_display)
    top_closeness_nodes = get_highest_centrality_nodes(G, centrality_type='closeness', k=number_of_nodes_to_display)
    top_betweenness_nodes = get_highest_centrality_nodes(G, centrality_type='betweenness', k=number_of_nodes_to_display)
    print(f"Top degree centrality node(s): {top_degree_nodes[0:number_of_nodes_to_display]}")
    print(f"Top closeness centrality node(s): {top_closeness_nodes[0:number_of_nodes_to_display]}")
    print(f"Top betweenness centrality node(s): {top_betweenness_nodes[0:number_of_nodes_to_display]}")

    #pos = nx.spring_layout(G)
    #nx.draw(G, pos, with_labels=False, node_size=2, font_size=4, linewidths=0.1, width=0.1, edge_color='gray', node_color='black')
    #nx.draw_networkx_nodes(G, pos, nodelist=[node for node, _ in top_degree_nodes[0:number_of_nodes_to_display]], node_color='r', node_size=50)
    #nx.draw_networkx_nodes(G, pos, nodelist=[node for node, _ in top_closeness_nodes[0:number_of_nodes_to_display]], node_color='orange', node_size=50)
    #nx.draw_networkx_nodes(G, pos, nodelist=[node for node, _ in top_betweenness_nodes[0:number_of_nodes_to_display]], node_color='b', node_size=50)
    #plt.show()

    # scatterplot degree distribution of nodes in log-log scale
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    fig, ax = plt.subplots()
    plt.scatter(deg, cnt)
    plt.yscale('log')
    plt.xscale('log')
    plt.title("Degree distribution")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    plt.show()

    # scatterplot degree centrality distributuion of nodes in log-log scale
    degree_centrality = nx.degree_centrality(G)
    degree_centrality_sequence = sorted([d for n, d in degree_centrality.items()], reverse=True)
    degreeCentralityCount = collections.Counter(degree_centrality_sequence)
    deg, cnt = zip(*degreeCentralityCount.items())
    fig, ax = plt.subplots()
    plt.scatter(deg, cnt)
    plt.yscale('log')
    plt.xscale('log')
    plt.title("Degree centrality distribution")
    plt.ylabel("Count")
    plt.xlabel("Degree centrality")
    plt.show()


def detect_communities(G, method='plm'):
    nk_graph, node_map, reverse_map = convert_nx_to_nk(G)
    if method == 'plm':
        algo = nk.community.PLM(nk_graph)
    else: # method == 'plm+'
        algo = nk.community.PLM(nk_graph, refine=True)
    algo.run()
    partition = algo.getPartition()
    modularity = nk.community.Modularity().getQuality(partition, nk_graph)
    communities = {reverse_map[node]: partition.subsetOf(node) for node in range(G.number_of_nodes())}
    return communities, modularity


def analyze_communities(communities):
    comm_to_nodes = defaultdict(list)
    for node, comm in communities.items():
        comm_to_nodes[comm].append(node)
    print(f"Number of communities: {len(comm_to_nodes)}")
    sizes = [len(nodes) for nodes in comm_to_nodes.values()]
    print(f"Community sizes: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.1f}")
    comm_list = []
    for comm_id, nodes in sorted(comm_to_nodes.items()):
        # print(f"Community {comm_id}: {len(nodes)} nodes")
        comm_list.append(len(nodes))
    print(f"Community sizes in detail: {sorted(comm_list)}")


def visualize_communities(G_nx, communities):
    n_communities = len(set(communities.values()))
    colors = plt.cm.rainbow(np.linspace(0, 1, n_communities))
    color_map = {comm_id: colors[i]
                for i, comm_id in enumerate(sorted(set(communities.values())))}
    node_colors = [color_map[communities[node]] for node in G_nx.nodes()]
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G_nx)
    nx.draw_networkx_nodes(G_nx, pos, node_color=node_colors, node_size=10)
    nx.draw_networkx_edges(G_nx, pos, alpha=0.05)
    plt.title(f"Network Communities (n={n_communities})")
    plt.axis('off')
    plt.show()


def plot_sir_curves(history):
    """Plot the SIRSPWPSPIWPIIM curves over time."""
    s = [h['S'] for h in history]
    i = [h['I'] for h in history]
    r = [h['R'] for h in history]
    sp = [h['SP'] for h in history]
    wp = [h['WP'] for h in history]
    spi = [h['SPI'] for h in history]
    wpi = [h['WPI'] for h in history]
    im = [h['IM'] for h in history]

    plt.figure(figsize=(10, 6))
    plt.plot(s, 'b-', label='Susceptible')
    plt.plot(i, 'y-', label='Infected')
    plt.plot(r, 'r-', label='Removed')
    plt.plot(sp, 'g-', label='Strong Protected')
    plt.plot(wp, 'c-', label='Weak Protected')
    plt.plot(spi, 'm-', label='Strong Protected Infected')
    plt.plot(wpi, 'k-', label='Weak Protected Infected')
    plt.plot(im, 'orange', label='Immune')
    plt.title('SIR(SPWPSPIWPIIM) Model Spread')
    plt.xlabel('Time Step')
    plt.ylabel('Number of Nodes')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_heatmap(final_states_one, final_states_two, final_states_three):
    # ax = sns.heatmap(final_states_one, linewidth=0.5, vmin=0, vmax=6000, cmap='coolwarm')
    # TODO: 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    sns.heatmap(final_states_one, linewidth=0.5, vmin=0, vmax=6000, cmap='coolwarm', ax=axs[0])
    sns.heatmap(final_states_two, linewidth=0.5, vmin=0, vmax=6000, cmap='coolwarm', ax=axs[1])
    sns.heatmap(final_states_three, linewidth=0.5, vmin=0, vmax=6000, cmap='coolwarm', ax=axs[2])
    plt.show()
    # do percentile within boxes of how much of the network is dead
    # swap axis so that 0 0 is in 1 origin
    # recheck code

    #plt.title('Comparison of Protection Strategies')
    #plt.show()


def cluster_coefficient(G):
    cluster_coefficient = nx.clustering(G)
    cluster_coefficient = list(cluster_coefficient.values())
    cluster_coefficient = np.array(cluster_coefficient)
    cluster_coefficient = np.mean(cluster_coefficient)
    print(f"Average clustering coefficient: {cluster_coefficient:.3f}")


def main():
    # filterTSV('22140-0003-Data.tsv', '22140-0003-Data-Filtered.tsv')
    G = nx.read_edgelist('Largest-Component.tsv')
    # G = getLargestConnectedComponent(G)
    # getNodeStats(G)
    # showGraph(G)
    # exportGraph(G, 'Largest-Component.tsv')
    # cluster_coefficient(G)
    # communities, modularity = detect_communities(G, method='plm+')
    # print(f"Modularity: {modularity:.3f}")
    # analyze_communities(communities)
    # visualize_communities(G, communities)
    '''
    model = SIRModel(G, num_of_initial_infected=10, rate_of_infection_spread=0.1,
                    num_of_initial_removed=0, removal_chance=0.1,
                    num_of_initial_strong_protected=0, rate_of_strong_protection_spread=0,
                    num_of_initial_weak_protected=10, rate_of_weak_protection_spread=0.1,
                    num_of_initial_weak_protected_but_infected=0,
                    num_of_initial_strong_protected_but_infected=0,
                    num_of_initial_immune=0, recovery_chance=0, immunity_chance=0,
                    weak_protection_failure_chance=0, protection_removal_chance=0,
                    protection_initialization_chance=0)
    '''

    num_of_protection_failure_tests = 5
    num_of_protection_spread_tests = 5
    num_of_averaging_tests = 2
    self_protection_starting_chance = 0
    initial_infected_num = 10
    initial_protected_num = 10
    chance_of_being_removed = 0.2
    protection_spread_rate = 0.05
    steps = 200
    final_states_one = np.zeros((num_of_protection_failure_tests, num_of_protection_spread_tests, num_of_averaging_tests))
    final_states_two = np.zeros((num_of_protection_failure_tests, num_of_protection_spread_tests, num_of_averaging_tests))
    final_states_three = np.zeros((num_of_protection_failure_tests, num_of_protection_spread_tests, num_of_averaging_tests))
    averaged_final_states_one = np.zeros((num_of_protection_failure_tests, num_of_protection_spread_tests))
    averaged_final_states_two = np.zeros((num_of_protection_failure_tests, num_of_protection_spread_tests))
    averaged_final_states_three = np.zeros((num_of_protection_failure_tests, num_of_protection_spread_tests))

    protection_failure_chance = 0.9
    infection_spread = 0.1
    for i in range(num_of_protection_failure_tests):
        protection_spread = 0
        protection_failure_chance -= 0.05
        for j in range(num_of_protection_spread_tests):
            for k in range(num_of_averaging_tests):
                model = SIRModel(G, num_of_initial_infected=initial_infected_num, rate_of_infection_spread=infection_spread,
                                num_of_initial_removed=0, removal_chance=chance_of_being_removed,
                                num_of_initial_strong_protected=0, rate_of_strong_protection_spread=0,
                                num_of_initial_weak_protected=initial_protected_num, rate_of_weak_protection_spread=protection_spread,
                                num_of_initial_weak_protected_but_infected=0,
                                num_of_initial_strong_protected_but_infected=0,
                                num_of_initial_immune=0, recovery_chance=0, immunity_chance=0,
                                weak_protection_failure_chance=protection_failure_chance, protection_removal_chance=0,
                                protection_initialization_chance=self_protection_starting_chance)
                history = model.run(max_steps=steps)
                final_states_one[i,j,k]=history[-1]['R']
            protection_spread += protection_spread_rate

    protection_failure_chance = 0.9
    infection_spread = 0.2
    for i in range(num_of_protection_failure_tests):
        protection_spread = 0
        protection_failure_chance -= 0.05
        for j in range(num_of_protection_spread_tests):
            for k in range(num_of_averaging_tests):
                model = SIRModel(G, num_of_initial_infected=initial_infected_num, rate_of_infection_spread=infection_spread,
                                num_of_initial_removed=0, removal_chance=chance_of_being_removed,
                                num_of_initial_strong_protected=0, rate_of_strong_protection_spread=0,
                                num_of_initial_weak_protected=initial_protected_num, rate_of_weak_protection_spread=protection_spread,
                                num_of_initial_weak_protected_but_infected=0,
                                num_of_initial_strong_protected_but_infected=0,
                                num_of_initial_immune=0, recovery_chance=0, immunity_chance=0,
                                weak_protection_failure_chance=protection_failure_chance, protection_removal_chance=0,
                                protection_initialization_chance=self_protection_starting_chance)
                history = model.run(max_steps=steps)
                final_states_two[i,j,k]=history[-1]['R']
            protection_spread += protection_spread_rate

    protection_failure_chance = 0.9
    infection_spread = 0.3
    for i in range(num_of_protection_failure_tests):
        protection_spread = 0
        protection_failure_chance -= 0.05
        for j in range(num_of_protection_spread_tests):
            for k in range(num_of_averaging_tests):
                model = SIRModel(G, num_of_initial_infected=initial_infected_num, rate_of_infection_spread=infection_spread,
                                num_of_initial_removed=0, removal_chance=chance_of_being_removed,
                                num_of_initial_strong_protected=0, rate_of_strong_protection_spread=0,
                                num_of_initial_weak_protected=initial_protected_num, rate_of_weak_protection_spread=protection_spread,
                                num_of_initial_weak_protected_but_infected=0,
                                num_of_initial_strong_protected_but_infected=0,
                                num_of_initial_immune=0, recovery_chance=0, immunity_chance=0,
                                weak_protection_failure_chance=protection_failure_chance, protection_removal_chance=0,
                                protection_initialization_chance=self_protection_starting_chance)
                history = model.run(max_steps=steps)
                final_states_three[i,j,k]=history[-1]['R']
            protection_spread += protection_spread_rate

    for i in range(num_of_protection_failure_tests):
        for j in range(num_of_protection_spread_tests):
            averaged_final_states_one[i,j] = np.mean(final_states_one[i,j])
            averaged_final_states_two[i,j] = np.mean(final_states_two[i,j])
            averaged_final_states_three[i,j] = np.mean(final_states_three[i,j])

    plot_heatmap(averaged_final_states_one, averaged_final_states_two, averaged_final_states_three)
    # history = model.run(max_steps=100)
    # final_state = history[-1]
    # print("\nFinal Statistics:")
    # print(f"Susceptible: {final_state['S']}")
    # print(f"Infected: {final_state['I']}")
    # print(f"Removed: {final_state['R']}")
    # print(f"Strong Protected: {final_state['SP']}")
    # print(f"Weak Protected: {final_state['WP']}")
    # print(f"Strong Protected Infected: {final_state['SPI']}")
    # print(f"Weak Protected Infected: {final_state['WPI']}")
    # print(f"Immune: {final_state['IM']}")
    # print(f"Total Infected: {final_state['R'] + final_state['I']}")
    # print(f"Infection Rate: {(final_state['R'] + final_state['I']) / len(G):.2%}")
    # plot_sir_curves(history)


if __name__ == '__main__':
    main()


'''
Notes from Luca meeting:
Not recommended to consider targeting high centraility people to remove or ptotect.
So no targeted manipulation of the network
Triggering mechanism for safe sex like majority of friends?
Homophily and associativity of network based on gender and sexual relation but that would be new direction
We could also use this metadata for weights for the edges

Questions to Michelle:
- What should we present, and how to present
- Software recommendations to show spread of infection
- How much should we dig into getting parameters
- Which direction should we go
- Would it be recommended to dig into metadata or this is enough?
- Should we keep the initialized infected etc. nodes at random or target hubs/high centralities?
    Would that be a differenet direction?

Notes from Michelle meeting:
Use a heatmap comparing the word of mouth and effectiveness for the different strategies
Multipple heatmaps with different infectivities
cytoscape for attempt of visualization
keep scope to one low target for now


Todo: Check for powerlaw
Todo: Start making the powerpoint
Todo: Visualize the spread after every K steps
Todo: something for the fourth person


Questions to Luca:
    - We can see it is not power law but we can not explain it, help
    - What should we present that would be relevant?

Should have on slides
    - Gif of spread
    - Basic SIR presenetation
    - Heatmap after SIR
    - Communities in the network
    - Power law not there, show
    - Distributions of centralities
    - Show whole network and largest component to argue why we went with largest component
    -

Todo: check for whole network

Luca answers:
    We can study time in the whole network
    We can look at certain nodes or communities are less affected
    Not powerlaw because how the data was collected, which gives a very specific perspective
        When we say that networks follow a power law then we refer to a whole network not just
        a connected group of people within

    Degree distribution, centrality, clustering coefficient
    Show the models, and results, so SIR and Heatmap perhaps with time as well
        For the future, community structure, if different communities are inpacted in different ways
    Be cohesive
'''
