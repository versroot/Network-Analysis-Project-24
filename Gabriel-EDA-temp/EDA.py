import networkx as nx
import matplotlib.pyplot as plt


def filterTSV(inputFile, outputFile):
    with open(inputFile, 'r') as f:
        with open(outputFile, 'w') as o:
            for line in f:
                line = line.strip()
                fields = line.split('\t')
                o.write('\t'.join(fields[1:3]) + '\n')


def createGraph(inputFile):
    G = nx.Graph()
    with open(inputFile, 'r') as f:
        for line in f:
            line = line.strip()
            fields = line.split('\t')
            if not G.has_node(fields[0]):
                G.add_node(fields[0])
            if not G.has_node(fields[1]):
                G.add_node(fields[1])
            if not G.has_edge(fields[0], fields[1]):
                G.add_edge(fields[0], fields[1])
            if not G.has_edge(fields[1], fields[0]):
                G.add_edge(fields[1], fields[0])
    return G


def getLargestConnectedComponent(G):
    components = nx.connected_components(G)
    largestComponent = max(components, key=len)
    print(len(largestComponent))
    return G.subgraph(largestComponent)


def exportGraph(G, outputFile):
    with open(outputFile, 'w') as f:
        for edge in G.edges():
            f.write('\t'.join(edge) + '\n')


def showGraph(G):
    # This thing will not run on my laptop...Too slow...
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=False)
    plt.show()


def main():
    # No clue why we are using .tsv files, but it is what it is...
    # filterTSV('22140-0003-Data.tsv', '22140-0003-Data-Filtered.tsv')
    G = createGraph('22140-0003-Data-Filtered.tsv')
    G = getLargestConnectedComponent(G)
    # showGraph(G)
    exportGraph(G, 'Largest-Component.tsv')


if __name__ == '__main__':
    main()
