import sys
import textwrap


class Node:
    def __init__(self, k_1_mer):
        self.k_1_mer = k_1_mer
        self.indegree = 0
        self.outdegree = 0


def read_kmers(fPath):
    print('Reading kmers...')
    with open(fPath, encoding='utf-8') as f:
        kmers = f.readlines()
    kmers = [s.strip() for s in kmers]
    return kmers


def construct_de_Bruijn_graph(kmers):
    print('Constructing de Bruijn graph...')
    k = len(kmers[0])
    nodes = dict()
    edges = dict()
    for kmer in kmers:
        for i in range(0, len(kmer) - (k - 1)):
            left_node = kmer[i:i + k - 1]
            right_node = kmer[i + 1:i + k]
            if nodes.get(left_node):
                nodes[left_node].outdegree += 1
                edges[left_node] += [right_node]
            elif not nodes.get(left_node):
                nodes[left_node] = Node(left_node)
                nodes[left_node].outdegree += 1
                edges[left_node] = [right_node]
            if nodes.get(right_node):
                nodes[right_node].indegree += 1
            elif not nodes.get(right_node):
                nodes[right_node] = Node(right_node)
                nodes[right_node].indegree += 1
                edges[right_node] = []
    return nodes, edges


def find_contigs(nodes, edges):
    print('Finding contigs from kmers...')
    paths = []
    for k_1_mer, node in nodes.items():
        if node.indegree != 1 or node.outdegree != 1:
            if node.outdegree > 0:
                for outgoing_node_k_1_mer in edges[node.k_1_mer]:
                    non_branching_path = [node.k_1_mer, outgoing_node_k_1_mer]
                    outgoing_node = nodes.get(outgoing_node_k_1_mer)
                    while outgoing_node.indegree == 1 and outgoing_node.outdegree == 1:
                        outgoing_node_k_1_mer = edges[outgoing_node.k_1_mer][0]
                        non_branching_path += [outgoing_node_k_1_mer]
                        outgoing_node = nodes.get(outgoing_node_k_1_mer)
                    paths.append(non_branching_path)

    visited = list(set([item for sublist in paths for item in sublist]))
    for k_1_mer, node in nodes.items():
        if not node.k_1_mer in visited:
            if node.indegree == 1 and node.outdegree == 1:
                cycle = [node.k_1_mer]
                if edges.get(node.k_1_mer):
                    next_node = edges[node.k_1_mer][0]
                    visited.append(node.k_1_mer)
                    while next_node.indegree == 1 and next_node.outdegree == 1:
                        cycle.append(next_node.k_1_mer)
                        if not next_node.k_1_mer in visited:
                            visited.append(next_node.k_1_mer)
                            next_node = edges[next_node][0]
                        else:
                            paths.append(cycle)
                            break
    return paths


def print_configs(paths):
    contigs = []
    for path in paths:
        contig = path[0]
        for k_1_mer in path[1:]:
            contig += k_1_mer[1:]
        contigs.append(textwrap.fill(contig, width=80))

    # print('Contigs:')
    # for contig in contigs:
    #     print(contig)

    print('Contigs with Occurences:')
    unq_contigs = dict((x, contigs.count(x)) for x in set(contigs))
    for unq_contig, count in unq_contigs.items():
        print('-' * 80)
        print(unq_contig + ' : ' + str(count))


def main(fPath):
    kmers = read_kmers(fPath)
    nodes, edges = construct_de_Bruijn_graph(kmers)
    paths = find_contigs(nodes, edges)
    print_configs(paths)


if __name__ == '__main__':
    main(sys.argv[1])
