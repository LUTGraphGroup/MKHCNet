import networkx as nx
from graphviz import Digraph
import pickle
import knowledge_graph.hierarchical_graph as tree
import pandas as pd
import csv


def generate_relation_graph(node_file, relation_file):
    """
    Build a directed NetworkX graph from ICD node file and CSV relation file.

    Parameters:
        node_file (str): Path to file containing ICD codes (e.g., 50_attribute.txt)
        relation_file (str): CSV file containing relations (e.g., head, tail, relation)

    Returns:
        G (nx.DiGraph): A directed graph with nodes and labeled edges
    """
    icd_list = []
    with open(node_file, 'r', encoding='utf-8') as node_f:
        for line in node_f:
            line = line.strip()
            array = line.split('\t')
            if len(array) < 1:
                continue
            icd_code = array[0]
            icd_list.append(icd_code)

    G = nx.DiGraph()
    G.add_nodes_from(icd_list)

    with open(relation_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            head_entity = row[0]
            tail_entity = row[2]
            relation = row[4]
            G.add_edge(head_entity, tail_entity, relation=relation)

    return G


def save_triplets_to_csv(graph, output_file):
    """
    Save graph edges as (head, relation, tail) triplets to a CSV file.

    Note:
        This function tries to access edge['weight'], but 'weight' is not set in generate_relation_graph.
        You may want to use 'relation' instead, or set 'weight' explicitly.

    Parameters:
        graph (nx.DiGraph): The graph to extract triplets from
        output_file (str): CSV file path for saving the triplets
    """
    triplets = []
    for edge in graph.edges(data=True):
        head_entity = edge[0]
        tail_entity = edge[1]
        # Changed from 'weight' to 'relation' to match edge creation
        relation = edge[2].get('relation', '')  # use .get() to avoid KeyError
        triplets.append((head_entity, relation, tail_entity))

    df = pd.DataFrame(triplets, columns=['Head_Entity', 'Relation', 'Tail_Entity'])
    df.to_csv(output_file, index=False)
    print("Triplets saved to", output_file)


def build_complete_graph(node_file, relation_file, complete_graph_file):
    """
    Combine relation and hierarchical subgraphs into a complete graph,
    and serialize it to a pickle (.pkl) file.

    Parameters:
        node_file (str): ICD node file path
        relation_file (str): CSV relation file path
        complete_graph_file (str): Path to save the final serialized graph
    """
    parient_children, level_0, level_1, level_2, level_3, adj, node2id, hier_dicts, hier_dicts_init, max_children_num = tree.build_tree(
        node_file)

    relation_G = generate_relation_graph(node_file, relation_file)
    print('Number of relation nodes:', relation_G.number_of_nodes())
    print('Number of relation edges:', relation_G.number_of_edges())

    hierarchical_G = tree.generate_hierarchical_graph(parient_children, node2id)
    hierarchical_G_sub = hierarchical_G.subgraph(relation_G.nodes)
    print('Number of hierarchical subgraph nodes:', hierarchical_G_sub.number_of_nodes())
    print('Number of hierarchical subgraph edges:', hierarchical_G_sub.number_of_edges())

    complete_G = nx.compose(relation_G, hierarchical_G_sub)
    print('Number of complete graph nodes:', complete_G.number_of_nodes())
    print('Number of complete graph edges:', complete_G.number_of_edges())

    serialize_graph(complete_graph_file, complete_G)


def draw_graph(G, graph_name):
    """
    Visualize the graph using Graphviz.

    Parameters:
        G (nx.Graph): The graph to visualize
        graph_name (str): Name for the Graphviz output
    """
    dot = Digraph(comment=graph_name)
    for node in G.nodes:
        dot.node(node)
    for edge in G.edges:
        dot.edge(edge[0], edge[1], label='')  # Label can be added if needed
    dot.render(graph_name, view=True)


def serialize_graph(file, G):
    """
    Serialize a NetworkX graph to disk using pickle.

    Parameters:
        file (str): Output .pkl file path
        G (nx.Graph): Graph to serialize

    Returns:
        True on success
    """
    with open(file, "wb") as f:
        pickle.dump(G, f)
    return True


def reload_graph(file):
    """
    Load a serialized graph from a pickle file.

    Parameters:
        file (str): Path to .pkl file

    Returns:
        G (nx.DiGraph): The loaded graph

    Raises:
        AssertionError if the loaded object is not a DiGraph
    """
    with open(file, "rb") as f:
        G = pickle.load(f)
        assert isinstance(G, nx.DiGraph), "Reloaded graph is not a DiGraph"
    return G


if __name__ == "__main__":
    node_file = "./"
    relation_file = "./"
    output_graph_file = "./"

    build_complete_graph(node_file, relation_file, output_graph_file)
    G = reload_graph(output_graph_file)
