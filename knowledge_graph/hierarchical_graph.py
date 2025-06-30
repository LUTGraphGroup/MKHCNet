import numpy as np
import networkx as nx

"""
Find the matching ICD range for a given ICD code.
If it starts with 'E' or 'V', remove the prefix before checking the range.
If a matching range is found (e.g., '001-009'), record it as (parent, child) and return the parent.

Parameters:
    icd (str): ICD code
    level (list): List of range strings
    parient_child (list): Output list of (parent, child) pairs

Returns:
    str or None: The matched range/parent string, or False if not matched
"""
def findrange(icd, level, parient_child):

    for item in level:
        if '-' in item:
            tokens = item.split('-')
            if icd.startswith('E') or icd.startswith('V'):
                if int(icd[1:]) in range(int(tokens[0][1:]), int(tokens[1][1:]) + 1):
                    parient_child.append((item, icd))
                    return item
            else:
                if int(icd) in range(int(tokens[0]), int(tokens[1]) + 1):
                    parient_child.append((item, icd))
                    return item
        else:
            if icd.startswith('E') or icd.startswith('V'):
                if int(icd[1:]) == int(item[1:]):
                    return False
            else:
                if int(icd) == int(item):
                    return False

    """
    Construct a hierarchical ICD tree from a flat ICD code list file.

    Returns:
        parient_child (list): All edges (parent ID, child ID)
        level0/1/2/3 (list): Nodes at each level of hierarchy
        adj (np.array): Adjacency matrix
        node2id (dict): Mapping from node names to integer IDs
        hier_dicts (dict): ICD-ID to path of parent node IDs
        hier_labels_init_new (dict): Simplified path of parent + self ID
        max_children_num (int): Maximum number of children across all nodes
    """
def build_tree(filepath):

    # Level 2 categories for ICDs (including E and V codes)
    level2 = [...]
    level2_E = [...]
    level2_V = [...]

    # Load all ICD codes from file
    allICDS = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for row in f:
            icd = row.split('\t')[0]
            allICDS.append(icd)

    allICDS_ = list(set(allICDS))
    allICDS_.sort(key=allICDS.index)
    print('Total unique ICD codes:', len(allICDS_))

    parient_child = []
    hier_icds = {}
    print('Building hierarchy for each ICD code...')

    for icd in allICDS_:
        hier_icd = [icd]
        if icd.startswith('E'):
            ...
        elif icd.startswith('V'):
            ...
        else:
            ...
        if icd not in hier_icds:
            hier_icds[icd] = hier_icd

    # Assign IDs to all nodes
    node2id = {}
    hier_labels_init = hier_icds.copy()
    for icd, hier_icd in hier_icds.items():
        if len(hier_icd) < 5:
            hier_icd += [hier_icd[-1]] * (5 - len(hier_icd))
        hier_icds[icd] = hier_icd
        for item in hier_icd:
            if item not in node2id:
                node2id[item] = len(node2id)

    hier_labels_init_new = {
        node2id[icd]: [node2id[item] for item in path]
        for icd, path in hier_labels_init.items()
    }

    node2id['ROOT'] = len(node2id)

    # Track nodes by hierarchy level
    level0, level1, level2_, level3 = set(), set(), set(), set()
    parient_child = []
    adj = np.zeros((len(node2id), len(node2id)))
    hier_dicts = {}

    print('Constructing adjacency matrix and edge list...')
    for icd, path in hier_icds.items():
        icdId = node2id[icd]
        path_ids = [node2id[item] for item in path]
        path_ids.insert(0, node2id['ROOT'])
        hier_dicts[icdId] = path_ids

        level0.add(path_ids[1])
        level1.add(path_ids[2])
        level2_.add(path_ids[3])
        level3.add(path_ids[4])

        for i in range(len(path_ids) - 1):
            adj[path_ids[i], path_ids[i + 1]] = 1
            parient_child.append([path_ids[i], path_ids[i + 1]])

    children_num = [len(np.argwhere(row)) for row in adj]
    max_children_num = max(len(level0), max(children_num))

    return (
        parient_child,
        list(level0), list(level1), list(level2_), list(level3),
        adj, node2id, hier_dicts, hier_labels_init_new,
        max_children_num
    )


def get_key_by_value(dictionary, target_value):
    """
    Helper function to find a key by its value.

    Parameters:
        dictionary (dict): A dictionary to search
        target_value: Value to look for

    Returns:
        The corresponding key, or None if not found
    """
    for key, value in dictionary.items():
        if value == target_value:
            return key
    return None


def generate_hierarchical_graph(parent_child, node2id):
    """
    Convert a list of (parent_id, child_id) edges to a NetworkX DiGraph
    using original node names via reverse lookup.

    Parameters:
        parent_child (list): List of (parent ID, child ID)
        node2id (dict): Mapping from node name to ID

    Returns:
        G (nx.DiGraph): A directed graph representing the hierarchy
    """
    G = nx.DiGraph()
    G.add_nodes_from(node2id.keys())
    edges_named = []

    for p_id, c_id in parent_child:
        parent_node = get_key_by_value(node2id, p_id)
        child_node = get_key_by_value(node2id, c_id)
        if parent_node and child_node:
            edges_named.append([parent_node, child_node])

    G.add_edges_from(edges_named)
    return G
