import json
import dgl
import torch
import numpy as np
from collections import defaultdict
import pickle

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def construct_graph(reviews, items):
    user_ids = set()
    item_ids = set()
    
    for review in reviews:
        user_ids.add(review['user_id'])
        item_ids.add(review['asin'])
    
    user_id_map = {user_id: i for i, user_id in enumerate(user_ids)}
    item_id_map = {item_id: i for i, item_id in enumerate(item_ids)}
    
    user_nodes = len(user_ids)
    item_nodes = len(item_ids)
    
    u_nodes = []
    i_nodes = []
    
    for review in reviews:
        u_nodes.append(user_id_map[review['user_id']])
        i_nodes.append(item_id_map[review['asin']])
    
    u_nodes = torch.tensor(u_nodes)
    i_nodes = torch.tensor(i_nodes)
    
    graph_data = {
        ('user', 'reviews', 'item'): (u_nodes, i_nodes),
        ('item', 'reviewed_by', 'user'): (i_nodes, u_nodes)
    }
    
    g = dgl.heterograph(graph_data, num_nodes_dict={'user': user_nodes, 'item': item_nodes})
    
    return g, user_id_map, item_id_map

def generate_labels(g, reviews, user_id_map, item_id_map, l_max=3, d_max=100, p_max=10):
    edge_labels = defaultdict(lambda: dict())
    path_labels = defaultdict(list)
    
    for review in reviews:
        user_id = review['user_id']
        item_id = review['asin']
        
        user_idx = user_id_map[user_id]
        item_idx = item_id_map[item_id]
        
        # Edge labels
        if (('user', user_idx), ('item', item_idx)) not in edge_labels:
            edge_labels[(('user', user_idx), ('item', item_idx))] = defaultdict(set)
        edge_labels[(('user', user_idx), ('item', item_idx))][('user', 'reviews', 'item')].add((user_idx, item_idx))
        
        # Path labels - For simplicity, consider direct review as path
        paths = [(('user', 'reviews', 'item'), user_idx, item_idx)]
        
        # Collect paths with constraints (example)
        # Note: In a real scenario, you need to define how to find these paths
        if len(paths) <= l_max and all(d <= d_max for d in [user_idx, item_idx]):
            path_labels[(('user', user_idx), ('item', item_idx))] = paths[:p_max]
    
    return edge_labels, path_labels

def main(review_file, item_file, edge_label_file, path_label_file):
    reviews = load_jsonl(review_file)
    items = load_jsonl(item_file)
    
    g, user_id_map, item_id_map = construct_graph(reviews, items)
    edge_labels, path_labels = generate_labels(g, reviews, user_id_map, item_id_map)
    
    with open(edge_label_file, 'wb') as f:
        pickle.dump(dict(edge_labels), f)
    
    with open(path_label_file, 'wb') as f:
        pickle.dump(dict(path_labels), f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate edge and path labels for Amazon Fashion dataset')
    parser.add_argument('--review_file', type=str, required=True, help='Path to the review JSONL file')
    parser.add_argument('--item_file', type=str, required=True, help='Path to the item JSONL file')
    parser.add_argument('--edge_label_file', type=str, required=True, help='Path to save the edge labels')
    parser.add_argument('--path_label_file', type=str, required=True, help='Path to save the path labels')
    
    args = parser.parse_args()
    
    main(args.review_file, args.item_file, args.edge_label_file, args.path_label_file)
