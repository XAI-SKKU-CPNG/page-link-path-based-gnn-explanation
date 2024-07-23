import json
import dgl
import scipy.sparse as sp
import torch
import numpy as np
from utils import eids_split, remove_all_edges_of_etype, get_num_nodes_dict
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
    category_ids = set()
    brand_ids = set()
    user_item_edges = []
    item_category_edges = []
    item_brand_edges = []
    item_item_edges = []

    for review in reviews:
        user_id = review['user_id']
        item_id = review['asin']
        user_ids.add(user_id)
        item_ids.add(item_id)
        user_item_edges.append((user_id, item_id))

    for item in items:
        item_id = item['parent_asin']
        category_id = item.get('main_category', None)
        brand_id = item['details'].get('Brand', None) if 'details' in item else None
        item_ids.add(item_id)
        if category_id:
            category_ids.add(category_id)
            item_category_edges.append((item_id, category_id))
        if brand_id:
            brand_ids.add(brand_id)
            item_brand_edges.append((item_id, brand_id))
        if 'bought_together' in item and item['bought_together']:
            for bought_item in item['bought_together']:
                item_item_edges.append((item_id, bought_item))

    user_id_map = {user_id: i for i, user_id in enumerate(user_ids)}
    item_id_map = {item_id: i for i, item_id in enumerate(item_ids)}
    category_id_map = {category_id: i for i, category_id in enumerate(category_ids)}
    brand_id_map = {brand_id: i for i, brand_id in enumerate(brand_ids)}

    user_item_edges = [(user_id_map[u], item_id_map[i]) for u, i in user_item_edges]
    item_category_edges = [(item_id_map[i], category_id_map[c]) for i, c in item_category_edges]
    item_brand_edges = [(item_id_map[i], brand_id_map[b]) for i, b in item_brand_edges]
    item_item_edges = [(item_id_map[i1], item_id_map[i2]) for i1, i2 in item_item_edges]

    graph_data = {
        ('user', 'reviews', 'item'): user_item_edges,
        ('item', 'belongs_to', 'category'): item_category_edges,
        ('item', 'belongs_to', 'brand'): item_brand_edges,
        ('item', 'bought_together', 'item'): item_item_edges
    }

    num_nodes_dict = {
        'user': len(user_ids),
        'item': len(item_ids),
        'category': len(category_ids),
        'brand': len(brand_ids)
    }

    g = dgl.heterograph(graph_data, num_nodes_dict)

    return g, user_id_map, item_id_map

def process_data(g, val_ratio, test_ratio, src_ntype='user', tgt_ntype='item', pred_etype='reviews', neg='src_tgt_neg'):
    u, v = g.edges(etype=pred_etype)
    src_N = g.num_nodes(src_ntype)
    tgt_N = g.num_nodes(tgt_ntype)

    M = u.shape[0] # number of directed edges
    eids = torch.arange(M)
    train_pos_eids, val_pos_eids, test_pos_eids = eids_split(eids, val_ratio, test_ratio)

    train_pos_u, train_pos_v = u[train_pos_eids], v[train_pos_eids]
    val_pos_u, val_pos_v = u[val_pos_eids], v[val_pos_eids]
    test_pos_u, test_pos_v = u[test_pos_eids], v[test_pos_eids]

    # Collect negative samples
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), shape=(src_N, tgt_N))
    adj_neg = 1 - adj.todense()
    neg_u, neg_v = np.where(adj_neg != 0)
    neg_eids = np.random.choice(neg_u.shape[0], min(neg_u.shape[0], M), replace=False)
    train_neg_eids, val_neg_eids, test_neg_eids = eids_split(torch.from_numpy(neg_eids), val_ratio, test_ratio)
    
    train_neg_u, train_neg_v = np.take(neg_u, train_neg_eids), np.take(neg_v, train_neg_eids)
    val_neg_u, val_neg_v = np.take(neg_u, val_neg_eids), np.take(neg_v, val_neg_eids)
    test_neg_u, test_neg_v = np.take(neg_u, test_neg_eids), np.take(neg_v, test_neg_eids)
    
    # Construct graphs
    pred_can_etype = (src_ntype, pred_etype, tgt_ntype)
    num_nodes_dict = get_num_nodes_dict(g)
    
    train_pos_g = dgl.heterograph({pred_can_etype: (train_pos_u, train_pos_v)}, num_nodes_dict)
    train_neg_g = dgl.heterograph({pred_can_etype: (train_neg_u, train_neg_v)}, num_nodes_dict)
    val_pos_g = dgl.heterograph({pred_can_etype: (val_pos_u, val_pos_v)}, num_nodes_dict)
    val_neg_g = dgl.heterograph({pred_can_etype: (val_neg_u, val_neg_v)}, num_nodes_dict)
    test_pos_g = dgl.heterograph({pred_can_etype: (test_pos_u, test_pos_v)}, num_nodes_dict)
    test_neg_g = dgl.heterograph({pred_can_etype: (test_neg_u, test_neg_v)}, num_nodes_dict)
    
    mp_g = remove_all_edges_of_etype(g, pred_etype) # Remove pred_etype edges but keep nodes
    return mp_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g


def load_amazon_fashion_dataset(review_file, item_file, edge_label_file, path_label_file, val_ratio, test_ratio):
    reviews = load_jsonl(review_file)
    items = load_jsonl(item_file)
    
    g, user_id_map, item_id_map = construct_graph(reviews, items)
    
    with open(edge_label_file, 'rb') as f:
        pred_pair_to_edge_labels = pickle.load(f)
    
    with open(path_label_file, 'rb') as f:
        pred_pair_to_path_labels = pickle.load(f)
    
    src_ntype, tgt_ntype = 'user', 'item'
    pred_etype = 'reviews'
    neg = 'src_tgt_neg'
    
    processed_g = process_data(g, val_ratio, test_ratio, src_ntype, tgt_ntype, pred_etype, neg)
    
    return g, processed_g, pred_pair_to_edge_labels, pred_pair_to_path_labels
