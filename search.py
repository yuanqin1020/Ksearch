import logging
import numpy as np
import argparse
from datetime import datetime

from datahandler import *
from processor import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", "-d", type=str, default="rel-heter",
                        choices=["rel-heter", "semi-heter", "semi-homo", "semi-rel", "geo-heter", "Unibench", "Drugbank"])
    parser.add_argument("--threshold", "-th", type=float, default=0.999)
    parser.add_argument("--topk", "-k", type=int, default=15)
    parser.add_argument("--method", "-m", type=str, default="grape", help="align, grape")
    parser.add_argument("--hops", "-hp", type=int, default=8)
    parser.add_argument("--words", "-w", type=int, default=3)
    parser.add_argument("--escal", "-es", type=float, default=1.0)
    parser.add_argument("--vscal", "-vs", type=float, default=1.0)

    args = parser.parse_args()
    print(args)

    return args


args = parse_args()


# Example
# keywords = ['chicago', 'american'] # mexican
keywords = queries[args.data_name][args.words]
print(f"keywords: {keywords}")

left_graph, left_vertex_index, right_graph, right_vertex_index = read_graphs(args.data_name)
print(f"left: {len(left_vertex_index)} vertex, {len(left_graph)} edge, right: {len(right_vertex_index)} vertex, {len(right_graph)} edge")

graph = left_graph[: int(len(left_graph) * args.vscal)] + right_graph[: int(len(right_graph) * args.vscal)]

vertex_index = left_vertex_index.copy()
for key, value in right_vertex_index.items():
    vertex_index[len(left_vertex_index) + key] = value

print(f"total graph triples: {len(graph)}")
print(f"total vertex number: {len(vertex_index)}")

if args.method != 'improve':
    attr_id_dict = {value: key for key, value in vertex_index.items()}
    print(f"attr inverse index complete: {len(attr_id_dict)}")

n = len(vertex_index)
dist_file = f"cache/dist_{args.data_name}_{args.method}.csv"

# improve = True if args.method == 'improve' else False
# prune = True if args.method == 'prune' else False

if args.method == 'base':
    begin = time.time()
    dist = shortest_dist(graph, n, attr_id_dict, dist_file, seeds=None)
    trees, cand = dp_min_steiner_tree(vertex_index, dist, keywords, args.threshold, args.topk) 

elif args.method == 'align':
    # seeds = read_seed(args.data_name)
    # seed_indices = [(attr_id_dict[i], attr_id_dict[j]) for i, j in seeds if i in attr_id_dict and j in attr_id_dict]
    # dist = shortest_dist(graph, n, attr_id_dict, dist_file, seed_indices)
    # begin = time.time()
    # trees, cand = dp_min_steiner_tree(vertex_index, dist, keywords, args.threshold, args.topk, align=True, seeds=seed_indices, prune=False)

    begin = time.time()
    dist_file = f"cache/dist_{args.data_name}"
    dist_file1 = dist_file + "_left.csv"
    dist_file2 = dist_file + "_right.csv"

    attr_id_dict1 = {value: key for key, value in left_vertex_index.items()}
    attr_id_dict2 = {value: key for key, value in right_vertex_index.items()}
    dist1 = partial_shortest_dist(left_graph, len(left_vertex_index), attr_id_dict1, dist_file1)
    dist2 = partial_shortest_dist(right_graph, len(right_vertex_index), attr_id_dict2, dist_file2)

    seeds = read_seed(args.data_name)
    seed_indices = [(attr_id_dict1[i], attr_id_dict2[j]) 
                    for i, j in seeds if i in attr_id_dict1 and j in attr_id_dict2]
    
    trees, cand = complete_dp_min_steiner_tree(left_vertex_index, right_vertex_index, dist1, dist2, keywords,
                                               args.threshold, args.topk, align=True, seeds=seed_indices)
                                          

elif args.method in ['improve', 'grape']:
    seeds = read_seed(args.data_name)
    if args.escal != 1.0:
        seeds = seeds[: int(len(seeds)*args.escal)]

    begin = time.time()
    subfile = f"cache/related_{args.data_name}"
    sub_graph1, key_nodes1 = extract_related_triplets(subfile + "_left", 
                                                      left_graph, keywords, args.threshold, args.hops, contain=False)
    sub_graph2, key_nodes2 = extract_related_triplets(subfile + "_right", 
                                                      right_graph, keywords, args.threshold, args.hops, contain=False)
    left_vertex_index = read_graph(sub_graph1)
    right_vertex_index = read_graph(sub_graph2)
    sub_graph = sub_graph1.union(sub_graph2)
    vertex_index = left_vertex_index.copy()
    for key, value in right_vertex_index.items():
        vertex_index[len(left_vertex_index) + key] = value

    print(f"left sub: {len(sub_graph1)}/{len(left_graph)} triples, extract {len(left_vertex_index)} vertex")
    print(f"right sub: {len(sub_graph2)}/{len(right_graph)} triples, extract {len(right_vertex_index)} vertex")
    print(f"related triples: {len(sub_graph)}/{len(graph)} triples, takes time {time.time() - begin}")
    
    key_nodes = key_nodes1.copy()
    for key, value in key_nodes2.items():
        key_nodes[key] = key_nodes[key].union(value)
    
    # print(f"key_nodes: {key_nodes}")
    # print(f"vertex_index: {vertex_index}")
    
    n = len(vertex_index)
    attr_id_dict = {value: key for key, value in vertex_index.items()}
    seed_indices = [(attr_id_dict[i], attr_id_dict[j]) for i, j in seeds if i in attr_id_dict and j in attr_id_dict]
    print(f"seed_indices before: {len(seed_indices)}")
    print(f"boundary vertex index: {len(left_vertex_index)}, seed_indices: {len(seed_indices)}")
    # print(f"attr_id_dict: {attr_id_dict}")

    key_nodes_id = {} # {keyword: (node, sim)}
    for (word, nodes) in key_nodes.items():
        if word not in key_nodes_id:
            key_nodes_id[word] = []
        key_nodes_id[word] = [(attr_id_dict[node[0]], node[1]) for node in nodes if node[0] in attr_id_dict]
        print(f"keyword ({word}) nodes: {len(key_nodes_id[word])}")
        
    begin = time.time()
    dist = shortest_dist(sub_graph, n, attr_id_dict, dist_file, seed_indices, improve=True)
    if args.method == 'improve':
        trees, cand = dp_min_steiner_tree(vertex_index, dist, keywords, args.threshold, args.topk, 
                                          align=True, seeds=seed_indices, prune=True, key_nodes=key_nodes_id)
    else:
        comm_file = f"output/comm_{args.data_name}_{args.hops}_{args.words}_{args.topk}_{args.escal}_{args.vscal}.txt"
        trees, cand = grape_steiner_tree(vertex_index, dist, keywords, args.threshold, args.topk, comm_file,
                                         align=True, seeds=seed_indices, prune=True, key_nodes=key_nodes_id)
    

print(f"online ({args.method}) search process takes time: {time.time() - begin} Sec")


cross = 0
print(f"=================={args.topk} results============")
with open(f"output/{args.data_name}_{args.method}_{args.topk}_{args.words}_{args.escal}_{args.vscal}.txt", "w") as f:
    for i, (tree, words, dist) in enumerate(trees):
        f.write(f"Tree {i+1}: {[{i: vertex_index[i]} for i in list(tree)]}, {[i for i in list(words)]}, dist: {dist}\n")
        print(f"Tree {i+1}: {[{i: vertex_index[i]} for i in list(tree)]}, {[i for i in list(words)]}, dist: {dist}\n")
        cross += 1 if check_is_cross(list(tree), len(left_vertex_index), args.data_name) else 0
    f.write(f"=========boundary: {len(left_vertex_index)}, cross rate {cross}/{len(trees)}")

print(f"{datetime.now()}, online ({args.method}) search has candidates: {cand}, cross rate: {cross}/{len(trees)}")
