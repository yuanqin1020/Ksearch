import logging
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import numpy as np
import heapq
import itertools
import time
import os

from datahandler import *


cache_dir = "/bert-base-uncased/"
print(cache_dir)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache_dir, local_files_only=True)
model = AutoModel.from_pretrained('bert-base-uncased', cache_dir=cache_dir, local_files_only=True)

def get_bert_embedding(text, model, tokenizer):
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids)
    embeddings = outputs[0][:, 0, :]

    return embeddings / embeddings.norm(dim=-1, keepdim=True)


def calculate_similarity(vec1, vec2):
    vec1 = vec1.reshape(-1, 1)
    vec2 = vec2.reshape(-1, 1)
    sim = np.dot(vec1.T, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    return round(sim.item(), 3)


# 计算所有节点对之间的最短路径长度
def shortest_dist(graph, n, attr_id_dict, dist_file, seeds=None, improve=False):
    begin = time.time()

    if os.path.exists(dist_file):
        dist = np.loadtxt(dist_file, delimiter=',')
        print(f"load dist from {dist_file}, takes time {time.time() - begin} Sec")

    else:
        dist = np.full((n, n), np.inf)
        np.fill_diagonal(dist, 0)
        
        if seeds is not None:
            for (i, j) in seeds:
                dist[i][j] = 0.1
                dist[j][i] = 0.1

        for triple in graph:
            a, b, c = triple
            start = attr_id_dict[a]
            end = attr_id_dict[c]
            dist[start][end] = 1
            # dist[end][start] = 1

        print("Dijkstra...")
        has_negative_cycle = False
        for i in range(n):
            pq = []
            heapq.heappush(pq, (0, i))

            while pq:
                cur_dist, node = heapq.heappop(pq)

                if cur_dist > dist[i][node]:
                    continue

                for j in range(n):
                    if dist[i][j] > dist[i][node] + dist[node][j]:
                        dist[i][j] = dist[i][node] + dist[node][j]
                        heapq.heappush(pq, (dist[i][j], j))

                    if i == j and dist[i][j] < 0:
                        has_negative_cycle = True
                        break

                if has_negative_cycle:
                    print("Detected negative weight cycle. Stopping distance calculation.")
                    break

        # floyd_warshall
        # print("Floyd-Warshall...")
        # for k in range(n):
        #     for i in range(n):
        #         for j in range(n):
        #             dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        
        if improve is not True:
            np.savetxt(dist_file, dist, delimiter=',')  # Write dist to a CSV file
        
        print(f"distance computation and write takes time: {time.time() - begin} Sec")

    return dist


def partial_shortest_dist(graph, n, attr_id_dict, dist_file):
    begin = time.time()

    if os.path.exists(dist_file):
        dist = np.loadtxt(dist_file, delimiter=',')
        print(f"load dist from {dist_file}, takes time {time.time() - begin} Sec")

    else:
        dist = np.full((n, n), np.inf)
        np.fill_diagonal(dist, 0)

        for triple in graph:
            a, b, c = triple
            start = attr_id_dict[a]
            end = attr_id_dict[c]
            dist[start][end] = 1
            # dist[end][start] = 1

        print("Dijkstra...")
        has_negative_cycle = False
        for i in range(n):
            pq = []
            heapq.heappush(pq, (0, i))

            while pq:
                cur_dist, node = heapq.heappop(pq)

                if cur_dist > dist[i][node]:
                    continue

                for j in range(n):
                    if dist[i][j] > dist[i][node] + dist[node][j]:
                        dist[i][j] = dist[i][node] + dist[node][j]
                        heapq.heappush(pq, (dist[i][j], j))

                    if i == j and dist[i][j] < 0:
                        has_negative_cycle = True
                        break

                if has_negative_cycle:
                    print("Detected negative weight cycle. Stopping distance calculation.")
                    break

        np.savetxt(dist_file, dist, delimiter=',')  # Write dist to a CSV file
        
        print(f"distance computation and write takes time: {time.time() - begin} Sec")

    return dist



import time
import os
import numpy as np
import heapq
from scipy.sparse import lil_matrix

def lil_shortest_dist(graph, n, attr_id_dict, dist_file, seeds):
    begin = time.time()

    if os.path.exists(dist_file):
        dist = np.loadtxt(dist_file, delimiter=',')
        print(f"load dist from {dist_file}, takes time {time.time() - begin} Sec")

    else:
        dist = lil_matrix((n, n))
        
        if seeds is not None:
            for (i, j) in seeds:
                dist[i, j] = 0.1
                dist[j, i] = 0.1

        for triple in graph:
            a, b, c = triple
            start = attr_id_dict[a]
            end = attr_id_dict[c]
            dist[start, end] = 1

        print("SPFA algorithm...")
        for i in range(n):
            pq = []
            in_queue = [False] * n
            in_queue[i] = True
            distance = np.full(n, np.inf)
            distance[i] = 0
            heapq.heappush(pq, (0, i))

            while pq:
                cur_dist, node = heapq.heappop(pq)
                in_queue[node] = False

                for j in range(n):
                    if dist[i, j] > dist[i, node] + dist[node, j]:
                        dist[i, j] = dist[i, node] + dist[node, j]
                        if not in_queue[j]:
                            in_queue[j] = True
                            heapq.heappush(pq, (dist[i, j], j))

        np.savetxt(dist_file, dist.toarray(), delimiter=',')  # Write dist to a CSV file
        
        print(f"distance computation and write takes time: {time.time() - begin} Sec")

    return dist



import scipy.sparse as sp
def sp_shortest_dist(graph, n, attr_id_dict, dist_file, seeds):
    begin = time.time()

    if sp.issparse(dist_file):
        dist = sp.load_npz(dist_file)
        print(f"Loaded sparse distance matrix from {dist_file}")
    else:
        dist = sp.lil_matrix((n, n))
        dist.setdiag(0)
        
        if seeds is not None:
            for (i, j) in seeds:
                dist[i, j] = 0.1
                dist[j, i] = 0.1

        for triple in graph:
            a, b, c = triple
            start = attr_id_dict[a]
            end = attr_id_dict[c]
            dist[start, end] = 1

        print("Dijkstra...")
        for i in range(n):
            pq = []
            heapq.heappush(pq, (0, i))

            while pq:
                cur_dist, node = heapq.heappop(pq)

                if cur_dist > dist[i, node]:
                    continue

                for j in range(n):
                    if dist[i, j] > dist[i, node] + dist[node, j]:
                        dist[i, j] = dist[i, node] + dist[node, j]
                        heapq.heappush(pq, (dist[i, j], j))

        sp.save_npz(dist_file, dist)  # Save the sparse matrix to a file

        print(f"distance computation and write takes time: {time.time() - begin} Sec")

    return dist


def calculate_tree_distance(tree, dist):
    tree_dist = 0
    # print(f"cal tree: {tree}")
    nodes = list(tree)
    for i in range(len(nodes) - 1):
        for j in range(i + 1, len(nodes)):
            dij = dist[nodes[i]][nodes[j]] if dist[nodes[i]][nodes[j]] != np.inf else 0
            dji = dist[nodes[j]][nodes[i]] if dist[nodes[j]][nodes[i]] != np.inf else 0
            tree_dist += max(dij, dji)  # Calculate the total distance of the tree (backward search)
    return tree_dist


def cal_complete_tree_distance(tree, dist, n, seed_dist):
    tree_dist = 0
    # print(f"cal tree: {tree}")
    nodes = list(tree)
    for i in range(len(nodes) - 1):
        for j in range(i + 1, len(nodes)):
            for k in range(n):
                dik = dist[nodes[i]][nodes[k]] if nodes[i] < n else np.inf
                dkj = seed_dist[k][n-j] if j > n and k in seed_dist.keys() else np.inf
                djk = dist[nodes[j]][nodes[k]] if nodes[j] < n else np.inf
                dki = seed_dist[k][n-i] if i > n and k in seed_dist.keys() else np.inf
                dij = min(dij, dik + dkj)
                dji = min(dji, djk + dki)

            dij = dij if dij != np.inf else 0
            dji = dji if dji else 0
            tree_dist += max(dij, dji)  # Calculate the total distance of the tree (backward search)
            
    return tree_dist


def tree_exists_in_queue(pq, tree, tree_keywords, root):
    for item in pq:
        if item[1] == tree and item[2] == tree_keywords and item[3] == root:
            return True
    return False


def update_or_push_tree(pq, tree_dist, tree, tree_keywords, root, cand, mark=None):
    for i, item in enumerate(pq):
        if item[1] == tree and item[2] == tree_keywords and item[3] == root:
            if item[0] <= tree_dist:
                return  # Do not push the current tree if existing tree's distance is smaller or equal
            else:
                pq.pop(i)  # Remove the existing tree with smaller distance
                cand -= 1
                heapq.heapify(pq)  # Re-heapify the queue after removing the element
                break

    if mark != None:
        heapq.heappush(pq, (tree_dist, tree, tree_keywords, root, mark))
    else:
        heapq.heappush(pq, (tree_dist, tree, tree_keywords, root))
    cand += 1



def grow(n, tree, dist, keywords, i, tree_keywords, ver_word, dp, pq, cand, prune, k_dist=None, len_isk=None):
    for j in range(n): # grow
        if j not in tree and dist[j, i] != np.inf: # (backward search)
            # print(f"grow: {i}-{j}, {tree_keywords}, {tree}")
            new_tree = tree.union(set([j]))  # Extend the tree with the new node j
            new_score = calculate_tree_distance(new_tree, dist)
            # print(f"new_tree: {new_tree}, {new_score}")

            if k_dist != None and len_isk and new_score >= k_dist:
                continue

            for k, keyword in enumerate(keywords):
                if keyword in tree_keywords: continue
                if ver_word[1 << k][j] != np.inf:
                    new_keywords = tree_keywords.union(set([keyword]))  # Update the set of keywords in the tree
                else:
                    new_keywords = tree_keywords
                    
                new_mask = sum(1 << keywords.index(keyword) for keyword in new_keywords)  # Calculate the new mask value

                # print(f"new_keywords: {new_keywords}; {new_mask}; {dp[new_mask][j]}")
                if new_score < dp[new_mask][j]:
                    dp[new_mask][j] = new_score
                    if prune:
                        update_or_push_tree(pq, new_score, new_tree, new_keywords, j, cand)  # Add the new tree to the priority queue
                    else:
                        heapq.heappush(pq, (new_score, new_tree, new_keywords, j))
                        cand += 1
            
            # print(f"growed: {new_score}, {new_tree}, {new_keywords}, {j}")



def merge(dp, pq, dist, cand, align=False, seeds=None, k_dist=None, comm_file=None, len_isk=None):
    # Merge trees with the same root node
    new_pq = []
    while pq:
        tree_dist1, tree1, tree_keywords1, root1 = heapq.heappop(pq)

        # if len(tree_keywords1) == len(keywords):  # Check if the tree contains all keywords
        #     tree_dist1 = calculate_tree_distance(tree1, dist)  # Calculate the total distance of the tree
        #     top_k_trees.add((tuple(tree1), tree_dist1))  # Add the tree and distance to the top-k trees list
        #     k_dist = max(k_dist, tree_dist1)
        #     continue

        found = False
        for tree_dist2, tree2, tree_keywords2, root2 in pq:
            if k_dist != None and len_isk and tree_dist1 + tree_dist2 >= k_dist: continue # pruning rule 2
            # if k_dist != None and len_isk: continue
            if tree1 == tree2: continue
            judge = (root1 == root2) if not align or seeds == None else ((root1 == root2) or ((root1, root2) in seeds) or ((root2, root1) in seeds))

            if judge == True:
                set1 = tree1.copy()
                set1.discard(root1)
                set2 = tree2.copy()
                set2.discard(root2)
                if ((root1, root2) in seeds) or ((root2, root1) in seeds):
                    with open(comm_file, "a") as f:
                        f.write(f"Tree: {tree_dist1}, {tree1}, {tree_keywords1}, {root1}\n")

                if not set1.intersection(set2) and not tree_keywords1.intersection(tree_keywords2):
                    # print(f"merge: {root1}; {root2}; {tree1}; {tree2}; {tree_keywords1}; {tree_keywords1}")
                    found = True
                    merged_tree_nodes = set1.union(set2)
                    merged_tree_nodes = merged_tree_nodes.union(set([root1]))
                    merged_tree_keywords = tree_keywords1.union(tree_keywords2)
                    merged_dist = calculate_tree_distance(merged_tree_nodes, dist)

                    if k_dist != None and len_isk and merged_dist >= k_dist: continue
                    update_or_push_tree(new_pq, merged_dist, merged_tree_nodes, merged_tree_keywords, root2, cand)
                    # update_or_push_tree(new_pq, merged_dist, merged_tree_nodes, merged_tree_keywords, root1, cand)
                    # print(f"merged ===== {merged_dist}, {merged_tree_nodes}, {merged_tree_keywords}, {root1}-{root2}")

                    # Update dp table with the merged tree's distance value
                    mask = len(merged_tree_keywords)
                    dp[mask][root1] = merged_dist
                
        if not found:
            new_pq.append((tree_dist1, tree1, tree_keywords1, root1))

    for tree_data in new_pq:
        tree_dist, tree, tree_keywords, root = tree_data
        update_or_push_tree(pq, tree_dist, tree, tree_keywords, root, cand)


def grape_steiner_tree(vertex_index, dist, keywords, threshold, topk, comm_file, align=False, seeds=None, prune=True, key_nodes=None):
    n = len(vertex_index)  # Get the number of nodes
    m = len(keywords)  # Get the number of keywords
    dp = np.inf * np.ones((1 << m, n))  # Initialize the dp array to store the shortest path lengths for different keyword combinations (2^m, n)

    start = time.time()
    cand = 0 # candidate number

    # Initialize each keyword node as a separate Steiner tree
    pq = []  # Initialize the priority queue
    for j, keyword in enumerate(keywords):
        if key_nodes is not None:
            for (i, sim) in key_nodes[keyword]:
                if sim <= threshold: continue
                dp[1 << j][i] = 1 - sim
                heapq.heappush(pq, (1 - sim, set([i]), set([keyword]), i))  # Add the initial node to the priority queue
                cand += 1
        else:
            keyword_embedding = get_bert_embedding(keyword, model, tokenizer)  # Get node embedding for keyword
            for i, attr in vertex_index.items():
                strs = attr.replace("\\", "").replace("-", " ").split(" ")
                for str in strs:
                    str_embedding = get_bert_embedding(str, model, tokenizer)  # Get node embedding for node i
                    similarity = calculate_similarity(str_embedding, keyword_embedding)
                    if similarity > threshold:  # Set a threshold for similarity
                        dp[1 << j][i] = 1 - similarity
                        heapq.heappush(pq, (1 - similarity, set([i]), set([keyword]), i))  # Add the initial node to the priority queue
                        cand += 1
                        break
    print(f"initialize dp table and priority queue: {len(pq)}, take time: {time.time() - start} Sec")

    ver_word = dp.copy()
    top_k_trees = set()  # Store the top-k minimum Steiner trees
    k_dist = 0
    while pq and len(top_k_trees) < topk:  # Loop until the priority queue is empty
        tree_dist, tree, tree_keywords, i = heapq.heappop(pq)  # Pop the node with the current minimum path length
        # print(f"tree: {tree}; {i}, {tree_keywords}")

        len_isk = (len(top_k_trees) == topk)
        if len(top_k_trees) >= topk and (tree_dist >= k_dist):  # If topk trees are found and the current path length is smaller than the k-th
            with open(comm_file, "a") as f:
                f.write(f"Score: {tree_dist}, {k_dist}\n")
            continue

        mask = len(tree_keywords)  # Get the mask value based on the number of keywords in the tree

        if mask == len(keywords):  # Check if the tree contains all keywords
            tree_dist = calculate_tree_distance(tree, dist)  # Calculate the total distance of the tree
            top_k_trees.add((tuple(tree), tuple(tree_keywords), tree_dist))  # Add the tree and distance to the top-k trees list
            k_dist = max(k_dist, tree_dist)
            with open(comm_file, "a") as f:
                f.write(f"Tree: {tree_dist}, {tree}, {tree_keywords}, {i}\n")

        grow(n, tree, dist, keywords, i, tree_keywords, ver_word, dp, pq, cand, prune, k_dist, len_isk)
        merge(dp, pq, dist, cand, align, seeds, k_dist, comm_file, len_isk)

    return top_k_trees, cand


# 使用优先队列和Floyd-Warshall算法优化
# Todo: 同一个节点包含多个关键字的情况
def dp_min_steiner_tree(vertex_index, dist, keywords, threshold, topk, align=False, seeds=None, prune=False, key_nodes=None):
    n = len(vertex_index)  # Get the number of nodes
    m = len(keywords)  # Get the number of keywords
    dp = np.inf * np.ones((1 << m, n))  # Initialize the dp array to store the shortest path lengths for different keyword combinations (2^m, n)

    start = time.time()
    cand = 0 # candidate number

    # Initialize each keyword node as a separate Steiner tree
    pq = []  # Initialize the priority queue
    for j, keyword in enumerate(keywords):
        if key_nodes is not None:
            for (i, sim) in key_nodes[keyword]:
                if sim <= threshold: continue
                dp[1 << j][i] = 1 - sim
                heapq.heappush(pq, (1 - sim, set([i]), set([keyword]), i))  # Add the initial node to the priority queue
                cand += 1
        else:
            keyword_embedding = get_bert_embedding(keyword, model, tokenizer)  # Get node embedding for keyword
            for i, attr in vertex_index.items():
                strs = attr.replace("\\", "").replace("-", " ").split(" ")
                for str in strs:
                    str_embedding = get_bert_embedding(str, model, tokenizer)  # Get node embedding for node i
                    similarity = calculate_similarity(str_embedding, keyword_embedding)
                    if similarity > threshold:  # Set a threshold for similarity
                        dp[1 << j][i] = 1 - similarity
                        heapq.heappush(pq, (1 - similarity, set([i]), set([keyword]), i))  # Add the initial node to the priority queue
                        cand += 1
                        break
    print(f"initialize dp table and priority queue: {len(pq)}, take time: {time.time() - start} Sec")

    ver_word = dp.copy()
    top_k_trees = set()  # Store the top-k minimum Steiner trees
    k_dist = 0
    while pq and len(top_k_trees) < topk:  # Loop until the priority queue is empty
        tree_dist, tree, tree_keywords, i = heapq.heappop(pq)  # Pop the node with the current minimum path length
        # print(f"tree: {tree}; {i}, {tree_keywords}")
        if len(top_k_trees) >= topk and (tree_dist > k_dist):  # If topk trees are found and the current path length is larger than the k-th
            continue

        mask = len(tree_keywords)  # Get the mask value based on the number of keywords in the tree

        if mask == len(keywords):  # Check if the tree contains all keywords
            tree_dist = calculate_tree_distance(tree, dist)  # Calculate the total distance of the tree
            top_k_trees.add((tuple(tree), tuple(tree_keywords), tree_dist))  # Add the tree and distance to the top-k trees list
            k_dist = max(k_dist, tree_dist)

        grow(n, tree, dist, keywords, i, tree_keywords, ver_word, dp, pq, cand, prune)
        merge(dp, pq, dist, cand, align, seeds)

    return top_k_trees, cand

"""
if key_nodes != None:
    for j, keyword in enumerate(keywords):
        for (i, sim) in key_nodes[keyword]:
            dp[1 << j][i] = 0
else:
    for j, keyword in enumerate(keywords):
        keyword_embedding = get_bert_embedding(keyword, model, tokenizer)  # Get node embedding for keyword
        for i, attr in vertex_index.items():
            strs = attr.replace("\\", "").replace("-", " ").split(" ")
            for str in strs:
                str_embedding = get_bert_embedding(str, model, tokenizer)  # Get node embedding for node i
                similarity = calculate_similarity(str_embedding, keyword_embedding)
                if similarity > threshold:  # Set a threshold for similarity
                    dp[1 << j][i] = 0
                    # print(f"{keyword} , {str}-{i} : {attr} , {similarity}")
                    break

print(f"dp table inialization takes time: {time.time() - start} Sec")

pq = []  # Initialize the priority queue
for mask in range(1 << m):
    for i in range(n):
        if dp[mask][i] != np.inf:
            words = set([keywords[k] for k in range(m) if mask & (1 << k)])
            heapq.heappush(pq, (dp[mask][i], set([i]), words, i))  # Add the initial node to the priority queue
            cand += 1
"""

def init_dp(vertex_index, keyword_embedding, dp, threshold, j):
    for i, attr in vertex_index.items():
        strs = attr.replace("\\", "").replace("-", " ").split(" ")
        for str in strs:
            str_embedding = get_bert_embedding(str, model, tokenizer)  # Get node embedding for node i
            similarity = calculate_similarity(str_embedding, keyword_embedding)
            if similarity > threshold:  # Set a threshold for similarity
                dp[1 << j][i] = 0
                # print(f"{keyword} , {str}-{i} : {attr} , {similarity}")
                break
    return 


def init_queue(n1, dp1, mask, keywords, m, pq, cand, mark):
    for i in range(n1):
        if dp1[mask][i] != np.inf:
            words = set([keywords[k] for k in range(m) if mask & (1 << k)])
            heapq.heappush(pq, (dp1[mask][i], set([i]), words, i, mark))  # Add the initial node to the priority queue
            cand += 1


def merge_and_complete(pq, seed_dist1, seed_dist2, cand, n1, align, seeds):
    # Merge trees with the same root node
    new_pq = []
    while pq:
        tree_dist1, tree1, tree_keywords1, root1, mark1 = heapq.heappop(pq)

        found = False
        for tree_dist2, tree2, tree_keywords2, root2, mark2 in pq:
            if tree1 == tree2: continue
            judge = (root1 == root2) if not align else (root1 == root2 or (root1, root2) in seeds or (root2, root1) in seeds)
            if judge == True:
                set1 = tree1.copy()
                set1.discard(root1)
                set2 = tree2.copy()
                set2.discard(root2)
                if not set1.intersection(set2) and not tree_keywords1.intersection(tree_keywords2):
                    # print(f"merge: {root1}; {root2}; {tree1}; {tree2}; {tree_keywords1}; {tree_keywords1}")
                    found = True
                    if root1 == root2:
                        merged_tree_nodes = set1.union(set2)
                        merged_tree_nodes = merged_tree_nodes.union(set([root1]))
                        if root1 < n1: mark = 'l'
                        else: mark = 'r'

                    elif root1 < root2: # left
                        merged_tree_nodes = set1.union([nd + n1 for nd in set2])
                        merged_tree_nodes = merged_tree_nodes.union(set([n1 + root2]))
                        mark = 'l-r'

                    else: # right
                        merged_tree_nodes = set2.union([nd + n1 for nd in set1])
                        merged_tree_nodes = merged_tree_nodes.union(set([root2 + n1]))
                        mark = 'r-l'

                    merged_tree_keywords = tree_keywords1.union(tree_keywords2)
                    merged_dist = cal_complete_tree_distance(merged_tree_nodes, dist, n, seed_dist)
                    calculate_tree_distance(merged_tree_nodes, seed_dist1, seed_dist2)
                    update_or_push_tree(new_pq, merged_dist, merged_tree_nodes, merged_tree_keywords, root2, cand, mark)                    
                
        # if not found:
        new_pq.append((tree_dist1, tree1, tree_keywords1, root1, mark1))

    for tree_data in new_pq:
        tree_dist, tree, tree_keywords, root, mark = tree_data
        update_or_push_tree(pq, tree_dist, tree, tree_keywords, root, cand, mark)



def cal_seeds_dist(dist1, dist2, seed):
    
    n = dist1.shape[0]
    m = dist2.shape[0]
    # seed_dist1 = {i: [] for i in range(n)}
    # seed_dist2 = {j: [] for j in range(m)}
    seed_dist1 = {}
    seed_dist2 = {}
    
    for (i, j) in seed:
        seed_dist1[i] = [dist2[j][k] for k in range(m)]
        seed_dist2[j] = [dist1[i][k] for k in range(n)]

    print(f"dist1: {len(dist1)}, dist2: {len(dist2)}, seed_dist1: {len(seed_dist1)}, seed_dist2: {len(seed_dist2)}")

    for i in range(n):
        if i in seed_dist1.keys():
            # print(f"seed_dist1[i]: {i}; {seed_dist1[i]}")
            for j in range(m):
                for k in range(m):
                    seed_dist1[i][j] = min(seed_dist1[i][j], seed_dist1[i][k] + dist1[k][j])

        if i in seed_dist2.keys():
            for j in range(m):
                for k in range(m):
                    seed_dist2[i][j] = min(seed_dist2[i][j], seed_dist2[i][k] + dist2[k][j])
    
    return seed_dist1, seed_dist2


def complete_dp_min_steiner_tree(vertex_index1, vertex_index2, dist1, dist2, keywords, threshold, topk, align, seeds):
    n1 = len(vertex_index1)
    n2 = len(vertex_index2)  # Get the number of nodes
    m = len(keywords)  # Get the number of keywords
    dp1 = np.inf * np.ones((1 << m, n1))  
    dp2 = np.inf * np.ones((1 << m, n2))

    start = time.time()
    cand = 0 # candidate number
    seed_dist1, seed_dist2 = cal_seeds_dist(dist1, dist2, seeds)

    # Initialize each keyword node as a separate Steiner tree
    for j, keyword in enumerate(keywords):
        keyword_embedding = get_bert_embedding(keyword, model, tokenizer)  # Get node embedding for keyword
        init_dp(vertex_index1, keyword_embedding, dp1, threshold, j)
        init_dp(vertex_index2, keyword_embedding, dp2, threshold, j)
    print(f"dp table inialization takes time: {time.time() - start} Sec")

    begin = time.time()
    ver_word1 = dp1.copy()
    ver_word2 = dp2.copy()

    pq = []  # Initialize the priority queue
    for mask in range(1 << m):
        init_queue(n1, dp1, mask, keywords, m, pq, cand, "l")
        init_queue(n2, dp2, mask, keywords, m, pq, cand, "r")
    print(f"initialize priority queue: {len(pq)}, takes time: {time.time() - begin} Sec")

    top_k_trees = set()  # Store the top-k minimum Steiner trees
    k_dist = 0
    while pq and len(top_k_trees) < topk:  # Loop until the priority queue is empty
        tree_dist, tree, tree_keywords, i, mark = heapq.heappop(pq)  # Pop the node with the current minimum path length
        print(f"tree: {tree}; {i}, {tree_keywords}")
        if len(top_k_trees) >= topk and (tree_dist < k_dist):  # If topk trees are found and the current path length is smaller than the k-th
            continue

        mask = len(tree_keywords)  # Get the mask value based on the number of keywords in the tree

        if mask == len(keywords):  # Check if the tree contains all keywords
            if mark == 'l': # Calculate the total distance of the tree
                tree_dist = calculate_tree_distance(tree, dist1)  
                grow(n1, tree, dist1, keywords, i, tree_keywords, ver_word1, dp1, pq, cand, prune=False)
            elif mark == 'r':
                tree_dist = calculate_tree_distance(tree, dist2)  
                grow(n2, tree, dist2, keywords, i, tree_keywords, ver_word2, dp2, pq, cand, prune=False)
            elif mark == 'l-r':
                tree_dist = cal_complete_tree_distance(tree, dist1, n1, seed_dist1)
                grow(n2, tree, dist2, keywords, i, tree_keywords, ver_word2, dp2, pq, cand, prune=False)
            elif mark == 'r-l':
                tree_dist = cal_complete_tree_distance(tree, dist1, n1, seed_dist2)
                grow(n1, tree, dist1, keywords, i, tree_keywords, ver_word1, dp1, pq, cand, prune=False)
            
            top_k_trees.add((tuple(tree), tuple(tree_keywords), tree_dist))  # Add the tree and distance to the top-k trees list
            k_dist = max(k_dist, tree_dist)
        
        merge_and_complete(pq, seed_dist1, seed_dist2, cand, n1, align, seeds)

    return top_k_trees, cand


from collections import deque

def find_similar_nodes(triplets, keywords, threshold, contain):
    tokens = 0
    B = set()
    key_nodes = {keyword: set() for keyword in keywords}
    if contain == False:
        keyword_embeddings = {keyword: get_bert_embedding(keyword, model, tokenizer) for keyword in keywords}
    
    for triplet in triplets:
        start_attr, _, end_attr = triplet
        if contain:
            start_attr = str(start_attr)
            end_attr = str(end_attr)
            for keyword in keywords:
                if keyword in start_attr:
                    B.add(start_attr)
                    key_nodes[keyword].add((start_attr, 1))
                if keyword in end_attr:
                    B.add(end_attr)
                    key_nodes[keyword].add((end_attr, 1))
        else: 
            strs = str(start_attr).split(" ")
            estrs = str(end_attr).split(" ") 
            tokens += len(strs)
            tokens += len(estrs)

            for string in strs:
                start_emb = get_bert_embedding(string, model, tokenizer)
                start_sim = [(calculate_similarity(start_emb, keyword_embeddings[keyword]), keyword) for keyword in keywords]
                for (sim, keyword) in start_sim:
                    if sim > threshold:
                        B.add(start_attr)
                        key_nodes[keyword].add((start_attr, sim))
                        # break
            
            for string in estrs:
                end_emb = get_bert_embedding(string, model, tokenizer)
                end_sim = [(calculate_similarity(end_emb, keyword_embeddings[keyword]), keyword) for keyword in keywords]
                for (sim, keyword) in end_sim:
                    if sim > threshold:
                        B.add(end_attr)
                        key_nodes[keyword].add((end_attr, sim))
                            # break

    print(f"total tokens: {tokens}")

    return B, key_nodes


def get_hops_subgraph(graph, triple, h):
    visited = set()
    subgraph = set()
    queue = deque([(triple, 0)])
    
    while queue:
        curr_triple, hops = queue.popleft()
        if hops > h: break
        
        subgraph.add(curr_triple)
        visited.add(curr_triple)
        for tr in graph:
            if curr_triple[0] in tr or curr_triple[2] in tr:
                if tr not in visited:
                    queue.append((tr, hops+1))
    
    return subgraph


def astar(graph, start, d):
    open_list = []  # 用于存储待探索节点的优先队列
    closed = set()  # 用于存储已访问过的节点
    parents = {}  # 存储节点的父节点
    distances = {start: 0}  # 存储节点与起始节点的距离
    
    heapq.heappush(open_list, (0, start))
    
    while open_list:
        f, current = heapq.heappop(open_list) # 弹出当前具有最小总代价的节点
        
        if distances[current] > d:
            break
        
        closed.add(current)
        
        for (edge_start, edge_attr, edge_end) in graph:
            if edge_end == current: # 如果三元组的尾节点是当前节点
                new_distance = distances[current] + 1
                if new_distance < d and (edge_start not in distances or new_distance < distances[edge_start]):
                    # 如果新距离小于最大距离d且没有记录过该节点或新距离更小
                    distances[edge_start] = new_distance 
                    parents[current] = (edge_start, edge_attr)
                    heapq.heappush(open_list, (new_distance, edge_start))
    
    # edge_dict = {(edge_start, edge_end): edge_attr for edge_start, edge_attr, edge_end in graph}

    subgraph = []
    for node, parent in parents.items():
        if distances[node] < d:
            subgraph.append((parent[0], parent[1], node))

    return subgraph



def extract_related_triplets(subfile, graph, keywords, threshold, hops, contain):
    graph_file = subfile + '_subgraph'
    node_file = subfile + f'_key_nodes_{hops}.txt'

    if os.path.exists(node_file):
        key_nodes = {}
        with open(node_file, 'r') as f:
            print(f"read node file: {node_file}")
            for line in f:
                key, value = line.strip().split(': ', 1)
                if key in keywords:
                    value_list = eval(value)
                    # print(f"value_list: {type(value_list)}")
                    key_nodes[key] = {(val.strip(), sim) for val, sim in value_list}
                    # print(f"line: {line}")
 
        
        triplets = set()
        for word in keywords:
            subgraph_file = graph_file + f"_{word}_{hops}.txt"
            with open(subgraph_file, 'r') as f:
                print(f"read subgraph file: {subgraph_file}")
                for line in f:
                    # print(line.rstrip("\n"))
                    triple = line.rstrip("\n").split('\t')
                    triplets.add((triple[0].strip(), triple[1].strip(), triple[2].strip()))

    else:
        triplets = set()
        B, key_nodes = find_similar_nodes(graph, keywords, threshold, contain)
        print(f"similar node triples: {len(B)}")

        for node in B:
            subgraph = astar(graph, node, hops)
            for triplet in subgraph:
                triplets.add(triplet)

        # for triplet in B:
        #     subgraph = get_hops_subgraph(graph, triplet, hops)
        #     for triplet in subgraph:
        #         triplets.append(triplet)

        # if subfile != "":
        #     with open(graph_file, 'w') as f:
        #         for triple in triplets:
        #             f.write(f"{triple[0]}\t{triple[1]}\t{triple[2]}\n")
            
        #     with open(node_file, 'w') as f:
        #         for key, value in key_nodes.items():
        #             f.write(f"{key}: {value}\n")
    
    return triplets, key_nodes


def extract_group_triplets(root, triplets, keywords, threshold, hops):
    tokens = 0
    B = {keyword: set() for keyword in keywords}
    key_nodes = {keyword: set() for keyword in keywords}
    keyword_embeddings = {keyword: get_bert_embedding(keyword, model, tokenizer) for keyword in keywords}
    
    for triplet in triplets:
        start_attr, _, end_attr = triplet
        strs = str(start_attr).split(" ")
        estrs = str(end_attr).split(" ") 
        tokens += len(strs)
        tokens += len(estrs)

        for string in strs:
            start_emb = get_bert_embedding(string, model, tokenizer)
            start_sim = [(calculate_similarity(start_emb, keyword_embeddings[keyword]), keyword) for keyword in keywords]
            for (sim, keyword) in start_sim:
                if sim > threshold:
                    B[keyword].add(start_attr)
                    key_nodes[keyword].add((start_attr, sim))
        
        for string in estrs:
            end_emb = get_bert_embedding(string, model, tokenizer)
            end_sim = [(calculate_similarity(end_emb, keyword_embeddings[keyword]), keyword) for keyword in keywords]
            for (sim, keyword) in end_sim:
                if sim > threshold:
                    B[keyword].add(end_attr)
                    key_nodes[keyword].add((end_attr, sim))
    
    print(f"{root} has tokens: {tokens}")

    node_file = f"{root}_key_nodes_{hops}.txt"
    with open(node_file, 'w') as f:
        for key, value in key_nodes.items():
            f.write(f"{key}: {value}\n")

    for word in keywords:
        sub = set()
        graph_file = f"{root}_subgraph_{word}_{hops}.txt"
        for node in B[word]:
            subgraph = astar(triplets, node, hops)
            for triplet in subgraph:
                sub.add(triplet)

        with open(graph_file, 'w') as f:
            for triplet in sub:
                f.write('\t'.join(triplet) + '\n')
    
    return B, key_nodes



def check_is_cross(lst, index, data_name):
    has_less_than_ = False
    has_greater_than_ = False
    
    for item in lst:
        if item < index:
            has_less_than_ = True
        if item >= index:
            has_greater_than_ = True
    
    if data_name in ["semi-heter", "Drugbank"]:
        return has_greater_than_
    else:
        return has_less_than_ and has_greater_than_
    

