import logging
import json
import pandas as pd
import time
import math


def rdb2rdf(path, files, cut=False):
    for f in files:
        file = path + f
        pre = "l " if f.startswith("left") else "r "
        print(f"read rdb file: {file}")
        with open(file, 'r') as fi:
            lines = fi.readlines()

        # 提取属性行和数据行
        if cut:
            attrs = lines[0].strip().split(',')
            cut_num = math.ceil(len(attrs) / 2)
            attributes = attrs[1:cut_num]
            table_data = [tuple(line.strip().split(',')[:cut_num]) for line in lines[1:]] ## todo
        else:
            attributes = lines[0].strip().split(',')[1:]
            table_data = [tuple(line.strip().split(',')) for line in lines[1:]] ## todo

        # 将关系表格数据转为graph的三元组
        graph_data = []
        for i, row in enumerate(table_data):
            # print(f"{i}; {row}")
            graph_data.append(("table " + file.split(".")[0], "has", pre + str(i)))
            for j, val in enumerate(row[1:]):
                if val == '': continue
                graph_data.append((pre + row[0], attributes[j], val))
    
    return graph_data


def json2rdf(path, files, cut=False):
    for file in files:
        fi = path + file
        print(f"read json file: {fi}")
        with open(fi, 'r') as f:
            data = json.load(f)

        pre = "l " if file.startswith("left") else "r "

        # 将json item转为三元组列表
        triples = set()
        for item in data:
            cut_num = math.ceil(len(item) / 2)
            if "id" in item.keys(): iid = str(item["id"])  
            elif "Id" in item.keys(): iid = str(item["Id"])
            else: iid = str(item["ID"])

            triples.add(("object " + file.split(".")[0], "has", pre + iid))
            for index, (key, value) in enumerate(item.items()):
                if key in ["id", "ID", "Id"]: continue
                if cut and index > cut_num: break
                # if iid == '5': 
                # print(f"{type(key)}, {type(value)}: {key}-{value}")
                if isinstance(value, list) or 'list' in str(type(value)):
                    value = set(value)
                    for val in value:
                        triples.add((pre + iid, key, val))
                elif isinstance(value, dict) or 'dict' in str(type(value)):
                    for k,v in value.items():
                        if isinstance(v, list) or 'list' in str(type(v)):
                            for val in v:
                                triples.add((pre + iid, k, val))
                        else: triples.add((pre + iid, k, v))
                else:
                    triples.add((pre + iid, key, value))
            
    return triples


def read_graph(triples):
    vertex = set()
    for triple in triples:
        a, b, c = triple
        vertex.add(a)
        vertex.add(c)
        # print(triple)
    
    vertex_index = {ind: v for ind, v in enumerate(vertex)}

    return vertex_index


read_graphs_funs = {
    "rel-heter": (rdb2rdf, rdb2rdf),
    "semi-homo": (json2rdf, json2rdf),
    "semi-heter": (json2rdf, json2rdf),
    "semi-rel": (rdb2rdf, json2rdf),
    # "semi-text-c": (read_semi_entities, read_text_entities),
    # "semi-text-w": (read_semi_entities, read_text_entities),
    # "rel-text": (read_text_entities, rdb2rdf),
    "geo-heter": (rdb2rdf, rdb2rdf),
    # "ukgov":(read_multi_entities, read_multi_entities)
}

floader = {
    "rel-heter": ["left.csv", "right.csv"],
    "semi-homo": ["left.json", "right.json"],
    "semi-heter": ["left.json", "right.json"],
    "semi-rel": ["left.csv", "right.json"],
    "semi-text-c": ["left.json", "right.txt"],
    "semi-text-w": ["left.json", "right.txt"],
    "rel-text": ["left.csv", "right.csv"],
    "geo-heter": ["left.csv", "right.csv"],
    "ukgov": ["left_1.csv", "right_1.csv", "left_2.csv", "right_2.csv", "left_3.csv", "right_3.csv"]
}


def read_unibench_left(cut=True):
    root = "data/Unibench/"
    graph_data = set()

    for f in ["Customer.csv", "Product.csv"]:
        file = root + f
        print(f"read rdb file: {file}")
        with open(file, 'r') as fi:
            lines = fi.readlines()

        # 提取属性行和数据行
        attrs = lines[0].strip().split(',')
        cut_num = int(2 * len(attrs) / 3)
        attributes = attrs[1:cut_num]
        table_data = [tuple(line.strip().split(',')[:cut_num]) for line in lines[1:]] ## todo

        # 将关系表格数据转为graph的三元组
        for i, row in enumerate(table_data):
            # print(f"{i}; {row}")
            graph_data.add(("table " + f.split(".")[0], "has", str(i)))
            for j, val in enumerate(row[1:]):
                if val == '': continue
                graph_data.add((row[0], attributes[j], val))


    with open(root + "Order.txt", 'r') as f:
        print(f"read json file") 

        for line in f:
            data = json.loads(line)
            graph_data.add(("object ", "has", "order_" + data['OrderId']))

            for key, value in data.items():
                if key == 'OrderId': continue
                if key == 'contains':
                    for item in value:
                        for index, (sub_key, sub_value) in enumerate(item.items()):
                            if index > 2: break
                            if sub_key == 'productId':
                                graph_data.add(('order_' + data['OrderId'], key, 'product_' + item['productId']))
                            else:
                                graph_data.add(('product_' + item['productId'], sub_key, sub_value))
                        
                elif key == 'CustomerId':
                    graph_data.add(('order_' + data['OrderId'], 'CustomerId', "customer_" + value))
                else:
                    graph_data.add(('order_' + data['OrderId'], key, value))
    
    return graph_data


def read_Drugbank_left(cut=True):
    root = "data/Drugbank/"
    graph_data = []
    
    for f in ["Chembl_uniport_mapping.csv","Chembl.csv", "Drug_indication.csv", "Drug_info.csv", 
              "Drug_tchem.csv", "Drug_tclin.csv", "Target.csv"]:
        file = root + f
        print(f"read rdb file: {file}")
        with open(file, 'r') as fi:
            lines = fi.readlines()

        # 提取属性行和数据行
        attributes = lines[0].strip().split(',')[1:]
        table_data = [tuple(line.strip().split(',')) for line in lines[1:]] ## todo

        # 将关系表格数据转为graph的三元组
        for i, row in enumerate(table_data):
            # print(f"{i}; {row}")
            graph_data.append(("table " + f.split(".")[0], "has", str(i)))
            for j, val in enumerate(row[1:]):
                if val == '': continue
                graph_data.append((row[0], attributes[j], val))
    
    return graph_data



def read_unibench_right(data_name):
    file = f"data/{data_name}/graph.txt"
    print(f"read graph file: {file}") 
    with open(file, 'r') as f:
        lines = f.readlines()
    
    graph_data = [tuple(line.strip().split(',')) for line in lines]

    return graph_data


def read_unibench_seed(data_name):
    seeds = list()
    file = f"data/{data_name}/entities.txt"
    with open(file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        item = line.split(",")
        seeds.append((item[0], item[1]))

    print(f"seeds number: {len(seeds)}")

    return seeds



def read_graphs(data_type):
    begin = time.time()
    if data_type == 'Unibench':
        left_graph = read_unibench_left(cut=True)
        right_graph = read_unibench_right(data_type)

    elif data_type == 'Drugbank':
        left_graph = read_Drugbank_left(cut=True)
        right_graph = read_unibench_right(data_type)

    else:
        read_graphs_fun = read_graphs_funs[data_type]
        file = floader[data_type]
        left_graph = read_graphs_fun[0](f"data/{data_type}/", [f for f in file if f.startswith('left')], cut=True)
        right_graph = read_graphs_fun[1](f"data/{data_type}/", [f for f in file if f.startswith('right')], cut=False)

    left_graph = list(left_graph)
    right_graph = list(right_graph)

    # with open(f"tmp/{data_type}_left.txt", "w") as f:
    #     for l in left_graph:
    #         f.write(f"{l}\n")

    # with open(f"tmp/{data_type}_right.txt", "w") as f:
    #     for r in right_graph:
    #         f.write(f"{r}\n")

    print(f"data mapping takes time: {time.time() - begin} Sec")
    left_vertex_index = read_graph(left_graph)
    right_vertex_index = read_graph(right_graph)
    
    return left_graph, left_vertex_index, right_graph, right_vertex_index


def read_seed(data_name):
    begin = time.time()
    if data_name == 'Unibench' or data_name == 'Drugbank':
        return read_unibench_seed(data_name)

    else:
        seeds = list()
        for fi in ["train", "test", "valid"]:
            file = "data/" + data_name + "/" + fi + ".csv"
            df = pd.read_csv(file, header=None)

            # seed = df[df['label'] == 1][['id1', 'id2']].values.tolist()
            seed = df[df.iloc[:, 2] == 1].iloc[:, [0, 1]].values.tolist()
            seeds.extend(seed)

        seeds_modified = [('l ' + str(id1), 'r ' + str(id2)) for id1, id2 in seeds]
        print(f"total seeds number: {len(seeds_modified)}, takes time {time.time() - begin}")
        # print(f"seeds_modified: {seeds_modified}")

        return seeds_modified


queries = {
    "rel-heter": {1: ['angeles'], 2: ['angeles', 'french'], 3: ['angeles', 'french', 'chicago'], 
                  4: ['angeles', 'french', 'chicago', 'american'], 5: ['angeles', 'french', 'chicago', 'american', 'river'],
                  6: ['angeles', 'french', 'chicago', 'american', 'river', 'chinese']},

    "semi-homo": {1: ['retrieval'], 2: ['retrieval', 'vldb'], 3: ['retrieval', 'vldb', 'sigmod'], 
                  4: ['retrieval', 'vldb', 'sigmod', 'garth'], 5: ['retrieval', 'vldb', 'sigmod', 'garth', 'model'], 
                  6: ['retrieval', 'vldb', 'sigmod', 'garth', 'model', 'harris']},
    "semi-heter": {1: ['learning'], 2: ['learning', 'english'], 3: ['learning', 'english', 'wiley'], 
                  4: ['learning', 'english', 'wiley', 'rebecca'], 5: ['learning', 'english', 'wiley', 'rebecca', 'database'], 
                  6: ['learning', 'english', 'wiley', 'rebecca', 'database', 'park']},
    "semi-rel": {1: ['billy'], 2: ['billy', 'peter'], 3: ['billy', 'peter', 'steve'], 
                  4: ['billy', 'peter', 'steve', 'usa'], 5: ['billy', 'peter', 'steve', 'usa', 'comedy'], 
                  6: ['billy', 'peter', 'steve', 'usa', 'comedy', 'discovery']},
    "geo-heter": {1: ['house'], 2: ['house', 'street'], 3: ['house', 'street', 'west'], 
                  4: ['house', 'street', 'west', 'station'], 5: ['house', 'street', 'west', 'station', 'center'], 
                  6: ['house', 'street', 'west', 'station', 'center', 'market']},
    "Unibench": {1: ['Ring'], 2: ["Monster", "Sunglasses"], 3: ["Ring", "Monster", "Sunglasses"], 
                  4: ["knife", "Ring", "Monster", "Sunglasses"], 5: ["knife", "Ring", "Monster", "Sunglasses", "Trainer"], 
                  6: ["Claudia", "knife", "Ring", "Monster", "Sunglasses", "Trainer"]},
    "Drugbank": {1: ["Moroidin"], 2: ["alogliptin", "Moroidin"], 3: ["alogliptin", "peptidase", "selective"],
                 4: ["alogliptin", "peptidase", "selective", "Moroidin"], 5: ["alogliptin", "peptidase", "selective", "phobia", "Moroidin"],
                 6: ["Moroidin", "alogliptin", "peptidase", "selective", "phobia", "receptor"]}
}

"""
1: ['chicago'], 2: ['chicago', 'westwood'], 3: ['chicago', 'westwood', 'hollywood'], 
                  4: ['chicago', 'westwood', 'hollywood', 'angeles'], 5: ['chicago', 'westwood', 'hollywood', 'angeles', 'french'],
                  6: ['chicago', 'westwood', 'hollywood', 'angeles', 'french', 'coffee']
"""

group = {
    "rel-heter": ['angeles', 'french', 'chicago', 'american', 'river', 'chinese'],
    "semi-homo": ['retrieval', 'vldb', 'sigmod', 'garth', 'model', 'harris'],
    "semi-heter": ['learning', 'english', 'wiley', 'rebecca', 'database', 'park'],
    "semi-rel": ['billy', 'peter', 'steve', 'usa', 'comedy', 'discovery'],
    "geo-heter": ['house', 'street', 'west', 'station', 'center', 'market'],
    "Drugbank": ["Moroidin", "alogliptin", "peptidase", "selective", "phobia", "receptor"],
    "Unibench": ["Claudia", "knife", "Ring", "Monster", "Sunglasses", "Trainer"]
}
