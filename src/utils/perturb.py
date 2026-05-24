
import pandas as pd
import random
import ast
import copy

def perturb_graph_objects(objects, drop_prob=0.2, keep_at_least_one=True):
    objects = copy.deepcopy(objects)
    node_ids = list(objects.keys())

    if len(node_ids) == 0:
        return objects

    kept_nodes = [nid for nid in node_ids if random.random() > drop_prob]

    if keep_at_least_one and len(kept_nodes) == 0:
        kept_nodes.append(random.choice(node_ids))

    kept_nodes = set(kept_nodes)

    new_objects = {}
    for nid in kept_nodes:
        obj = objects[nid]
        obj["relations"] = [
            rel for rel in obj.get("relations", [])
            if rel["object"] in kept_nodes
        ]
        new_objects[nid] = obj

    return new_objects



def perturb_dataframe(df, drop_prob=0.2):
    df = df.copy()
    df["objects"] = [
        str(perturb_graph_objects(ast.literal_eval(obj_str), drop_prob=drop_prob))
        for obj_str in df["objects"]
    ]
    return df



