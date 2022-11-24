from collections import OrderedDict, defaultdict
from typing import Callable, Tuple, Dict, List

import numpy as np
from tqdm.auto import tqdm


def distance(pointA: np.ndarray, documents: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pointA - documents, axis=1, keepdims=True)

def create_sw_graph(
        data: np.ndarray,
        num_candidates_for_choice_long: int = 10,
        num_edges_long: int = 5,
        num_candidates_for_choice_short: int = 10,
        num_edges_short: int = 5,
        use_sampling: bool = False,
        sampling_share: float = 0.05,
        dist_f: Callable = distance
    ) -> Dict[int, List[int]]:
    graph = {}
    for i in range(data.shape[0]):
        distanses = dist_f(data[i], data)
        distanses = np.argsort(distanses.reshape(1, -1), axis=1)[0][1:]
        far = np.random.choice(distanses[-num_candidates_for_choice_long:], num_edges_long, replace=False)
        close = np.random.choice(distanses[:num_candidates_for_choice_short], num_edges_short, replace=False)
        graph[i] = far.tolist() + close.tolist()
    return graph

def control_queue(queue: list, visited_vertex: dict, search_k: int, all_vertex: list):
    if queue and len(visited_vertex) < search_k:
        remainder = list(set(all_vertex).difference(set(visited_vertex.keys())))
        queue.append(np.random.choice(remainder, 1)[0])

def nsw(query_point: np.ndarray, all_documents: np.ndarray, 
        graph_edges: Dict[int, List[int]],
        search_k: int = 10, num_start_points: int = 5,
        dist_f: Callable = distance) -> np.ndarray:
    
    all_vertex = list(range(all_documents.shape[0]))
    
    queue = list(np.random.choice(all_vertex, num_start_points, replace=False))
    visited_vertex = dict()
    
    while queue:
        point = queue.pop()
        if point in visited_vertex:
            control_queue(queue, visited_vertex, search_k, all_vertex)
            continue
        else:
            neighbours = []
            for neighbour in graph_edges[point]:
                if neighbour in visited_vertex:
                    continue
                neighbours.append(neighbour)
            distances = dist_f(query_point, all_documents[neighbours]).squeeze()
            if len(neighbours) == 1:
                distances = [distances]
            visited_vertex.update(list(zip(neighbours, distances)))
            queue.extend(neighbours)
        control_queue(queue, visited_vertex, search_k, all_vertex)
        
    nearest = list(zip(*sorted(visited_vertex.items(), key=lambda x: x[1])))[0][:search_k]
    return nearest
