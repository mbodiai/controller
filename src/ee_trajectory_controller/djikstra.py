import numpy as np

from typing import Tuple

from numpy.typing import NDArray

'''
            A
        3 /   \ 5
         /  1  \
        B ----- C
       3 \    1 / 
            D


'''


def findMinIndex(elements: list[NDArray])-> int:
    distances = [element[0] for element in elements]
    min_idx = np.argmin(distances)
    return min_idx



def djikstra(graph: NDArray, start: int) -> Tuple[NDArray, NDArray]:
    
    n = len(graph)
    distances = np.full(n, np.inf)
    previous = np.full(n, -1)
    pq = [(0, start)]
    
    distances[start] = 0
    
    while (len(pq) > 0 ):
        min_idx = findMinIndex(pq)
        distance, node = pq.pop(min_idx)
        
        if distance > distances[node]:
            continue

        for neighbor in range(n):
            if graph[node][neighbor] > 0: #there is an edge
                alt = distances[node] + graph[node][neighbor]
                if alt < distances[neighbor]:
                    distances[neighbor] = alt
                    previous[neighbor] = node
                    pq.append((alt, neighbor))
        

    return (distances, previous)




def go():
    
    graph = np.array([
        [0, 3, 5, 0],
        [3, 0, 1, 3],
        [5, 1, 0, 1],
        [0, 3, 1, 0],
    ])
    
    distances, previous = djikstra(graph, 0)
    print("distances: {}".format(distances))
    print("previous: {}".format(previous))
    


if __name__ == "__main__":
    go()
