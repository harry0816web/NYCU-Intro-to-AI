import csv
from heapq import *
edgeFile = 'edges.csv'
graph={} #dict

f=open(edgeFile, "r")
file=csv.reader(f)
next(file, None)

for row in file:
    start, end, dis= int(row[0]), int(row[1]), float(row[2])
    if start not in graph:
        graph[start]=[] #list
    graph[start].append((end, dis))
    
def ucs(start, end):
    # Begin your code (Part 3)
    #raise NotImplementedError("To be implemented")
    if start not in graph or end not in graph:
        return [], 0, 0
    
    path = []
    dis={start:0.0}
    final_dis=0
    visited=set()
    parent={start:None}
    
    pq=[(0, start)]
    while pq:
        node_dis, node = heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        if node == end:
            final_dis=node_dis
            while node is not None:
                path.append(node)
                node=parent[node]
            path.reverse()
            return path, round(final_dis, 3), len(visited)
        
        for i, distance in graph.get(node, []):
            if i in visited:
                continue
            if i not in dis or node_dis + distance < dis[i]:
                dis[i] = node_dis + distance
                heappush(pq, (dis[i], i))
                parent[i] = node
    
    return [], 0, 0 
# B 150 push to pq// 100 push
        
    # End your code (Part 3)


if __name__ == '__main__':
    path, dist, num_visited = ucs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
