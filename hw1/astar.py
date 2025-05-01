import csv
from heapq import *
edgeFile = 'edges.csv'
heuristicFile = 'heuristic_values.csv'

graph={}
heuristic = {}

def load_graph():
    f1=open(edgeFile, "r")
    file1=csv.reader(f1)
    next(file1, None)

    for row in file1:
        start, end, dis= int(row[0]), int(row[1]), float(row[2])
        if start not in graph:
            graph[start]=[] #list
        graph[start].append((end, dis))

def load_heuristic():
    f2=open(heuristicFile, "r")
    file2=csv.reader(f2)
    header=next(file2, None)
    sample=list(map(int, header[1:]))

    for row in file2:
        n=int(row[0])
        heuristic[n]={sample[i]:float(row[i+1]) for i in range(len(sample))}

def astar(start, end):
    # Begin your code (Part 4)
    #raise NotImplementedError("To be implemented")
    path = []
    dis = {start: 0.0}
    visited = set()
    parent = {start: None}
    
    pq=[(heuristic.get(start, {}).get(end, 0), 0, start)] #h(), g(), node
    while pq:
        _, dis_node, node = heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        
        if node == end:
            while node is not None:
                path.append(node)
                node=parent[node]
            path.reverse()
            return path, round(dis[end], 3), len(visited)
        
        for i, distance in graph.get(node, []):
            if i in visited:
                continue
            
            new_dis=dis_node+distance
            if i not in dis or new_dis<dis[i]:
                dis[i]= new_dis
                total_cost=new_dis + heuristic.get(i, {}).get(end, 0)
                heappush(pq, (total_cost, new_dis, i))
                parent[i]=node
        
    return [], 0, len(visited)
    
    
    
    # End your code (Part 4)

load_graph()
load_heuristic()
if __name__ == '__main__':
    path, dist, num_visited = astar(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
