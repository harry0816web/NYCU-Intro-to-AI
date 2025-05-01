import csv
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

def bfs(start, end):
    # Begin your code (Part 1)
    #raise NotImplementedError("To be implemented")
    if start not in graph or end not in graph:
        return [], 0, 0
    
    path = []
    dis=0
    visited=set()
    
    q=[start]
    parent={start:None}
    node=0
    while q:
        node=q.pop(0)
        if node == end:
            while node:
                path.append(node)
                if node == start:
                    break
                node=parent[node][0]
                dis=dis+parent[node][1]
            path.reverse()
            return path, round(dis, 3), len(visited)
        
        for i, distance in graph.get(node, []):
            if i in visited:
                continue
            q.append(i)
            parent[i]=[node, distance]
            visited.add(i)
    return [], 0, 0
    # End your code (Part 1)


if __name__ == '__main__':
    path, dist, num_visited = bfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
