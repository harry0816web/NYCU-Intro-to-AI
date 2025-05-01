import csv
import queue

edgeFile = 'edges.csv'
graph = {}

# 讀取 CSV 並建立鄰接表
with open(edgeFile, 'r') as f:
    reader = csv.reader(f)

    # 跳過header
    header = next(reader, None)
    for row in reader:
        start, end, distance = int(row[0]), int(row[1]), float(row[2])
        if start not in graph:
            graph[start] = []
        graph[start].append((end, distance))

# BFS 實作
def bfs(start, end):
    qu = queue.Queue()
    qu.put((start, [start], 0))  # (當前節點, 路徑, 距離)
    visited = set()

    while not qu.empty():
        node, path, dist = qu.get()

        if node in visited:
            continue
        visited.add(node)

        if node == end:
            dist = round(dist, 3)
            return path, dist, len(visited)

        for neighbor, cost in graph.get(node, []):
            if neighbor not in visited:
                qu.put((neighbor, path + [neighbor], dist + cost))

    return [], float('inf'), len(visited)  # 若找不到路徑

# 測試 BFS
if __name__ == '__main__':
    path, dist, num_visited = bfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
