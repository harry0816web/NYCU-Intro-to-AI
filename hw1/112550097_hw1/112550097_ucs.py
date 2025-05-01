import csv
import heapq

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


def ucs(start, end):
    # Begin your code (Part 3)
    pq = []
    heapq.heappush(pq, (0, start, [start]))  # (cost, node, path)
    visited = set()
    while pq:
        # 每次取出最小cost
        cost, node, path = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        if node == end:
            cost = round(cost, 3)
            return path, cost, len(visited)
        for neighbor, neighbor_cost in graph.get(node, []):
            if neighbor not in visited:
                heapq.heappush(pq, (cost + neighbor_cost, neighbor, path + [neighbor]))
    # End your code (Part 3)


if __name__ == '__main__':
    path, dist, num_visited = ucs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
