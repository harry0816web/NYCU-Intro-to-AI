import queue
qu = queue.Queue()
qu.put((1, [1], 0))
print(qu.get())  # (1, [1], 0)
print(qu.empty())  # True

visited=set()
visited.add(1)
print(1 in visited)  # True
print(2 in visited)  # False

graph = {1: [(2, 1), (3, 2)], 2: [(1, 1)], 3: [(1, 2)]}
print(graph.get(1, []))  # [(2, 1), (3, 2)]

