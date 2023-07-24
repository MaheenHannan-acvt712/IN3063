#Code clean up for less work
def dimensions(grid):
    w = len(grid[0])
    h = len(grid)
    return w, h

#Converting grid to node-cost graph
def costs(grid):
    x, y = dimensions(grid)
    graph = {}
    for i in range(0, x):
        for j in range(0, y):
            graph.update({(i, j): grid[i][j]})
            
    return graph

#Converting grid to weighted graph
def graph(grid, full=False):
    x, y = dimensions(grid)
    graph = {}
    for i in range(0, x):
        for j in range (0, y):
            subgraphs = {}
            if full:
                if i - 1 >= 0:
                    subgraphs.update({(i-1,j):grid[i-1][j]})
                if j - 1 >= 0:
                    subgraphs.update({(i,j-1):grid[i][j-1]})
            if i + 1 < x:
                subgraphs.update({(i+1,j):grid[i+1][j]})
            if j + 1 < y:
                subgraphs.update({(i,j+1):grid[i][j+1]})
            graph.update({(i, j): subgraphs})
            
    return graph

#Creating a completely weighted graph with pre-set infinities
def infinities(graph, source_node=(0, 0)):
    path = {source_node: (None, 0)}
    try:
        for node in graph:
            if node != source_node:
                path.update({node: (None, float('infinity'))})
        return path
    except:
        return path


