#Importing helper functions and heap functions
from heapq import heappop, heappush
from util import graph, costs, infinities, dimensions
from time import time


class Dijkstras:

    #Initial values
    def __init__(self, grid):
        self.cost = 0
        self.runtime = 0
        self.stack = [(0, 0)]
        self.graph = graph(grid)
        self.costs = costs(grid)
        self.x, self.y = dimensions(grid)

    #Algorithm body
    def algorithm(self, gm):
        
        #Capturing start time
        start_time = time()
        
        #Capturing source and target nodes
        source = (0,0)
        target = (self.x - 1, self.y - 1)
        
        #Creating heaps
        path = infinities(self.graph)
        heap_path = [(0, source)]
        
        #Visiting unvisted paths
        while len(heap_path):
            
            #Capturing current cost and node
            distance, node = heappop(heap_path)

            #Skip if current cost to node is greater than visted node cost
            if distance > path[node][1]:
                continue

            #Iterating through each child node
            for child_node, weight in self.graph[node].items():
                
                #Checking game mode for possible cell movement
                if gm == 1:
                    cost = self.costs[child_node] + weight
                elif gm == 2:
                    cost = abs(int(self.costs[node]) - 
                               int(self.costs[child_node])) + weight
                    
                #Capturing cell movement if current cost
                #is les than visited node cost
                #Source https://bradfieldcs.com/algos/graphs/dijkstras-algorithm/
                if cost < path[child_node][1]:
                    path[child_node] = (node, cost)
                    heappush(heap_path, (cost, child_node))
                #End Source
           
        #Capturing lowest node movement to stack
        #Source https://benalexkeen.com/implementing-djikstras-shortest-path-algorithm-with-python/
        while target is not None:
            self.stack.append(target)
            next_node = path[target][0]
            target = next_node
        #End Source
            
        #Sorting stack and removing duplicates
        self.stack = list(dict.fromkeys(self.stack[::-1]))
        
        #Calculating cost based on game mode
        if gm == 1:
            
            #Iterating through stack and capturing node cost
            for i in self.stack:
                self.cost = self.cost + int(self.costs[i])
                
        elif gm == 2:
            for i in range(0, len(self.stack)-1):
                
                #Capturing previous node and current node
                prev = self.stack[i]
                current = self.stack[i+1]
                
                #Calculating absolute difference of previous and current node
                self.cost = self.cost + (abs(int(self.costs[prev]) -
                                             int(self.costs[current])))

        #Capturing final run time
        self.runtime = format((time() - start_time), '.100g')

    #Returning results
    def get_results(self, gm):
        Dijkstras.algorithm(self, gm)
        return self.runtime, self.cost, len(self.stack), self.stack
