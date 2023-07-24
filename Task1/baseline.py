#Importing helper functions
from util import dimensions, costs
from time import time


class Baseline:
    
    #Initial values
    def __init__(self, grid):
        self.cost = 0
        self.runtime = 0
        self.stack = [(0, 0)]
        self.graph = costs(grid)
        self.x, self.y = dimensions(grid)

    #Algorithm body
    def algorithm(self, gm):
        
        #Capturing start time
        start_time = time()
        
        #Getting grid lengths
        x = self.x - 1
        y = self.y - 1
        
        #Visiting unvisted cells
        while (x, y) not in self.stack:  
            
            #Capturing possible movements
            right = (self.stack[-1][0]+1, self.stack[-1][1])
            down = (self.stack[-1][0], self.stack[-1][1]+1)
            
            #Checking if we are at any edges
            if right[0] <= x and down[1] <= y:
                
                #Checking game mode for possible cell movement
                if gm == 1:
                    
                    #Capturing cell movement
                    if self.graph[(right)] <= self.graph[(down)]:
                        self.stack.append(right)
                    elif self.graph[(right)] > self.graph[(down)]:
                        self.stack.append(down)
                        
                if gm == 2:
                    
                    #Sub function for game mode 2 calculation
                    def calc(a):
                        return abs(
                            int(self.graph[self.stack[-1]]) -
                            int (self.graph[(a)]))
                    
                    #Capturing cell movement
                    if calc(right) <= calc(down):
                        self.stack.append(right)
                    elif calc(right) > calc(down):
                        self.stack.append(down)
            
            #Creating movement when agent is at an edge
            elif right[0] > x:
                self.stack.append(down)
            elif down[1] > y:
                self.stack.append(right)
                         
        #Checking game mode for cost calculation
        if gm == 1:
            for i in self.stack:
                self.cost = int(self.cost) + int(self.graph[i])
        elif gm == 2:
            for i in range(0, len(self.stack)-1):
                prev = self.stack[i]
                current = self.stack[i+1]
                self.cost = self.cost + (abs(int(self.graph[prev]) -
                                             int(self.graph[current])))
        
        #Capturing final run time
        self.runtime = format((time() - start_time), '.100g')

    #Returning results
    def get_results(self, gm):
        Baseline.algorithm(self, gm)
        return self.runtime, self.cost, len(self.stack), self.stack
        
