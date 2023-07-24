#Importing algorithm classes and external modules (Numpy and CSV)
import numpy as np
from dijkstras import Dijkstras
from baseline import Baseline
from util import dimensions
import csv

#Test function
def full_test(runs, outText=False, outFile=False):
    all_results = []
    gameModes = [1, 2]
    
    #Iterating through the runs
    for i in range(1, runs + 1):
        
        #Creating the grid
        grid = np.random.randint(9, size=(i, i), dtype="uint8")
        
        #Iterating through the game modes
        for gm in gameModes:
            
            #Capturing the algorithm functions
            algorithms = [Baseline(grid), Dijkstras(grid)]
            
            #Iterating through the algorithms
            for algorithm in algorithms:
                
                #Capturing the results from each algorithm run
                time, cost, stack_size, stack = algorithm.get_results(gm)
                
                #Formatting the results
                raw_results = ("---Results---\n" +
                      "\n" +
                      "---Grid---\n" +
                      str(grid) +
                      "\n\n" +
                      "Name: " + str(algorithm.__class__.__name__) + "\n" +
                      "Game Mode: " + str(gm) + "\n" +
                      "Grid Size: " + str(i) + "\n" +
                      "Time: " + str(time) + "\n" +
                      "Cost: " + str(cost) + "\n" +
                      "Path (Row, Column): " + str(stack) + "\n" +
                      "Stack Size: " + str(stack_size) + "\n")
   
                #Capturing results
                if time is not float("infinity"):
                    result = {"name": algorithm.__class__.__name__,
                              "size": i*i,
                              "time": time,
                              "cost": cost,
                              "stack": stack_size,
                              "mode": gm}
                    all_results.append(result)
                    
                #Printing results to console
                if outText:
                    print(raw_results)
        
        #Printing progress
        pk = float('%.10g' % (i/runs*100))
        print("Completion: " + str(pk) + "%")
        
    #Exporting results to file (excluding grid)
    if outFile:
        keys = all_results[0].keys()
        with open('results\\results.csv', 'w', newline='') as output:
            output_file = csv.DictWriter(output, keys)
            output_file.writeheader()
            output_file.writerows(all_results)
            
    print()

#Executing test program
if __name__ == '__main__':
    full_test(5, outText=True, outFile=True)
    
