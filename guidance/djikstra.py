import csv
import numpy as np
from collections import defaultdict
"""
# Python program for Dijkstra's single
# source shortest path algorithm. The program is
# for adjacency matrix representation of the graph
class Graph():
    def __init__(self, vertices):
        self.V = vertices
        print(self.V)
        self.graph = [[0 for column in range(vertices)]
                      for row in range(vertices)]
 
    def printSolution(self, dist):
        print("Vertex Distance from Source")
        for node in range(self.V):
            print(node, "", dist[node])
 
    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    def minDistance(self, dist, sptSet):
 
        # Initialize minimum distance for next node
        min = 1e7
        
        # Search not nearest vertex not in the
        # shortest path tree
        for v in range(self.V):
            if dist[v] < min and sptSet[v] == False:
                min = dist[v]
                min_index = v
 
        return min_index
 
    # Function that implements Dijkstra's single source
    # shortest path algorithm for a graph represented
    # using adjacency matrix representation
    def dijkstra(self, src):
 
        dist = [1e7] * self.V
        dist[src] = 0
        sptSet = [False] * self.V
        self.paths = []
        paths_aux = []
        
        for cout in range(self.V):
 
            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = self.minDistance(dist, sptSet)
 
            # Put the minimum distance vertex in the
            # shortest path tree
            sptSet[u] = True

            #paths_aux.append(u)
 
            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shortest path tree
            
            for v in range(self.V):
                if (self.graph[u][v] > 0 and sptSet[v] == False and dist[v] > dist[u] + self.graph[u][v]):
                    dist[v] = dist[u] + self.graph[u][v]
                paths_aux.append(v)
            self.paths.append(paths_aux)     
        self.printSolution(dist)
        print(self.paths)
"""
class Graph:
 
    # A utility function to find the
    # vertex with minimum dist value, from
    # the set of vertices still in queue
    def minDistance(self,dist,queue):
        # Initialize min value and min_index as -1
        minimum = float("Inf")
        min_index = -1
         
        # from the dist array,pick one which
        # has min value and is till in queue
        for i in range(len(dist)):
            if dist[i] < minimum and i in queue:
                minimum = dist[i]
                min_index = i
        return min_index
 
 
    # Function to print shortest path
    # from source to j
    # using parent array
    def printPath(self, parent, j):
         
        #Base Case : If j is source
        if parent[j] == -1 :
            print(j)
            return
        self.printPath(parent , parent[j])
        print (j)
         
 
    # A utility function to print
    # the constructed distance
    # array
    """def printSolution(self, dist, parent):
        src = 0
        print("Vertex \t\tDistance from Source\tPath")
        for i in range(1, len(dist)):
            print("\n%d --> %d \t\t%d \t\t\t\t\t" % (src, i, dist[i])),
            self.printPath(parent,i)"""
    def printSolution(self, dist, parent, origin, dest):
        print("Origin: %d\n Destination: %d\nDistance: %d\nPath: " % (origin, dest, dist[dest]))
        self.printPath(parent,dest)
 
 
    '''Function that implements Dijkstra's single source shortest path
    algorithm for a graph represented using adjacency matrix
    representation'''
    def dijkstra(self, graph, src, dest):
 
        row = len(graph)
        col = len(graph[0])
 
        # The output array. dist[i] will hold
        # the shortest distance from src to i
        # Initialize all distances as INFINITE
        dist = [float("Inf")] * row
 
        #Parent array to store
        # shortest path tree
        parent = [-1] * row
 
        # Distance of source vertex
        # from itself is always 0
        dist[src] = 0
     
        # Add all vertices in queue
        queue = []
        for i in range(row):
            queue.append(i)
             
        #Find shortest path for all vertices
        while queue:
 
            # Pick the minimum dist vertex
            # from the set of vertices
            # still in queue
            u = self.minDistance(dist,queue)
 
            # remove min element    
            queue.remove(u)
 
            # Update dist value and parent
            # index of the adjacent vertices of
            # the picked vertex. Consider only
            # those vertices which are still in
            # queue
            for i in range(col):
                '''Update dist[i] only if it is in queue, there is
                an edge from u to i, and total weight of path from
                src to i through u is smaller than current value of
                dist[i]'''
                if graph[u][i] and i in queue:
                    if dist[u] + graph[u][i] < dist[i]:
                        dist[i] = dist[u] + graph[u][i]
                        parent[i] = u
 
 
        # print the constructed distance array
        self.printSolution(dist, parent, src, dest)


file = open('distancia_entre_nodes.csv')
type(file)

csvreader = csv.reader(file)


header = []
header = next(csvreader)

graph = []
for row in csvreader:
    for i in range(22):
        row[i] = float(row[i])
    graph.append(row)


# Driver program
g = Graph()
#g.graph = graph

#Inputs Grafo, origem, destino
g.dijkstra(graph, 2, 7)
 
