import csv
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

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

def calc_dist(x1,y1,x2,y2):
    dist = np.sqrt((x1-x2)^2 + (y1-y2)^2)
    return dist

def check_area(x, y, area_list):
    areas = []
    for row in area_list:
        if (x >= row[1] and y >= row[2] and x <= row[3] and y <= row[4]):
            if(row[0] not in areas): areas.append(row[0])
    return areas

def add_node(x,y, node_graph, area_list):
    """areas = check_area(x,y,area_list)
    if not areas: return areas  #point not in an area
    

    # ainda nao adicionei tudo

    new_node = np.zeros(len(node_graph[0]) + 1)
    node_graph = np.append(node_graph, new_node, axis=1)
    for area in areas:
        if area == 'A':
            new_node[0] = calc_dist(x,y,63,340)
            new_node[1] = calc_dist(x,y,1083,257)  
        elif area == 'B':
            new_node[6] = calc_dist(x,y,1307,340)
            new_node[1] = calc_dist(x,y,1083,257)
        elif area == 'C':
            new_node[6] = calc_dist(x,y,1307,340)
        elif area == 'D':
            new_node[7] = calc_dist(x,y,2625,1626)
        elif area == 'E':
            new_node[2] = calc_dist(x,y,2511,205)
            new_node[3] = calc_dist(x,y,2681,191)
        elif area == 'F':
            new_node[3] = calc_dist(x,y,2681,191)
            new_node[4] = calc_dist(x,y,3732,129)
        elif area == 'G':
            new_node[4] = calc_dist(x,y,3732,129)
            new_node[5] = calc_dist(x,y,4193,191)
        elif area == 'H':
            new_node[5] = calc_dist(x,y,4193,191)
            new_node[10] = calc_dist(x,y,4385,2086)
        elif area == 'I':
            new_node[3] = calc_dist(x,y,2681,191)
        elif area == 'J':
            new_node[7] = calc_dist(x,y,2625,1626)
        elif area == 'K':
            new_node[9] = calc_dist(x,y,2731,1808)
        elif area == 'L':
            new_node[8] = calc_dist(x,y,2805,1606)
        elif area == 'M':
            new_node[11] = calc_dist(x,y,1391,3134)
        elif area == 'N':
            new_node[12] = calc_dist(x,y,1391,3946)
        elif area == 'O':
            new_node[9] = calc_dist(x,y,2731,1808)
            new_node[15] = calc_dist(x,y,2959,3826)
        elif area == 'P':
            new_node[13] = calc_dist(x,y,1611,3928)
        elif area == 'Q':
            new_node[17] = calc_dist(x,y,1449,4876)
        elif area == 'R':
            new_node[13] = calc_dist(x,y,1611,3928)
        elif area == 'S':
            new_node[14] = calc_dist(x,y,2761,3852)
        elif area == 'T':
            new_node[15] = calc_dist(x,y,2959,3826)
        elif area == 'U':
            new_node[19] = calc_dist(x,y,2893,5279)
        elif area == 'V':
            new_node[15] = calc_dist(x,y,2959,3826)
        elif area == 'W':
            new_node[18] = calc_dist(x,y,1679,4860)
        elif area == 'X':
            new_node[19] = calc_dist(x,y,2893,5279)
            new_node[20] = calc_dist(x,y,3059,5269)
        elif area == 'Y':
            new_node[20] = calc_dist(x,y,3059,5269)
            new_node[21] = calc_dist(x,y,4585,5014)
        elif area == 'Z':
            new_node[16] = calc_dist(x,y,4491,3169)
            new_node[21] = calc_dist(x,y,4585,5014)
        
        node_graph.append(new_node)"""
    #node_graph.append(new_node)

    return node_graph



def read_nodes_file():
    file1 = open('path_nodes_position.csv')
    type(file1)

    csvreader = csv.reader(file1)


    header = []
    header = next(csvreader)

    nodes_graph = []
    for row in csvreader:
        for i in range(22):
            row[i] = float(row[i])
        nodes_graph.append(row)
    return nodes_graph

def read_area_file():
    file = open('rectangles_position.csv')
    type(file)

    csvreader = csv.reader(file)

    header = []
    header = next(csvreader)

    area_list = []
    for row in csvreader:
        for i in range(9):
            if(i != 0): row[i] = int(row[i])
        area_list.append(row)
    return area_list



######################## MAIN ############################
nodes_graph = read_nodes_file()
area_list = read_area_file()

# Driver program
g = Graph()
#g.graph = graph

valid_points = False

while(not valid_points):
    mapa_ist = plt.imread('ist_map.png')

    plt.imshow(mapa_ist)
    inputs = plt.ginput(2)

    x_init = int(round(inputs[0][0]))
    y_init = int(round(inputs[0][1]))
    x_end = int(round(inputs[1][0]))
    y_end = int(round(inputs[1][1]))

    only_street_mat = plt.imread("ist_only_streets.png")
    
    #Note: only_street_mat is transposed
    if(only_street_mat[y_init][x_init] == 1 or only_street_mat[y_end][x_end] == 1): 
        print()
        print("Invalid input (outside of valid street)!")
        continue

    nodes_graph = add_node(x_init, y_init, nodes_graph, area_list)
    if(not nodes_graph): 
        print("Invalid input (outside of valid area)!")
        continue

    nodes_graph = add_node(x_end, y_end, nodes_graph, area_list)
    if(not nodes_graph): 
        print("Invalid input (outside of valid area)!")
        continue

    valid_points = True

#Inputs Grafo, origem, destino
g.dijkstra(nodes_graph, 2, 7)
 
