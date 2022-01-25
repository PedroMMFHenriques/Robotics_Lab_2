import csv
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from math import *
from scipy.interpolate import *

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
    def printPath(self, parent, j, path):
        #Base Case : If j is source
        if parent[j] == -1 :
            print(j)
            path.append(j)
            return

        path.append(j)
        self.printPath(parent , parent[j], path)
        
        print (j)
        return(path)
 
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
        print("Origin: %d\n Destination: %d\nDistance: %d\nPath: " % (origin, dest, dist[dest])) #apagar?
        path = []
        path = self.printPath(parent,dest, path)
        
        return(path[::-1])
 
 
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

        return(self.printSolution(dist, parent, src, dest))

def calc_dist(x1,y1,x2,y2):
    dist = sqrt((x1-x2)**2 + (y1-y2)**2)
    return dist

def check_area(x, y, area_list):
    areas = []
    for row in area_list:
        if (x >= row[1] and y >= row[2] and x <= row[3] and y <= row[4]):
            if(row[0] not in areas): areas.append(row[0])
    return areas

def add_node(x,y, node_graph, area_list, final_areas=None, xy_final=None):
    areas = check_area(x,y,area_list)
    if not areas: return None  #point not in an area
   
    #add 1 column for the new node
    node_graph = list(np.hstack((node_graph, np.zeros((len(node_graph),1)))))

    new_node = list(np.zeros(len(node_graph[0])))
    for area in areas:
        if area == 'A':
            new_node[0] = calc_dist(x,y,63,340)
            node_graph[0][-1] = new_node[0]
            new_node[1] = calc_dist(x,y,1083,257)  
            node_graph[1][-1] = new_node[1]

            if(final_areas != None and area in final_areas):
                new_node[-2] = calc_dist(x,y,xy_final[0],xy_final[1])
                node_graph[-1][-1] = new_node[-2]

        elif area == 'B':
            new_node[6] = calc_dist(x,y,1307,340)
            node_graph[6][-1] = new_node[6]
            new_node[1] = calc_dist(x,y,1083,257)
            node_graph[1][-1] = new_node[1]

            if(final_areas != None and area in final_areas):
                new_node[-2] = calc_dist(x,y,xy_final[0],xy_final[1])
                node_graph[-1][-1] = new_node[-2]

        elif area == 'C':
            new_node[6] = calc_dist(x,y,1307,340)
            node_graph[7][-1] = calc_dist(x,y,2625,1626)

            if(final_areas != None and area in final_areas):
                if(node_graph[-1][6] < new_node[6]):
                    new_node[-2] = calc_dist(x,y,xy_final[0],xy_final[1])
                else: node_graph[-1][-1] = calc_dist(x,y,xy_final[0],xy_final[1])

        elif area == 'D':
            new_node[7] = calc_dist(x,y,2625,1626)
            node_graph[2][-1] = calc_dist(x,y,2511,205)

            if(final_areas != None and area in final_areas):
                if(node_graph[-1][7] < new_node[7]):
                    new_node[-2] = calc_dist(x,y,xy_final[0],xy_final[1])
                else: node_graph[-1][-1] = calc_dist(x,y,xy_final[0],xy_final[1])

        elif area == 'E':
            new_node[2] = calc_dist(x,y,2511,205)
            node_graph[2][-1] = new_node[2]
            new_node[3] = calc_dist(x,y,2681,191)
            node_graph[3][-1] = new_node[3]

            if(final_areas != None and area in final_areas):
                new_node[-2] = calc_dist(x,y,xy_final[0],xy_final[1])
                node_graph[-1][-1] = new_node[-2]

        elif area == 'F':
            new_node[3] = calc_dist(x,y,2681,191)
            node_graph[3][-1] = new_node[3]
            new_node[4] = calc_dist(x,y,3732,129)
            node_graph[4][-1] = new_node[4]

            if(final_areas != None and area in final_areas):
                new_node[-2] = calc_dist(x,y,xy_final[0],xy_final[1])
                node_graph[-1][-1] = new_node[-2]

        elif area == 'G':
            new_node[4] = calc_dist(x,y,3732,129)
            node_graph[4][-1] = new_node[4]
            new_node[5] = calc_dist(x,y,4193,191)
            node_graph[5][-1] = new_node[5]

            if(final_areas != None and area in final_areas):
                new_node[-2] = calc_dist(x,y,xy_final[0],xy_final[1])
                node_graph[-1][-1] = new_node[-2]

        elif area == 'H':
            new_node[5] = calc_dist(x,y,4193,191)
            node_graph[5][-1] = new_node[5]
            new_node[10] = calc_dist(x,y,4385,2086)
            node_graph[10][-1] = new_node[10]

            if(final_areas != None and area in final_areas):
                new_node[-2] = calc_dist(x,y,xy_final[0],xy_final[1])
                node_graph[-1][-1] = new_node[-2]

        elif area == 'I':
            new_node[3] = calc_dist(x,y,2681,191)
            node_graph[8][-1] = calc_dist(x,y,2805,1606)

            if(final_areas != None and area in final_areas):
                if(node_graph[-1][3] < new_node[3]): new_node[-2] = calc_dist(x,y,xy_final[0],xy_final[1])
                else: node_graph[-1][-1] = calc_dist(x,y,xy_final[0],xy_final[1])

        elif area == 'J':
            new_node[7] = calc_dist(x,y,2625,1626)
            node_graph[8][-1] = calc_dist(x,y,2805,1606)

            if(final_areas != None and area in final_areas):
                if(node_graph[-1][7] < new_node[7]): new_node[-2] = calc_dist(x,y,xy_final[0],xy_final[1])
                else: node_graph[-1][-1] = calc_dist(x,y,xy_final[0],xy_final[1])

        elif area == 'K':
            new_node[9] = calc_dist(x,y,2731,1808)
            node_graph[7][-1] = calc_dist(x,y,2625,1626)

            if(final_areas != None and area in final_areas):
                if(node_graph[-1][9] < new_node[9]): new_node[-2] = calc_dist(x,y,xy_final[0],xy_final[1])
                else: node_graph[-1][-1] = calc_dist(x,y,xy_final[0],xy_final[1])
                
        elif area == 'L':
            new_node[8] = calc_dist(x,y,2805,1606)
            node_graph[9][-1] = calc_dist(x,y,2731,1808)

            if(final_areas != None and area in final_areas):
                if(node_graph[-1][8] < new_node[8]): new_node[-2] = calc_dist(x,y,xy_final[0],xy_final[1])
                else: node_graph[-1][-1] = calc_dist(x,y,xy_final[0],xy_final[1])

        elif area == 'M':
            new_node[11] = calc_dist(x,y,1391,3134)
            node_graph[6][-1] = calc_dist(x,y,1307,340)

            if(final_areas != None and area in final_areas):
                if(node_graph[-1][11] < new_node[11]): new_node[-2] = calc_dist(x,y,xy_final[0],xy_final[1])
                else: node_graph[-1][-1] = calc_dist(x,y,xy_final[0],xy_final[1])

        elif area == 'N':
            new_node[12] = calc_dist(x,y,1391,3946)
            node_graph[6][-1] = calc_dist(x,y,1391,3134)

            if(final_areas != None and area in final_areas):
                if(node_graph[-1][12] < new_node[12]): new_node[-2] = calc_dist(x,y,xy_final[0],xy_final[1])
                else: node_graph[-1][-1] = calc_dist(x,y,xy_final[0],xy_final[1])

        elif area == 'O':
            new_node[9] = calc_dist(x,y,2731,1808)
            node_graph[9][-1] = new_node[9]
            new_node[15] = calc_dist(x,y,2959,3826)
            node_graph[15][-1] = new_node[15]

            if(final_areas != None and area in final_areas):
                new_node[-2] = calc_dist(x,y,xy_final[0],xy_final[1])
                node_graph[-1][-1] = new_node[-2]

        elif area == 'P':
            new_node[13] = calc_dist(x,y,1611,3928)
            node_graph[12][-1] = calc_dist(x,y,1391,3946)

            if(final_areas != None and area in final_areas):
                if(node_graph[-1][13] < new_node[13]): new_node[-2] = calc_dist(x,y,xy_final[0],xy_final[1])
                else: node_graph[-1][-1] = calc_dist(x,y,xy_final[0],xy_final[1])

        elif area == 'Q':
            new_node[17] = calc_dist(x,y,1449,4876)
            node_graph[12][-1] = calc_dist(x,y,1391,3946)

            if(final_areas != None and area in final_areas):
                if(node_graph[-1][17] < new_node[17]): new_node[-2] = calc_dist(x,y,xy_final[0],xy_final[1])
                else: node_graph[-1][-1] = calc_dist(x,y,xy_final[0],xy_final[1])

        elif area == 'R':
            new_node[13] = calc_dist(x,y,1611,3928)
            node_graph[18][-1] = calc_dist(x,y,1679,4860)

            if(final_areas != None and area in final_areas):
                if(node_graph[-1][13] < new_node[13]): new_node[-2] = calc_dist(x,y,xy_final[0],xy_final[1])
                else: node_graph[-1][-1] = calc_dist(x,y,xy_final[0],xy_final[1])
                
        elif area == 'S':
            new_node[14] = calc_dist(x,y,2761,3852)
            node_graph[13][-1] = calc_dist(x,y,1611,3928)

            if(final_areas != None and area in final_areas):
                if(node_graph[-1][14] < new_node[14]): new_node[-2] = calc_dist(x,y,xy_final[0],xy_final[1])
                else: node_graph[-1][-1] = calc_dist(x,y,xy_final[0],xy_final[1])

        elif area == 'T':
            new_node[15] = calc_dist(x,y,2893,5279)
            node_graph[15][-1] = calc_dist(x,y,2893,5279)
            new_node[14] = calc_dist(x,y,3059,5269)
            node_graph[14][-1] = calc_dist(x,y,3059,5269)

            if(final_areas != None and area in final_areas):
                new_node[-2] = calc_dist(x,y,xy_final[0],xy_final[1])
                node_graph[-1][-1] = new_node[-2]

        elif area == 'U':
            new_node[19] = calc_dist(x,y,2893,5279)
            node_graph[14][-1] = calc_dist(x,y,2761,3852)

            if(final_areas != None and area in final_areas):
                if(node_graph[-1][19] < new_node[19]): new_node[-2] = calc_dist(x,y,xy_final[0],xy_final[1])
                else: node_graph[-1][-1] = calc_dist(x,y,xy_final[0],xy_final[1])

        elif area == 'V':
            new_node[15] = calc_dist(x,y,2959,3826)
            node_graph[20][-1] = calc_dist(x,y,3059,5269)

            if(final_areas != None and area in final_areas):
                if(node_graph[-1][15] < new_node[15]): new_node[-2] = calc_dist(x,y,xy_final[0],xy_final[1])
                else: node_graph[-1][-1] = calc_dist(x,y,xy_final[0],xy_final[1])

        elif area == 'W':
            new_node[18] = calc_dist(x,y,1679,4860)
            node_graph[17][-1] = calc_dist(x,y,1449,4876)

            if(final_areas != None and area in final_areas):
                if(node_graph[-1][18] < new_node[18]): new_node[-2] = calc_dist(x,y,xy_final[0],xy_final[1])
                else: node_graph[-1][-1] = calc_dist(x,y,xy_final[0],xy_final[1])

        elif area == 'X':
            new_node[19] = calc_dist(x,y,2893,5279)
            node_graph[19][-1] = calc_dist(x,y,2893,5279)
            new_node[20] = calc_dist(x,y,3059,5269)
            node_graph[20][-1] = calc_dist(x,y,3059,5269)

            if(final_areas != None and area in final_areas):
                new_node[-2] = calc_dist(x,y,xy_final[0],xy_final[1])
                node_graph[-1][-1] = new_node[-2]

        elif area == 'Y':
            new_node[20] = calc_dist(x,y,3059,5269)
            node_graph[20][-1] = calc_dist(x,y,3059,5269)
            new_node[21] = calc_dist(x,y,4585,5014)
            node_graph[21][-1] = calc_dist(x,y,4585,5014)

            if(final_areas != None and area in final_areas):
                new_node[-2] = calc_dist(x,y,xy_final[0],xy_final[1])
                node_graph[-1][-1] = new_node[-2]

        elif area == 'Z':
            new_node[16] = calc_dist(x,y,4491,3169)
            node_graph[16][-1] = calc_dist(x,y,4491,3169)
            new_node[21] = calc_dist(x,y,4585,5014)
            node_graph[21][-1] = calc_dist(x,y,4585,5014)

            if(final_areas != None and area in final_areas):
                new_node[-2] = calc_dist(x,y,xy_final[0],xy_final[1])
                node_graph[-1][-1] = new_node[-2]
        
    node_graph.append(new_node)

    return node_graph, areas



def read_nodes_dist_file():
    file1 = open('path_nodes_dist.csv')
    type(file1)

    csvreader = csv.reader(file1)

    header = []
    header = next(csvreader)

    nodes_graph = []
    for row in csvreader:
        for i in range(len(row)):
            row[i] = float(row[i])
        nodes_graph.append(row)
    return nodes_graph

def read_nodes_list():
    file = open('path_nodes_positions.csv')
    type(file)

    csvreader = csv.reader(file)

    header = []
    header = next(csvreader)

    nodes_list = []
    for row in csvreader:
        for i in range(len(row)):
            row[i] = int(row[i])
        nodes_list.append(row)
    return nodes_list


def read_area_file():
    file = open('rectangles_position.csv')
    type(file)

    csvreader = csv.reader(file)

    header = []
    header = next(csvreader)

    area_list = []
    for row in csvreader:
        for i in range(len(row)):
            if(i != 0): row[i] = int(row[i])
        area_list.append(row)
    return area_list


def read_small_steps_list():
    file = open('small_steps.csv')
    type(file)

    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)

    steps_list = []
    for row in csvreader:
        for i in range(len(row)):
            
            if(row[i] =='0'): row[i] = int(row[i])
            else:  
                aux = (row[i].split(";"))

                for k in range(len(aux)):
                    aux[k] = aux[k].split(" ")
                    
                    """aux[k][0] = int(aux[k][0])
                    aux[k][1] = int(aux[k][1])"""
                row[i] = aux 
        steps_list.append(row)
    return steps_list

def trajectory_interpol(precise_path):
    trajectory = []
    #velocity = []
    for i in range(len(precise_path) - 1):
        if(i == (len(precise_path)-1)):
            vel = (5/0.05073825503)*1000/3600
            distance = sqrt((precise_path[i-1][0] - precise_path[i][0])**2 + (precise_path[i-1][1] - precise_path[i][1]))
            theta = atan2(precise_path[i][1]-precise_path[i-1][1], precise_path[i][0]-precise_path[i-1][0])
            [traject], [velocity], orientation  = interpol([precise_path[i-1][0], precise_path[i-1][1]], [precise_path[i][0], precise_path[i][1]], [vel*cos(theta), vel*sin(theta)], [vel*cos(theta), vel*sin(theta)], 1, distance/vel)

        else:
            vel = (5/0.05073825503)*1000/3600
            distance = sqrt((precise_path[i][0] - precise_path[i+1][0])**2 + (precise_path[i][1] - precise_path[i+1][1]))
            theta = atan2(precise_path[i+1][1]-precise_path[i][1], precise_path[i+1][0]-precise_path[i][0])
            [traject], [velocity], orientation  = interpol([precise_path[i][0], precise_path[i][1]], [precise_path[i+1][0], precise_path[i+1][1]], [vel*cos(theta), vel*sin(theta)], [vel*cos(theta), vel*sin(theta)], 1, distance/vel)
        trajectory.append(traject)
        if(i == 1):
            print(trajectory)
    return trajectory

def interpol(pos1, pos2, vel1, vel2, t1, t2, vel_max = 5.555, acc_max = 1):
    tfi = t2 - t1
    #interpolacao x
    ai0x = pos1[0]
    ai1x = vel1[0]
    
    ai2x = (3/tfi**2)*(pos2[0] - pos1[0]) - (2/tfi)*vel1[0] - (1/tfi)*vel2[0]
    ai3x = -(2/tfi**3)*(pos2[0] - pos1[0]) + (1/tfi**2)*(vel2[0] - vel1[0])
    
    #interpolacao y 
    ai0y = pos1[1]
    ai1y = vel1[1]
    
    ai2y = (3/tfi**2)*(pos2[1] - pos1[1]) - (2/tfi)*vel1[1] - (1/tfi)*vel2[1]
    ai3y = -(2/tfi**3)*(pos2[1] - pos1[1]) + (1/tfi**2)*(vel2[1] - vel1[1])

    x_interp = []
    vel_x_interp = []
    y_interp = []
    vel_y_interp = []
    orientation = []
    
    for i in np.arange(t1, t2, 0.1):
        x_aux = ai0x + ai1x*i + ai2x*(i**2) + ai3x*(i**3)
        vel_x_aux = ai1x + 2*ai2x*i + 3*ai3x*(i**2)
        if(vel_x_aux > vel_max): vel_x_aux = vel_max
        
        #Possivel interpolacao de aceleracao

        x_interp.append(x_aux)
        vel_x_interp.append(vel_x_aux)
        
        ############################################
        y_aux = ai0y + ai1y*i + ai2y*(i**2) + ai3y*(i**3)
        vel_y_aux = ai1y + 2*ai2y*i + 3*ai3y*(i**2)

        if(vel_y_aux > vel_max): vel_y_aux = vel_max
        #Possivel interpolacao de aceleracao

        y_interp.append(y_aux)
        vel_y_interp.append(vel_y_aux)

        if(i == t1): orientation.append(atan2(y_aux-pos1[1], x_aux-pos1[0]))
        else: orientation.append(atan2(y_aux-y_interp[-1], x_aux-x_interp[-1]))

    return [x_interp, y_interp],[vel_x_interp, vel_y_interp], orientation

def add_prev_and_next_node(path, xy_init, xy_end, area_list):
    areas_init = check_area(xy_init[0], xy_init[0], area_list)
    areas_end = check_area(xy_end[0], xy_end[0], area_list)

    complete_path = []

    for i in range(2):
        if i == 0: areas = areas_init
        else: areas = areas_end

        if path[1] == 0 or path[-2] == 0:
            complete_path.append(1)            
        elif path[1] == 1 or path[-2] == 1:
            if 'A' in areas:
                complete_path.append(0)
            else: 
                complete_path.append(6)
        elif path[1] == 2 or path[-2] == 2:
            complete_path.append(3)
        elif path[1] == 3 or path[-2] == 3:
            if 'F' in areas and 'J' not in areas :
                complete_path.append(4)
            elif 'F' not in areas and 'J' in areas :
                complete_path.append(8)
            #else: intersection of areas -> just go to the next node
        elif path[1] == 4 or path[-2] == 4:
            if 'F' in areas:
                complete_path.append(3)
            else: complete_path.append(5)
        elif path[1] == 5 or path[-2] == 5:
            if 'G' in areas:
                complete_path.append(4)
            else: complete_path.append(10)
        elif path[1] == 6 or path[-2] == 6:
            if 'B' in areas and 'C' not in areas :
                complete_path.append(1)
            elif 'B' not in areas and 'C' in areas :
                complete_path.append(7)
            #else: intersection of areas -> just go to the next node
        elif path[1] == 7 or path[-2] == 7:
            if 'D' in areas and 'J' not in areas :
                complete_path.append(2)
            elif 'D' not in areas and 'J' in areas :
                complete_path.append(8)
            #else: intersection of areas -> just go to the next node
        elif path[1] == 8 or path[-2] == 8:
            complete_path.append(9)
        elif path[1] == 9 or path[-2] == 9:
            if 'K' in areas:
                complete_path.append(7)
            else: complete_path.append(15)
        elif path[1] == 10 or path[-2] == 10:
            complete_path.append(5)
        elif path[1] == 11 or path[-2] == 11:
            complete_path.append(6)
        elif path[1] == 12 or path[-2] == 12:
            complete_path.append(11)
        elif path[1] == 13 or path[-2] == 13:
            if 'P' in areas and 'Q' not in areas :
                complete_path.append(12)
            elif 'P' not in areas and 'Q' in areas :
                complete_path.append(18)
            #else: intersection of areas -> just go to the next node
        elif path[1] == 14 or path[-2] == 14:
            if 'S' in areas and 'T' not in areas :
                complete_path.append(13)
            elif 'S' not in areas and 'T' in areas :
                complete_path.append(15)
            #else: intersection of areas -> just go to the next node
        elif path[1] == 15 or path[-2] == 15:
            if 'V' in areas:
                complete_path.append(20)
            elif 'O' in areas and 'T' not in areas :
                complete_path.append(9)
            elif 'O' not in areas and 'T' in areas :
                complete_path.append(14)
            #else: intersection of areas -> just go to the next node
        elif path[1] == 16 or path[-2] == 16:
            complete_path.append(21)
        elif path[1] == 17 or path[-2] == 17:
            complete_path.append(12)
        elif path[1] == 18 or path[-2] == 18:
            complete_path.append(17)
        elif path[1] == 19 or path[-2] == 19:
            if 'U' in areas and 'X' not in areas :
                complete_path.append(14)
            elif 'U' not in areas and 'X' in areas :
                complete_path.append(20)
            #else: intersection of areas -> just go to the next node
        elif path[1] == 20 or path[-2] == 20:
            if 'X' in areas and 'Y' not in areas :
                complete_path.append(19)
            elif 'X' not in areas and 'Y' in areas :
                complete_path.append(21)
            #else: intersection of areas -> just go to the next node
        elif path[1] == 21 or path[-2] == 21:
            if 'Y' in areas:
                complete_path.append(20)
            else: complete_path.append(16)

        complete_path.append(path)
    return complete_path

def init_to_trajectory(complete_path, nodes_list, steps_list):
    prev_node = complete_path[0]
    first_node = complete_path[2]

    init_point = nodes_list[complete_path[1]]
    first_node_point = nodes_list[first_node]

    init_dist = calc_dist(init_point[0], init_point[1], first_node_point[0], first_node_point[1])

    small_steps = steps_list[prev_node][first_node]

    for step_point in small_steps:
        step_dist = calc_dist(step_point[0], step_point[1], first_node_point[0], first_node_point[1])

        if step_dist < init_dist: 
            return init, step atual, steps seguintes


    #return

def gen_precise_path(path, steps_list, nodes_list, area_list):
    if len(path) <= 2: return path

    steps_list = list(np.hstack((steps_list, np.zeros((len(steps_list),2)))))
    init_node = list(np.zeros(len(steps_list[0])))
    end_node = list(np.zeros(len(steps_list[0])))
    steps_list.append(end_node)
    steps_list.append(init_node)

    xy_init = nodes_list[-1]
    xy_end = nodes_list[-2]
    steps_list[path[0]][path[1]] = xy_init
    steps_list[path[-2]][path[-1]] = xy_end

    print(path)
    complete_path = add_prev_and_next_node(path, xy_init, xy_end, area_list)
    print(complete_path)

    precise_path = []
    for i in range(len(path)):
        if i == 0:
            #init_to_trajectory
        elif(i == (len(path) - 1)): 
            precise_path.append(xy_end)
        else:
            for k in range(len(steps_list[path[i]][path[i+1]])):
                precise_path.append((steps_list[path[i]][path[i+1]][k][0], steps_list[path[i]][path[i+1]][k][1]))

    return precise_path

######################## MAIN ############################
nodes_graph = read_nodes_dist_file()
area_list = read_area_file()
nodes_list = read_nodes_list()
steps_list = read_small_steps_list()

# Driver program
g = Graph()
#g.graph = graph

n_nodes = len(nodes_graph)
valid_points = False

while(not valid_points):
    mapa_ist = plt.imread('ist_map.png')

    plt.imshow(mapa_ist)
    
    inputs = plt.ginput(2)
    
    """fig1 = plt.figure() 
    fig1.canvas.mpl_connect('close_event', lambda _: fig1.canvas.manager.window.destroy()) #cena para fechar caso nao fuincione"""
    
    x_init = int(round(inputs[0][0]))
    y_init = int(round(inputs[0][1]))
    x_end = int(round(inputs[1][0]))
    y_end = int(round(inputs[1][1]))

    only_street_mat = plt.imread("ist_only_streets.png")
    
    #Note: only_street_mat is transposed
    if(only_street_mat[y_init][x_init] == 1 or only_street_mat[y_end][x_end] == 1): 
        plt.clf()
        print("Invalid input (outside of valid street)!")
        continue
    
    nodes_graph, final_areas = add_node(x_end, y_end, nodes_graph, area_list)
    if(nodes_graph == None): 
        plt.clf()
        print("Invalid input (outside of valid area)!")
        continue

    nodes_graph, aux = add_node(x_init, y_init, nodes_graph, area_list, final_areas=final_areas, xy_final=[x_end, y_end])
    if(nodes_graph == None):
        nodes_graph = []
        plt.clf()
        print("Invalid input (outside of valid area)!")
        continue
    
    valid_points = True
    #plt.close()

path = g.dijkstra(nodes_graph, n_nodes+1, n_nodes)

nodes_list.append((x_end, y_end))
nodes_list.append((x_init, y_init))

precise_path = path
#precise_path = gen_precise_path(path, steps_list, nodes_list, area_list)

checkpoints_x = []
checkpoints_y = []
for i in range(len(precise_path)):
    checkpoints_x.append(nodes_list[precise_path[i]][0])
    checkpoints_y.append(nodes_list[precise_path[i]][1])


vetor_trajectories = trajectory_interpol(precise_path)
xx = vetor_trajectories[0][:]
yy = vetor_trajectories[1][:]

"""num_pts = np.arange(len(precise_path))

cs_x = CubicSpline(num_pts, checkpoints_x)
cs_y = CubicSpline(num_pts, checkpoints_y)

time = np.arange(0, len(precise_path)-1, 0.01)

xx = cs_x(time)
yy = cs_y(time)"""

orientation = []
for i in range(len(xx)):
    if(i == len(xx) - 1): 
        theta = atan2(yy[i]-yy[i-1], xx[i]-xx[i-1])
    else:
        theta = atan2(yy[i+1]-yy[i], xx[i+1]-xx[i])

    orientation.append(theta)
    

plt.imshow(mapa_ist)
num = int(calc_dist(nodes_graph[precise_path[0]][0], nodes_graph[precise_path[0]][1], nodes_graph[precise_path[1]][0], nodes_graph[precise_path[1]][1])/20)

plt.scatter(xx, yy, s = 10)

plt.savefig('trajectory.png', bbox_inches='tight')
plt.show()


fout = open('trajectory_points.csv', 'w', newline='')
writer = csv.writer(fout)
for i in range(len(xx)):
    if(i == (len(xx)-1)):
        writer.writerow([int(xx[i]), int(yy[i]), orientation[i], orientation[i]-orientation[i-1]])
    else:
        writer.writerow([int(xx[i]), int(yy[i]), orientation[i], orientation[i+1]-orientation[i]])

fout.close()
