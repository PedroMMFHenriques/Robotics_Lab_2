import matplotlib
matplotlib.use('Qt5Agg') # pip install pyqt5
import csv
import numpy as np
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
    """
    Calculates the distance between 2 given points
    Input: x and y position of a pair of points
    Output : euclidean distance between the pair of points
    """
    dist = sqrt((x1-x2)**2 + (y1-y2)**2)
    return dist

def check_area(x, y, area_list):
    """
    Makes a list of drivable areas the point is inside
    Input: x and y position of the point; list of drivable areas
    Output: List of the areas the point is inside
    """
    areas = []
    for row in area_list:
        if (x >= row[1] and y >= row[2] and x <= row[3] and y <= row[4]):
            if(row[0] not in areas): areas.append(row[0])
    return areas

def add_node(x,y, node_graph, area_list, final_areas=None, xy_final=None):
    """
    For each area the point is inside, add the point to the node list and assigns which nodes are adjacent to it and which ones it is adjacent to
    Input: x and y position of the point; list of nodes; list of drivable areas; (areas of the end point and its xy coordinates)
    Output: Updated node list with the new node
    """
    areas = check_area(x,y,area_list)
    if not areas: return None, None #point not in an area
   
    #add 1 column for the new node
    node_graph = list(np.hstack((node_graph, np.zeros((len(node_graph),1)))))

    new_node = list(np.zeros(len(node_graph[0])))
    for area in areas:
        if area == 'A':
            new_node[0] = calc_dist(x,y,63,340)
            node_graph[0][-1] = new_node[0]
            new_node[1] = calc_dist(x,y,1083,257)  
            node_graph[1][-1] = new_node[1]

            #if init node and end node in the same area(s):
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
            node_graph[11][-1] = calc_dist(x,y,1391,3134)

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
    """
    Reads file with the distances between each adjacent node and turns the data into a list
    Output: List of the distances between each adjacent node
    """
    file1 = open('resources/path_nodes_dist.csv')
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
    """
    Reads file with the position of each node and turns the data into a list
    Output: List of the position of each node
    """

    file = open('resources/path_nodes_positions.csv')
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
    """
    Reads file with the corner positions of the (rectangular) drivable areas and turns the data into a list
    Output: List of the corner positions of the (rectangular) drivable areas
    """
    file = open("resources/rectangles_position.csv")
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
    """
    Reads file with the position of each step ("mini" nodes between 2 nodes) and turns the data into a list
    Output: List of the position of each step
    """
    file = open('resources/small_steps.csv')
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
                    aux[k][0] = int(aux[k][0])
                    aux[k][1] = int(aux[k][1])
                row[i] = aux 
        steps_list.append(row)
    return steps_list


def trajectory_interpol(precise_path):
    """
    Interpolates the trajectory between each point in the list with the precise path trough quintic polynomials

    Input: list with all the point coordinates that the car needs to travel to
    Output: list with interpolated trajectory (x, y), list with interpolated orientations (theta)
    """
    time_list = []
    trajectory_x = []
    trajectory_y = []
    velocity_list = []
    orientation_list = []

    if(len(precise_path) <= 2):

        sx = precise_path[0][0]  # start x position [m]
        sy = precise_path[0][1]  # start y position [m]
        syaw = atan2(precise_path[1][1]-precise_path[0][1], precise_path[1][0]-precise_path[0][0])  # start yaw angle [rad]
        sv = (5/0.05073825503)*1000/3600  # start speed [m/s]
        sa = 0  # start accel [m/ss]
        gx = precise_path[1][0]  # goal x position [m]
        gy = precise_path[1][1] # goal y position [m]
        gyaw = syaw # goal yaw angle [rad]
        gv = (0/0.05073825503)*1000/3600  # goal speed [m/s]
        ga = 0  # goal accel [m/ss]
        max_accel = 3.0/0.05073825503  # max accel [m/ss]
        max_jerk = 0.5/0.05073825503  # max jerk [m/sss]
        dt = 0.1   # time tick [s]
        time, x, y, yaw, v, a, j = quintic_polynomials_planner(sx, sy, syaw, sv, sa, gx, gy, gyaw, gv, ga, max_accel, max_jerk, dt)
        time_list.append(time)
        trajectory_x.append(x)
        trajectory_y.append(y)
        orientation_list.append(yaw)    
        velocity_list.append(v)
    else:

        sa = 0.01/0.05073825503   # start accel [m/ss]
        ga = 0.01/0.05073825503   # goal accel [m/ss]
        max_accel = 0.5/0.05073825503  # max accel [m/ss]
        max_jerk = 0.51/0.05073825503  # max jerk [m/sss]
        dt = 0.1   # time tick [s]

        
        for i in range(len(precise_path) - 1):
            if(i == (len(precise_path)-2)):
                sx = trajectory_x[-1][-1]  # start x position [m]
                sy = trajectory_y[-1][-1]   # start y position [m]
                syaw = atan2(precise_path[i+1][1]-trajectory_y[-1][-1], precise_path[i+1][0]-trajectory_x[-1][-1])  # start yaw angle [rad]
                syaw = orientation_list[-1][-1] # goal yaw angle [rad]
                sv = (1/0.05073825503)*1000/3600  # start speed [m/s]
                gx = precise_path[i+1][0]  # goal x position [m]
                gy = precise_path[i+1][1] # goal y position [m]
                gyaw = syaw  # goal yaw angle [rad]
                gv = (0/0.05073825503)*1000/3600  # goal speed [m/s]

                time, x, y, yaw, v, a, j = quintic_polynomials_planner(sx, sy, syaw, sv, sa, gx, gy, gyaw, gv, ga, max_accel, max_jerk, dt)
                trajectory_x.append(x[:-1])
                trajectory_y.append(y[:-1])
                orientation_list.append(yaw[:-1])  
                velocity_list.append(v[:-1])
            elif(i == 0):
                sx = precise_path[0][0]  # start x position [m]
                sy = precise_path[0][1]  # start y position [m]
                syaw = atan2(precise_path[1][1]-precise_path[0][1], precise_path[1][0]-precise_path[0][0])  # start yaw angle [rad]
                sv = 0  # start speed [m/s]
                gx = precise_path[1][0]  # goal x position [m]
                gy = precise_path[1][1] # goal y position [m] 
                ang1 = atan2(precise_path[2][1]-precise_path[1][1], precise_path[2][0]-precise_path[1][0]) # angle between the current point and the target point
                ang2 = atan2(precise_path[2-1][1]-precise_path[1-1][1], precise_path[2-1][0]-precise_path[1-1][0]) # angle between the next point and its' next point 
                gyaw = atan2((sin(ang2) + sin(ang1)),(cos(ang1) + cos(ang2))) # goal yaw angle [rad]
                gv = (1/0.05073825503)*1000/3600  # goal speed [m/s]
                
                time, x, y, yaw, v, a, j = quintic_polynomials_planner(sx, sy, syaw, sv, sa, gx, gy, gyaw, gv, ga, max_accel, max_jerk, dt)
                trajectory_x.append(x[:-1])
                trajectory_y.append(y[:-1])
                orientation_list.append(yaw[:-1])  
                velocity_list.append(v[:-1])

            #else:

            else:
                sx = trajectory_x[-1][-1]  # start x position [m]
                sy = trajectory_y[-1][-1]   # start y position [m]
                syaw = orientation_list[-1][-1] # start yaw angle [rad]
                sv = velocity_list[-1][-1]  # start speed [m/s] 
                gx = precise_path[i+1][0]  # goal x position [m]
                gy = precise_path[i+1][1] # goal y position [m]
                ang1 = atan2(precise_path[i+2][1]-precise_path[i+1][1], precise_path[i+2][0]-precise_path[i+1][0]) # angle between the current point and the target point
                ang2 = atan2(precise_path[i-1+2][1]-precise_path[i-1+1][1], precise_path[i-1+2][0]-precise_path[i-1+1][0]) # angle between the next point and its' next point 
                gyaw = atan2((sin(ang2) + sin(ang1)),(cos(ang1) + cos(ang2))) # goal yaw angle [rad]
                gv = (1/0.05073825503)*1000/3600  # goal speed [m/s]
                time, x, y, yaw, v, a, j = quintic_polynomials_planner(sx, sy, syaw, sv, sa, gx, gy, gyaw, gv, ga, max_accel, max_jerk, dt)
                trajectory_x.append(x[:-1])
                trajectory_y.append(y[:-1])
                orientation_list.append(yaw[:-1])  
                velocity_list.append(v[:-1])
                
    return [trajectory_x, trajectory_y], orientation_list



MAX_T = 100.0  # maximum time to the goal [s]
MIN_T = 3.0  # minimum time to the goal[s]
show_animation = False




class QuinticPolynomial:

    def __init__(self, xs, vxs, axs, xe, vxe, axe, time):
        # calc coefficient of quintic polynomial
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[time ** 3, time ** 4, time ** 5],
                      [3 * time ** 2, 4 * time ** 3, 5 * time ** 4],
                      [6 * time, 12 * time ** 2, 20 * time ** 3]])
        b = np.array([xe - self.a0 - self.a1 * time - self.a2 * time ** 2,
                      vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2

        return xt


def quintic_polynomials_planner(sx, sy, syaw, sv, sa, gx, gy, gyaw, gv, ga, max_accel, max_jerk, dt):
    """
    quintic polynomial planner
    input
        s_x: start x position [m]
        s_y: start y position [m]
        s_yaw: start yaw angle [rad]
        sa: start accel [m/ss]
        gx: goal x position [m]
        gy: goal y position [m]
        gyaw: goal yaw angle [rad]
        ga: goal accel [m/ss]
        max_accel: maximum accel [m/ss]
        max_jerk: maximum jerk [m/sss]
        dt: time tick [s]
    return
        time: time result
        rx: x position result list
        ry: y position result list
        ryaw: yaw angle result list
        rv: velocity result list
        ra: accel result list
    """

    vxs = sv * cos(syaw)
    vys = sv * sin(syaw)
    vxg = gv * cos(gyaw)
    vyg = gv * sin(gyaw)

    axs = sa * cos(syaw)
    ays = sa * sin(syaw)
    axg = ga * cos(gyaw)
    ayg = ga * sin(gyaw)

    time, rx, ry, ryaw, rv, ra, rj = [], [], [], [], [], [], []

    for T in np.arange(MIN_T, MAX_T, MIN_T):
        xqp = QuinticPolynomial(sx, vxs, axs, gx, vxg, axg, T)
        yqp = QuinticPolynomial(sy, vys, ays, gy, vyg, ayg, T)

        time, rx, ry, ryaw, rv, ra, rj = [], [], [], [], [], [], []

        for t in np.arange(0.0, T + dt, dt):
            time.append(t)
            rx.append(xqp.calc_point(t))
            ry.append(yqp.calc_point(t))

            vx = xqp.calc_first_derivative(t)
            vy = yqp.calc_first_derivative(t)
            v = np.hypot(vx, vy)

            yaw = atan2(vy, vx)
            rv.append(v)
            ryaw.append(yaw)

            ax = xqp.calc_second_derivative(t)
            ay = yqp.calc_second_derivative(t)
            a = np.hypot(ax, ay)
            if len(rv) >= 2 and rv[-1] - rv[-2] < 0.0:
                a *= -1
            ra.append(a)

            jx = xqp.calc_third_derivative(t)
            jy = yqp.calc_third_derivative(t)
            j = np.hypot(jx, jy)
            if len(ra) >= 2 and ra[-1] - ra[-2] < 0.0:
                j *= -1
            rj.append(j)

        if max([abs(i) for i in ra]) <= max_accel and max([abs(i) for i in rj]) <= max_jerk:
            break

    return time, rx, ry, ryaw, rv, ra, rj


def add_prev_and_next_node(path, xy_init, xy_end, area_list):
    """
    Find in which edge the chosen initial and end points are by determining which node came before or after, respectively, adding them to the path
    Input: Path (list of nodes in the path); coordinates of the initial and end points; list of the corners of each area
    Output: Extended path with the new auxiliary nodes
    """
    #find the area(s) the points are in
    areas_init = check_area(xy_init[0], xy_init[1], area_list)
    areas_end = check_area(xy_end[0], xy_end[1], area_list)

    complete_path = []

    for i in range(2):
        #i = 0: initial point
        #i = 1: end point
        if i == 0: areas = areas_init
        else: areas = areas_end

        if (i == 0 and path[1] == 0) or (i == 1 and path[-2] == 0):
            complete_path.append(1)

        elif (i == 0 and path[1] == 1) or (i == 1 and path[-2] == 1):
            if 'A' in areas:
                complete_path.append(0)
            else: 
                complete_path.append(6)

        elif i == 0 and path[1] == 2:
            complete_path.append(3)
        elif i == 1 and path[-2] == 2:
            if 'E' in areas and 'D' not in areas :
                complete_path.append(3)
            elif 'E' not in areas and 'D' in areas :
                complete_path.append(7)
            #else: intersection of areas -> just go to the next node  

        elif i == 0 and path[1] == 3:
            if 'E' in areas and 'F' not in areas and 'I' not in areas:
                complete_path.append(2)
            elif 'E' not in areas and 'F' in areas and 'I' not in areas :
                complete_path.append(4)
            elif 'E' not in areas and 'F' not in areas and 'I' in areas :
                complete_path.append(8)
            #else: intersection of areas -> just go to the next node
        elif i == 1 and path[-2] == 3:
            if 'E' in areas and 'F' not in areas:
                complete_path.append(2)
            elif 'E' not in areas and 'F' in areas:
                complete_path.append(4)

        elif (i == 0 and path[1] == 4) or (i == 1 and path[-2] == 4):
            if 'F' in areas:
                complete_path.append(3)
            else: complete_path.append(5)

        elif (i == 0 and path[1] == 5) or (i == 1 and path[-2] == 5):
            if 'G' in areas:
                complete_path.append(4)
            else: complete_path.append(10)

        elif i == 0 and path[1] == 6:
            if 'B' in areas:
                complete_path.append(1)
            else: complete_path.append(7)
        elif i == 1 and path[-2] == 6:
            if 'B' in areas and 'M' not in areas :
                complete_path.append(1)
            elif 'B' not in areas and 'M' in areas :
                complete_path.append(11)
            #else: intersection of areas -> just go to the next node

        elif i == 0 and path[1] == 7:
            if 'D' in areas and 'J' not in areas :
                complete_path.append(2)
            elif 'D' not in areas and 'J' in areas :
                complete_path.append(8)
            #else: intersection of areas -> just go to the next node
        elif i == 1 and path[-2] == 7:
            if 'C' in areas and 'K' not in areas :
                complete_path.append(6)
            elif 'C' not in areas and 'K' in areas :
                complete_path.append(9)
            #else: intersection of areas -> just go to the next node

        elif i == 0 and path[1] == 8:
            complete_path.append(9)
        elif i == 1 and path[-2] == 8:
            if 'I' in areas and 'J' not in areas :
                complete_path.append(3)
            elif 'I' not in areas and 'J' in areas :
                complete_path.append(7)
            #else: intersection of areas -> just go to the next node

        elif i == 0 and path[1] == 9:
            if 'K' in areas:
                complete_path.append(7)
            else: complete_path.append(15)
        elif i == 1 and path[-2] == 9:
            if 'L' in areas:
                complete_path.append(8)
            else: complete_path.append(15)

        elif (i == 0 and path[1] == 10) or (i == 1 and path[-2] == 10):
            complete_path.append(5)

        elif i == 0 and path[1] == 11:
            complete_path.append(6)
        elif i == 1 and path[-2] == 11:
            complete_path.append(12)

        elif i == 0 and path[1] == 12:
            complete_path.append(11)
        elif i == 1 and path[-2] == 12:
            if 'P' in areas and 'Q' not in areas :
                complete_path.append(13)
            elif 'P' not in areas and 'Q' in areas :
                complete_path.append(17)    
            #else: intersection of areas -> just go to the next node
            
        elif (i == 0 and path[1] == 13):
            if 'P' in areas and 'R' not in areas :
                complete_path.append(12)
            elif 'P' not in areas and 'R' in areas :
                complete_path.append(18)
            #else: intersection of areas -> just go to the next node
        elif i == 1 and path[-2] == 13:
            complete_path.append(14)
            
        elif i == 0 and path[1] == 14:
            if 'S' in areas and 'T' not in areas :
                complete_path.append(13)
            elif 'S' not in areas and 'T' in areas :
                complete_path.append(15)
            #else: intersection of areas -> just go to the next node
        elif i == 1 and path[-2] == 14:
            if 'T' in areas and 'U' not in areas :
                complete_path.append(15)
            elif 'T' not in areas and 'U' in areas :
                complete_path.append(19)
            #else: intersection of areas -> just go to the next node
            

        elif i == 0 and path[1] == 15:
            if 'O' in areas and 'T' not in areas and 'V' not in areas:
                complete_path.append(9)
            elif 'O' not in areas and 'T' in areas and 'V' not in areas:
                complete_path.append(14)
            elif 'O' not in areas and 'T' not in areas and 'V' in areas:
                complete_path.append(20)
            #else: intersection of areas -> just go to the next node
        elif i == 1 and path[-2] == 15:
            if 'O' in areas and 'T' not in areas :
                complete_path.append(9)
            elif 'O' not in areas and 'T' in areas :
                complete_path.append(14)
            #else: intersection of areas -> just go to the next node
            

        elif (i == 0 and path[1] == 16) or (i == 1 and path[-2] == 16):
            complete_path.append(21)

        elif i == 0 and path[1] == 17:
            complete_path.append(12)
        elif i == 1 and path[-2] == 17:
            complete_path.append(18) 
            
        elif i == 0 and path[1] == 18:
            complete_path.append(17)
        elif i == 1 and path[-2] == 18:
            complete_path.append(13) 
            
        elif i == 0 and path[1] == 19:
            if 'U' in areas and 'X' not in areas :
                complete_path.append(14)
            elif 'U' not in areas and 'X' in areas :
                complete_path.append(20)
            #else: intersection of areas -> just go to the next node
        elif i == 1 and path[-2] == 19:   
            complete_path.append(20)
        
        elif i == 0 and path[1] == 20:
            if 'X' in areas and 'Y' not in areas :
                complete_path.append(19)
            elif 'X' not in areas and 'Y' in areas :
                complete_path.append(21)
            #else: intersection of areas -> just go to the next node
        elif i == 1 and path[-2] == 20:
            if 'V' in areas and 'X' not in areas and 'Y' not in areas:
                complete_path.append(15)
            elif 'V' not in areas and 'X' in areas and 'Y' not in areas :
                complete_path.append(19)
            elif 'V' not in areas and 'X' not in areas and 'Y' in areas :
                complete_path.append(21)
            #else: intersection of areas -> just go to the next node
            
        elif (i == 0 and path[1] == 21) or (i == 1 and path[-2] == 21):
            if 'Y' in areas:
                complete_path.append(20)
            else: complete_path.append(16)

        if i == 0: complete_path = list(np.append(complete_path, path))
    return complete_path

def init_to_trajectory(complete_path, nodes_list, steps_list):
    """
    Find the "mini" nodes (steps) that lead from the chosen initial point to the first node by comparing the distances to it
    Input: Extended path (with the node previous from the initial point); list with the position of each node; list with the position of each step ("mini" nodes between 2 nodes) 
    Output: Path from the initial point to the first node
    """
    #Node before the intial point chosen
    prev_node = complete_path[0]
    #Node after the initial point chosen
    first_node = complete_path[2]
    #Position of the first node
    first_node_point = nodes_list[first_node]

    #Initial point chosen
    init_point = nodes_list[complete_path[1]]
    
    #Distance from the intial chosen node to the first node
    init_dist = calc_dist(init_point[0], init_point[1], first_node_point[0], first_node_point[1])

    #List of steps from the previous node to the next one
    small_steps = steps_list[prev_node][first_node]

    init_to_first_node = []
    init_to_first_node.append([init_point[0], init_point[1]])

    for step_point in small_steps:
        step_dist = calc_dist(step_point[0], step_point[1], first_node_point[0], first_node_point[1])
        #Add to the path only the points that are closer to the next node than the initial point
        if step_dist < init_dist: init_to_first_node.append([step_point[0],step_point[1]])
    return init_to_first_node

def end_from_trajectory(complete_path, nodes_list, steps_list):
    """
    Find the "mini" nodes (steps) that lead from the last node to the chosen end point by comparing the distances to it
    Input: Extended path (with the node after the end point); list with the position of each node; list with the position of each step ("mini" nodes between 2 nodes) 
    Output: Path from the final node to the end point
    """
    #Node after the end point chosen
    next_node = complete_path[-1]
    #Node before the end point chosen
    last_node = complete_path[-3]
    #Position of the last node
    last_node_point = nodes_list[last_node]

    #Chosen end point
    end_point = nodes_list[complete_path[-2]]
    
    #Distance from the final node to the chosen end point
    end_dist = calc_dist(end_point[0], end_point[1], last_node_point[0], last_node_point[1])

    #List of steps from the previous node to the next one
    small_steps = steps_list[last_node][next_node]

    end_from_last_node = []
    for step_point in small_steps:
        step_dist = calc_dist(step_point[0], step_point[1], last_node_point[0], last_node_point[1])
        #Add to the path only the points that are closer to the previous node than the initial point
        if step_dist < end_dist: end_from_last_node.append([step_point[0],step_point[1]])

    end_from_last_node.append([end_point[0], end_point[1]])
    return end_from_last_node

def gen_precise_path(path, steps_list, nodes_list, area_list): #rever esta |
    """
    Receives the path calculated by the Djikstra algorithm and improves it by adding intermediate targets between each node
    Input: Path (calculated by Djikstra); list with the position of each step ("mini" nodes between 2 nodes); list with the coordinates of key regions of the IST Campus map 
    Output: Precise path from the initial node to the end point
    """
    xy_init = nodes_list[-1]
    xy_end = nodes_list[-2]

    if len(path) <= 2: return [list(xy_init), list(xy_end)]

    complete_path = add_prev_and_next_node(path, xy_init, xy_end, area_list)

    precise_path = []
    for i in range(len(path)-1):
        if i == 0:
            if complete_path[0] == path[0]: precise_path.append(list(xy_init)) #if first node is the actual init node
            else: 
                first_nodes = init_to_trajectory(complete_path, nodes_list, steps_list)
                for node in first_nodes:
                    precise_path.append(node)
        elif(i == (len(path) - 2)): 
            if complete_path[-1] == path[-1]: precise_path.append(list(xy_end)) #if last node is the actual end node
            else: 
                last_nodes = end_from_trajectory(complete_path, nodes_list, steps_list)
                for node in last_nodes:
                    precise_path.append(node)
        else:
            for k in range(len(steps_list[path[i]][path[i+1]])):
                precise_path.append([steps_list[path[i]][path[i+1]][k][0], steps_list[path[i]][path[i+1]][k][1]])
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
    mapa_ist = plt.imread('resources/ist_map.png')
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    plt.imshow(mapa_ist)
    
    inputs = plt.ginput(2)
    
    x_init = int(round(inputs[0][0]))
    y_init = int(round(inputs[0][1]))
    x_end = int(round(inputs[1][0]))
    y_end = int(round(inputs[1][1]))

    plt.scatter([x_init, x_end], [y_init, y_end], c = "r", marker = "+")

    only_street_mat = plt.imread("resources/ist_only_streets.png")

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

    nodes_graph, aux = add_node(x_init, y_init, nodes_graph, area_list, final_areas, [x_end, y_end])
    if(nodes_graph == None):
        nodes_graph = []
        plt.clf()
        print("Invalid input (outside of valid area)!")
        continue
    
    valid_points = True

path = g.dijkstra(nodes_graph, n_nodes+1, n_nodes)

nodes_list.append((x_end, y_end))
nodes_list.append((x_init, y_init))

precise_path = gen_precise_path(path, steps_list, nodes_list, area_list)

vetor_trajectories, orientation_list = trajectory_interpol(precise_path)
xx = []
yy = []
orientation = []
for i in vetor_trajectories[0][:]:
    for j in i:
        xx.append(int(j))
        
for i in vetor_trajectories[1][:]:
    for j in i:
        yy.append(int(j))
        
for i in orientation_list[:]:
    for j in i:
        orientation.append(float(j))
            
for i in range(len(xx), 0, -1):
    if(i%5 != 0):
        del xx[i]
        del yy[i]
        del orientation[i]

plt.imshow(mapa_ist)
num = int(calc_dist(precise_path[0][0], precise_path[0][1], precise_path[1][0], precise_path[1][1])/20)

plt.scatter(xx, yy, s = 10)


plt.savefig('trajectory.pdf', bbox_inches='tight')
plt.show()


fout = open('trajectory_points.csv', 'w', newline='')
writer = csv.writer(fout)
for i in range(len(xx)):
    if(i == (len(xx)-1)):
        writer.writerow([int(xx[i]), int(yy[i]), orientation[i]])
    else:
        writer.writerow([int(xx[i]), int(yy[i]), orientation[i]])

fout.close()
