# Robotics lab 2, Navigation
"""
    Authors:
    - Goncalo Teixeira, 93068
    - Goncalo Fernandes, 93070
    - Pedro Martins, 93153
    Date last modified: 31/01/2022
"""


from distutils.ccompiler import gen_preprocess_options
from pickle import FALSE, TRUE
from time import sleep
import pygame 
import math
import numpy as np
from pygame.locals import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from control import read_csv_file, control_it
from guidance import get_trajectory


def sensor(car_pos, car_direction, background):

    s_colors = [[],[],[],[],[]]
    sensor=[0,0,0,0,0]

    hit = [0,0,0,0,0]

    for i in range(0 , 100, 2):

        s_colors[0] = background.get_at( (car_pos[0] + int(i * math.cos(car_direction - 90)) , car_pos[1] + int(i * math.cos(car_direction - 90)) ))
        s_colors[1] = background.get_at( (car_pos[0] + int(i * math.cos(car_direction - 45)) , car_pos[1] + int(i * math.cos(car_direction - 45)) ))
        s_colors[2] = background.get_at( (car_pos[0] + int(i * math.cos(car_direction     )) , car_pos[1] + int(i * math.cos(car_direction     )) ))
        s_colors[3] = background.get_at( (car_pos[0] + int(i * math.cos(car_direction + 45)) , car_pos[1] + int(i * math.cos(car_direction + 45)) ))
        s_colors[4] = background.get_at( (car_pos[0] + int(i * math.cos(car_direction + 90)) , car_pos[1] + int(i * math.cos(car_direction + 90)) ))


        for j in range(0, 5):
            if s_colors[j] != (242, 243, 245,255) and hit[j] == 0:
                sensor[j] = i
                hit[j] = 1
                
    return sensor

def generate_motion(point_1, point_2):
    """Function interpolates positions between 2 points"""

    min_step = 2

    positions = []
    scaling = 0

    x_signal = 1
    y_signal = 1

    x_p1 , y_p1 = point_1
    x_p2 , y_p2 = point_2

    delta_x = x_p2 - x_p1
    delta_y = y_p2 - y_p1

    steps_x = abs(delta_x // min_step)
    steps_y = abs(delta_y // min_step)

    if delta_x < 0:
        x_signal = -1
    
    if delta_y < 0:
        y_signal = -1

    positions.append( point_1)

    if abs(delta_x) > abs(delta_y) :

        for i in range(1, steps_x):

            angle = int(180 * -math.atan2((delta_y),(delta_x))/ math.pi)

            positions.append((positions[i-1][0] + (min_step * x_signal) ,positions[i-1][1]+ (min_step*scaling * y_signal), angle))

            if scaling == 1 :
                scaling = 0
            if i%(delta_x//delta_y) == 0:
                scaling = 1
                
    
    elif abs(delta_x) < abs(delta_y) :

        for i in range(1, steps_y):
            
            angle = int(180 * -math.atan2((delta_y),(delta_x)) / math.pi)

            positions.append((positions[i-1][0] + (min_step * scaling * x_signal) ,positions[i-1][1]+ (min_step * y_signal), angle))

            if scaling == 1 :
                scaling = 0
            if i%(delta_y//delta_x) == 0:\
                scaling = 1

    elif abs(delta_x) == abs(delta_y) :

        for i in range(0, steps_y):
            angle = int(180 * math.atan2((delta_y),(delta_x)) / math.pi)

            positions.append((positions[i-1][0] + (min_step * x_signal) ,positions[i-1][1]+ (min_step * y_signal), angle))

    return positions


def check_colision(car_size, car_pos,angle, corner_angle, collision_im):
    """
    Function that detects if the car has droven outside of the road 
    inputs: dimension of the car , position of the car , orientation angle of the car the angle from the center of the car to its edges
    output: colision value 1 if the car as droven out of the road , 0 otherwise
    
    """

    collision = 0
    corner = 0

    corner_top_right =  collision_im.get_at( (car_pos[0] + int(car_size[0]//2 * math.cos(corner_angle - angle)) , car_pos[1] + int(car_size[1]//2 * math.sin(corner_angle - angle)) ))
    corner_top_left =  collision_im.get_at( (car_pos[0] + int(car_size[0]//2 * math.cos(corner_angle + angle)), car_pos[1] - int(car_size[1]//2 * math.sin(corner_angle + angle)) ))
    corner_bot_right =  collision_im.get_at( (car_pos[0] - int(car_size[0]//2 * math.cos(corner_angle + angle)) , car_pos[1] + int(car_size[1]//2 * math.sin(corner_angle + angle)) ))
    corner_bot_left =  collision_im.get_at( (car_pos[0] - int(car_size[0]//2 * math.cos(corner_angle - angle)) , car_pos[1] - int(car_size[1]//2 * math.sin(corner_angle - angle)) ))

    #top right
    pygame.draw.circle(screen,blue,(screen_center[0] + (car_size[0]//2 * math.cos(corner_angle - angle)), screen_center[1] + (car_size[1]//2 * math.sin(corner_angle - angle)) ), 5)
     #top left
    pygame.draw.circle(screen,blue,(screen_center[0] + (car_size[0]//2 * math.cos(corner_angle + angle)) , screen_center[1] - (car_size[1]//2 * math.sin(corner_angle + angle))), 5)
    #bot right
    pygame.draw.circle(screen,blue,(screen_center[0] - (car_size[0]//2 * math.cos(corner_angle + angle)), screen_center[1] + (car_size[1]//2 * math.sin(corner_angle + angle)) ), 5)
    #bot left
    pygame.draw.circle(screen,blue,(screen_center[0] - (car_size[0]//2 * math.cos(corner_angle - angle)) , screen_center[1] - (car_size[1]//2 * math.sin(corner_angle - angle)) ), 5)
    

    if corner_top_right[0] == 255 and corner_top_right[1] == 255 and corner_top_right[2] == 255:
        pygame.draw.circle(screen,red,(screen_center[0] + (car_size[0]//2 * math.cos(corner_angle - angle)), screen_center[1] + (car_size[1]//2 * math.sin(corner_angle - angle)) ), 5)
        collision = 1
        corner = 1
    
    elif corner_top_left[0] == 255 and corner_top_left[1] == 255 and corner_top_left[2] == 255:
        pygame.draw.circle(screen,red,(screen_center[0] + (car_size[0]//2 * math.cos(corner_angle + angle)) , screen_center[1] - (car_size[1]//2 * math.sin(corner_angle + angle))), 5)
        collision = 1
        corner = 2

    elif corner_bot_right[0] == 255 and corner_bot_right[1] == 255 and corner_bot_right[2] == 255:
        pygame.draw.circle(screen,red,(screen_center[0] - (car_size[0]//2 * math.cos(corner_angle + angle)), screen_center[1] + (car_size[1]//2 * math.sin(corner_angle + angle)) ), 5)
        collision = 1
        corner = 3
    
    elif corner_bot_left[0] == 255 and corner_bot_left[1] == 255 and corner_bot_left[2] == 255:
        pygame.draw.circle(screen,red,(screen_center[0] - (car_size[0]//2 * math.cos(corner_angle - angle)) , screen_center[1] - (car_size[1]//2 * math.sin(corner_angle - angle)) ), 5)
        collision = 1
        corner = 4

    return collision, corner

def car_model(v, w, theta, phi, L, SR):
    """Function that simulates de real life motion of a car given its velocity and stearing velocity
    Input: car velocity , car's stearing velocity ,cars position, cars orientation angle (theta), car wheal base distance 
    Output : variation in X , variation in Y , variation in orientation (THETA) """
    delta = np.dot(np.array([[np.cos(theta), 0], [np.sin(theta), 0 ],[np.tan(phi)/L , 0], [0, 1]]), np.transpose(np.array([v, w])))
    delta_x = SR*delta[0]
    delta_y =  SR*delta[1]
    delta_theta = SR*delta[2]

    return delta_x, delta_y, delta_theta


def GPS(pos, angle):
    """"Function that simulates the uncertainty of a gps system 
    Input: car's position  
    Output: car's position with associated uncertainty """
    new_pos_x = pos[0] + np.random.normal(0, np.abs(0.3 + (0.7*np.sin(angle))))
    new_pos_y = pos[1] + np.random.normal(0, np.abs(0.3 + (0.7*np.cos(angle))))

    return (new_pos_x,new_pos_y)


#################### MAIN ######################
Energy = float(input("Initial Energy (Wh): "))
Energy_spent = 0

get_trajectory(show_trajectory = False)
pygame.init()

screen = pygame.display.set_mode((1000,800))

car_image = "resources/red_car.png"
background_image = "resources/ist_map.png"
collisions_image = "resources/ist_only_streets.png"
trajectory_file = "trajectory_points.csv"
GPS_on_image = "resources/GPS_on.png"
GPS_off_image = "resources/GPS_off.png"
bar_3 = "resources/bars_3.png"
bar_2 = "resources/bars_2.png"
bar_1 = "resources/bars_1.png"
bar_0 = "resources/bars_0.png"

background = pygame.image.load(background_image)
collision_im = pygame.image.load(collisions_image)
car = pygame.image.load(car_image)
gps_indicator_on = pygame.image.load(GPS_on_image)
gps_indicator_off = pygame.image.load(GPS_off_image)
bars_3 = pygame.image.load(bar_3)
bars_2 = pygame.image.load(bar_2)
bars_1 = pygame.image.load(bar_1)
bars_0 = pygame.image.load(bar_0)


car_size = car.get_size()
bg_size = background.get_size()
screen_size = screen.get_size()

screen_center = (screen_size[0]//2,screen_size[1]//2)

intended_trajectory = read_csv_file(trajectory_file)

position = intended_trajectory[0]

bg_x = -(position[0] - (screen_size[0]//2))
bg_y = -(position[1] - (screen_size[1]//2))

running = True

corner_angle = math.atan2(car_size[1]/2, car_size[0]/2)

#____________RGB colors______________
red = (255,0,0)
green =(0,255,0)
blue = (0,0,255)
#____________________________________

counter_point = 1
next_point = 0

clock = pygame.time.Clock()

trajectory_made = []
colision_points = []

colision_counter = 0
colision_corner = 0
last_colision_val = 0
gps_status = 1

w = 0
vel = 0
fps = 60

sample_rate = 6/fps

M = 810 

############################## NEW VARIABLES #########################################
xref = np.array(intended_trajectory)[:, 0]
yref = np.array(intended_trajectory)[:, 1]
theta_ref = np.array(intended_trajectory)[:, 2]


xref = np.array(xref)*5.073825503/100
yref = np.array(yref)*5.073825503/100
theta_ref = np.array(theta_ref)
angle = -theta_ref[0]*180/np.pi


L = 2.2
#h - periodo de sampling
h = 0.1 
#kv - constante de erro em x controla a velocidade
Kv = 0.03 * 16
#ks - constante de erro de orientcao (theta)
Ks = 100
#ki - constante de erro em y 
Ki = 8
#cenas para o filtro
time = []
theta = []
phi = []
x = []
y = []

# Setting standard filter requirements.
ws_filtered = []
v_filtered = []


time.append(0)
theta.append(theta_ref[0])
phi.append(0)
#we - erros no frame do mundo
we = np.zeros(3)
x.append(xref[0])
y.append(yref[0])
v_no_filter = []
ws_no_filter = []
ws = []
v = []
i = 1
j = 0
k = 0
counter=0
######################################################################################

#________________ GPS DEAD ZONES ______________#


GPS_dead_1_x1 = 2330
GPS_dead_1_y1 = 500
GPS_dead_1_x2 = 2700
GPS_dead_1_y2 = 1400

GPS_dead_2_x1 = 1870
GPS_dead_2_y1 = 3800
GPS_dead_2_x2 = 2600
GPS_dead_2_y2 = 4000


while running and j < len(intended_trajectory) - 3: 

    #fps
    clock.tick(fps)
    bg_x = -(int(position[0]) - (screen_size[0]//2))
    bg_y = -(int(position[1]) - (screen_size[1]//2)) 

    car_pos = (-bg_x + screen_size[0]//2 , -bg_y + screen_size[1]//2 )

    screen.fill((255, 255, 255))
    screen.blit(background,(bg_x,bg_y))

    colisions, colision_corner = check_colision(car_size ,(-bg_x + screen_size[0]//2 , -bg_y + screen_size[1]//2 ),(math.pi * angle)/180, corner_angle, collision_im)

    if colisions != last_colision_val:
        colision_counter += colisions
        
        if colisions == 1 :
            if colision_corner == 1 :
                colision_points.append((position[0] + int(car_size[0]//2 * math.cos(corner_angle - angle)) ,position[1] + int(car_size[0]//2 * math.sin(corner_angle - angle))))
            if colision_corner == 2 :
                colision_points.append((position[0] + int(car_size[0]//2 * math.cos(corner_angle + angle)), position[1]  - int(car_size[0]//2 * math.sin(corner_angle + angle))))
            if colision_corner == 3 :
                colision_points.append((position[0] - int(car_size[0]//2 * math.cos(corner_angle + angle)),position[1]  + int(car_size[0]//2 * math.sin(corner_angle + angle))))
            if colision_corner == 4 :
                colision_points.append((position[0] - int(car_size[0]//2 * math.cos(corner_angle - angle)),position[1]  - int(car_size[0]//2 * math.sin(corner_angle - angle))))

        last_colision_val = colisions

    
    if gps_status == True:
        gps_x = 0
        gps_y = 0
        for _ in range(0, 500) :
            X_x , Y_y = GPS((x[i-1], y[i-1]), angle)
            gps_x += X_x
            gps_y += Y_y

        x[i-1] = gps_x/500
        y[i-1] = gps_y/500

        vel, w, j, new_phi, new_theta = control_it(i, j, time, xref, yref, theta_ref, x, y, intended_trajectory, we, theta, v_no_filter, v, Kv, ws_no_filter, Ks, Ki, ws, h, phi, L, counter,  ws_filtered,	v_filtered)
    else:
        vel, w, j, new_phi, new_theta = control_it(i, j, time, xref, yref, theta_ref, x, y, intended_trajectory, we, theta, v_no_filter, v, Kv, ws_no_filter, Ks, Ki, ws, h, phi, L, counter,  ws_filtered, v_filtered)

    delta_x , delta_y , delta_theta = car_model(vel ,w, new_theta, new_phi, 22, sample_rate)
    
    delta_x = delta_x/(5.073825503/100)
    delta_y = delta_y/(5.073825503/100)
    position = (position[0] + (delta_x), position[1] + (delta_y))
    delta_theta = delta_theta * 180 /np.pi
    
    angle -= delta_theta / sample_rate  

    car_copy = pygame.transform.rotate(car, angle)
    car_size = car_copy.get_size()

    screen.blit(car_copy,(screen_center[0] - car_size[0]//2,screen_center[1] - car_size[1]//2 ))

    if (position[0] > GPS_dead_1_x1 and position[0] < GPS_dead_1_x2) and (position[1] >  GPS_dead_1_y1 and position[1] < GPS_dead_1_y2):
        
        screen.blit(gps_indicator_off, (0,0))
        gps_status = False
    elif (position[0] > GPS_dead_2_x1 and position[0] < GPS_dead_2_x2) and (position[1] >  GPS_dead_2_y1 and position[1] < GPS_dead_2_y2):
        
        screen.blit(gps_indicator_off, (0,0))
        gps_status = False
    else :
        screen.blit(gps_indicator_on, (0,0))
        gps_status = True

    trajectory_made.append((position[0],position[1]))
    
    
    delta_v = (v[i-1] - v[i-2])/sample_rate
    P_0 = 0
    delta_e = (M*abs(delta_v)*abs(v[i-1]) + P_0)*sample_rate/(3600) #Wh
    Energy_spent += delta_e
    if i%10 == 0:
        print("Energy Spent(Wh): " + str(round(float(Energy_spent),2)) + "/" + str(Energy))
    
    if Energy_spent < 1/3*Energy:
        screen.blit(bars_3, (100,0))
    elif Energy_spent < 2/3*Energy:
        screen.blit(bars_2, (100,0))
    elif Energy_spent < Energy:
        screen.blit(bars_1, (100,0))
    elif round(Energy_spent,2) == Energy:
        screen.blit(bars_0, (100,0))
    elif Energy_spent > Energy:
        screen.blit(bars_0, (100,0))
        print("Out of Battery!")
        break
    
    pygame.display.flip() 

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    i+=1



print("Number of colisions: " + str(colision_counter))

plt.rcParams["figure.figsize"] = (7,8)

trajectory_made_x = [x for x,y in trajectory_made ]
trajectory_made_y = [y for x,y in trajectory_made ]

intended_trajectory_x = [x for x,y,z in intended_trajectory]
intended_trajectory_y = [y for x,y,z in intended_trajectory]
colision_points_x = [x for x,y in colision_points]
colision_points_y = [y for x,y in colision_points]

mapa = mpimg.imread(background_image)
plt.imshow(mapa)


plt.plot(intended_trajectory_x,intended_trajectory_y)
plt.plot(trajectory_made_x,trajectory_made_y)
plt.scatter(colision_points_x,colision_points_y,c = "r", marker="*")
plt.show()
