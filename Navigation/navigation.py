from time import sleep
import pygame 
import math
import numpy as np
from pygame.locals import *
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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

        #print(i)

        #print(s_colors)


        for j in range(0, 5):
            if s_colors[j] != (242, 243, 245,255) and hit[j] == 0:
                sensor[j] = i
                hit[j] = 1
                #print(s_colors[j])
                

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


    print(delta_x)
    print(delta_y)


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


def check_colision(car_size, car_pos,angle, corner_angle, background):
    """
    Function that detects if the car has droven outside of the road 
    inputs: dimension of the car , position of the car , orientation angle of the car the angle from the center of the car to its edges
    output: colision value 1 if the car as droven out of the road , 0 otherwise
    
    """

    colision = 0


    corner_top_right =  background.get_at( (car_pos[0] + int(car_size[0]//2 * math.cos(corner_angle + angle)) , car_pos[1] + int(car_size[1]//2 * math.sin(corner_angle + angle)) ))
    corner_top_left =  background.get_at( (car_pos[0] - int(car_size[0]//2 * math.cos(corner_angle + angle)), car_pos[1] + int(car_size[1]//2 * math.sin(corner_angle + angle)) ))
    corner_bot_right =  background.get_at( (car_pos[0] + int(car_size[0]//2 * math.cos(corner_angle + angle)) , car_pos[1] - int(car_size[1]//2 * math.sin(corner_angle + angle)) ))
    corner_bot_left =  background.get_at( (car_pos[0] - int(car_size[0]//2 * math.cos(corner_angle + angle)) , car_pos[1] - int(car_size[1]//2 * math.sin(corner_angle + angle)) ))


    pygame.draw.circle(screen,blue,(screen_center[0] - (car_size[0]//2 * math.cos(corner_angle + angle)), screen_center[1] + (car_size[1]//2 * math.sin(corner_angle + angle)) ), 5)
    pygame.draw.circle(screen,blue,(screen_center[0] + (car_size[0]//2 * math.cos(corner_angle - angle)), screen_center[1] + (car_size[1]//2 * math.sin(corner_angle - angle)) ), 5)
    pygame.draw.circle(screen,blue,(screen_center[0] + (car_size[0]//2 * math.cos(corner_angle + angle)) , screen_center[1] - (car_size[1]//2 * math.sin(corner_angle + angle))), 5)
    pygame.draw.circle(screen,blue,(screen_center[0] - (car_size[0]//2 * math.cos(corner_angle - angle)) , screen_center[1] - (car_size[1]//2 * math.sin(corner_angle - angle)) ), 5)

   


    if corner_top_right[0] < 250 or corner_top_right[1] < 250 or corner_top_right[2] < 250:
        pygame.draw.circle(screen,red,(screen_center[0] + (car_size[0]//2 * math.cos(corner_angle - angle)) , screen_center[1] + (car_size[1]//2 * math.sin(corner_angle - angle))), 5)
        #print(corner_top_right)
        colision = 1
    
    if corner_top_left[0] < 250 or corner_top_left[1] < 250 or corner_top_left[2] < 250:
        pygame.draw.circle(screen,red,(screen_center[0] -  (car_size[0]//2 * math.cos(corner_angle + angle)) , screen_center[1] + (car_size[1]//2 * math.sin(corner_angle + angle))), 5)
        #print(corner_top_left)
        colision = 1

    if corner_bot_left[0] < 250 or corner_bot_left[1] < 250 or corner_bot_left[2] < 250:
        pygame.draw.circle(screen,red,(screen_center[0] + (car_size[0]//2 * math.cos(corner_angle + angle)), screen_center[1] - (car_size[1]//2 * math.sin(corner_angle + angle)) ), 5)
        #print(corner_bot_left)
        colision = 1

    if corner_bot_right[0] < 250 or corner_bot_right[1] < 250 or corner_bot_right[2] < 250:
        pygame.draw.circle(screen,red,(screen_center[0] - (car_size[0]//2 * math.cos(corner_angle - angle)), screen_center[1] - (car_size[1]//2 * math.sin(corner_angle - angle)) ), 5)
        #print(corner_bot_right)
        colision = 1


    

    return colision

def car_model(v, w, theta, L, SR):
    """Function that simulates de real life motion of a car given its velocity and stearing velocity
    Input: car velocity , car's stearing velocity ,cars position, cars orientation angle (theta), car wheal base distance 
    Output : variation in X , variation in Y , variation in orientation (THETA) """

    if v < 0.00001 and v > -0.00001:
        phi = 0
    else:
        phi = math.asin((w/v)*L) 

    delta_x = math.cos(theta)*math.cos(phi)*v
    delta_y = math.sin(theta)*math.cos(phi)*v
    delta_theta = (180 * (math.sin(phi)/L) * v)/math.pi
    delta_phi = w

    return delta_x * SR, delta_y  * SR, delta_theta * SR


def GPS(pos, angle):
    """"Function that simulates the uncertainty of a gps system 
    Input: car's position  
    Output: car's position with associated uncertainty """

    new_pos_x = pos[0] + np.random.normal(0, 3 + (7*np.sin(angle)))
    new_pos_y = pos[1] + np.random.normal(0, 3 + (7*np.cos(angle)))

    return (new_pos_x,new_pos_y)


pygame.init()


points =((1083,257),
         (2511,205),
         (2681,191),
         (3732,129),
         (4193,191),
         (1307,1718),
         (2625,1626),
         (2805,1606),
         (2731,1808),
         (4385,2016),
         (1391,3134),
         (1391,3946),
         (1611,3928),
         (2761,3852),
         (2959,3826),
         (4491,3169),
         (1449,4876),
         (1679,4860),
         (2893,5279),
         (3059,5269),
         (4585,5014))

screen = pygame.display.set_mode((1000,800))

car_image = "mclarinho_50px.png"
background_image = "Tecnico_high_res_og_lowres_forreal-f383eb38-795c-11ec-a2bb-5e02152dd6df.png"
trajectory_file = "../guidance/trajectory_points.csv"


background = pygame.image.load(background_image)

car = pygame.image.load(car_image)


car_size = car.get_size()
bg_size = background.get_size()
screen_size = screen.get_size()

screen_center = (screen_size[0]//2,screen_size[1]//2)

position = points[12]

#print(bg_size)
bg_x = -(position[0] - (screen_size[0]//2))
bg_y = -(position[1] - (screen_size[1]//2))


#lista_pontos =generate_motion(points[12],points[13])

input_file = open(trajectory_file)

csv_file = csv.reader(input_file)

sleep(1)



csv_list = []

for row in csv_file:
        for i in range(len(row)):
            row[i] = float(row[i])
        csv_list.append(row)
    
intended_trajectory = csv_list

running = True

angle = 0
corner_angle = math.atan2(car_size[1]/2, car_size[0]/2)

#____________RGB colors______________

red = (255,0,0)
green =(0,255,0)
blue = (0,0,255)

#____________________________________


counter_point = 1
next_point = 0
yikes = 0

clock = pygame.time.Clock()

trajectory_made = []

colision_counter = 0
last_colision_val = 0
colisions = 0
gps_status = 1

w = 0
v = 0
fps = 25

sample_rate = 1/fps

while running: 

    #fps
    clock.tick(fps)

    bg_x = -(int(position[0]) - (screen_size[0]//2))
    bg_y = -(int(position[1]) - (screen_size[1]//2)) 

    car_pos = (-bg_x + screen_size[0]//2 , -bg_y + screen_size[1]//2 )

    screen.blit(background,(bg_x,bg_y))

    #sensores = sensor((-bg_x + screen_size[0]//2 , -bg_y + screen_size[1]//2 ),angle ,background)

    colisions = check_colision(car_size ,(-bg_x + screen_size[0]//2 , -bg_y + screen_size[1]//2 ),(math.pi * angle)/180, corner_angle, background)

    if colisions != last_colision_val:
        colision_counter += colisions
        last_colision_val = colisions
    
    car_copy = pygame.transform.rotate(car, angle)
    car_size = car_copy.get_size()

    screen.blit(car_copy,(screen_center[0] - car_size[0]//2,screen_center[1] - car_size[1]//2 ))
    keys = pygame.key.get_pressed()

    
    """#MODO: TECLAS

    delta_x , delta_y , new_angle = car_model(v,w, -(math.pi * angle)/180, 20, sample_rate)

    position = (position[0] + (delta_x ), position[1] + (delta_y ))


    angle -= new_angle

    keys = pygame.key.get_pressed()

    if keys[pygame.K_RIGHT]:
            w += 0.1
    if keys[pygame.K_LEFT]:
            w -= 0.1
     
    if keys[pygame.K_UP]:
            v += 10

    if keys[pygame.K_DOWN]:
            v -= 10"""
    
    
    
    """
    #MODO: RETAS

    position = lista_pontos[next_point]
    if next_point + 1 < len(lista_pontos):
        next_point+=1
    else:
        print(colision_counter)
    """

    #MODO: ONLY GUIDANCE
    position = csv_list[next_point]
    if next_point + 1 < len(csv_list):
        next_point+=1

    
    """
    GPS

    intended_gps_value = (-bg_x + screen_size[0]//2 , -bg_y + screen_size[1]//2 )

    gps_x = 0
    gps_y = 0

    for _ in range(0, 100) :

        X_x , Y_y = GPS((-bg_x + screen_size[0]//2 , -bg_y + screen_size[1]//2 ), (math.pi * angle)/180)

        gps_x += X_x
        gps_y += Y_y

    print("-----------------")
    print("intended_gps_value")
    print(intended_gps_value)
    print("gps")
    print((gps_x//100, gps_y//100))
    print("single measurment")
    print((X_x,Y_y))
    print("-----------------")

    """

    if next_point > 2:
        angle = -(180/np.pi)* position[2]

    trajectory_made.append((position[0],position[1]))

    pygame.display.flip() 

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

print("Number of colisions: " + str(colision_counter))

plt.rcParams["figure.figsize"] = (7,8)

trajectory_made_x = [x for x,y in trajectory_made ]
trajectory_made_y = [y for x,y in trajectory_made ]

intended_trajectory_x = [x for x,y,z in intended_trajectory ]
intended_trajectory_y = [y for x,y,z in intended_trajectory]

mapa = mpimg.imread(background_image)
plt.imshow(mapa)


plt.plot(intended_trajectory_x,intended_trajectory_y)
plt.plot(trajectory_made_x,trajectory_made_y)
plt.show()
