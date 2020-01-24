


import keras
from keras.models import Sequential
from keras.layers import Dense
import math
import random
import numpy as np
import os
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



def get_min_angle(enemy_distance):
    r = math.sqrt(enemy_distance**2+40**2)
    alfa = math.asin(40.0/r)
    return alfa


def curve(v):
    y = v[0]*math.tan(v[1]) - (9.806/2) * (v[0]**2/(v[2]**2)*math.cos(v[1])**2)
    return y

def get_best_velocity(distance,min_alfa):
    min_v = 0.1
    zero_val = curve([distance,min_alfa,min_v])
    fit = abs( 55.0 - zero_val)
    for v in range(1,500):
        tmp_val = curve([distance,min_alfa,v])
        tmp_fit = abs(55.0 -  tmp_val)
        if tmp_fit < fit:
            min_v = v
            fit = tmp_fit
    
    tmp_min_v = min_v
    for i in range(-2000,2000):
        v = min_v + i/1000.0
        tmp_val = curve([distance,min_alfa,v])
        tmp_fit = abs(55.0 -  tmp_val)
        if tmp_fit < fit:
            tmp_min_v = v
            fit = tmp_fit

    return tmp_min_v


def find_distance(angle, velocity):
    min_d = 0.1
    zero_val = curve([min_d,angle,velocity])
    fit = abs( 55.0 - zero_val)
    for d in range(300,18000):
        tmp_val = curve([d,angle,velocity])
        tmp_fit = abs(55.0 -  tmp_val)
        if tmp_fit < fit:
            min_d = d
            fit = tmp_fit
    
    tmp_min_d = min_d
    for i in range(-2000,2000):
        d = min_d + i/1000.0
        tmp_val = curve([min_d,angle,velocity])
        tmp_fit = abs(55.0 -  tmp_val)
        if tmp_fit < fit:
            tmp_min_d = d
            fit = tmp_fit
    return tmp_min_d

def evaluate_data(angle,velocity,player_distance,enemy_distance):
    distance = find_distance(angle,velocity)
    best_dist = find_distance(get_min_angle(enemy_distance),get_best_velocity(player_distance,get_min_angle(enemy_distance)))
    if angle < get_min_angle(enemy_distance):
        # print('angle error')
        return False
    if abs(distance-best_dist) > 45:
        # print('distance error ->',abs(distance - best_dist))
        return False
    return True

def prepare_eval_data():
    ev_in_range = [675,1800]
    ev_in_enemy = [300]
    for i in range (7,18):
        ev_in_range.append(i*100)
        ev_in_range.append(i*100+25)
        ev_in_range.append(i*100+50)
        ev_in_range.append(i*100+75)

    
    for i in range (1,2):
        ev_in_enemy.append(i*100)
        ev_in_enemy.append(i*100+25)
        ev_in_enemy.append(i*100+50)
        ev_in_enemy.append(i*100+75)

    return ev_in_range,ev_in_enemy

def prepare_model_eval_data():
    temp = [[675,300]]
    temp.append([1800,300])
    # ev_in_range = [675,1800]
    # ev_in_enemy = [300]
    for i in range (7,18):
        v1 = i*100
        for j in range (1,2):
            temp.append([v1,j*100])
            temp.append([v1,j*100+25])
            temp.append([v1,j*100+50])
            temp.append([v1,j*100+75])
        
        v1 = i*100 + 25
        for j in range (1,2):
            temp.append([v1,j*100])
            temp.append([v1,j*100+25])
            temp.append([v1,j*100+50])
            temp.append([v1,j*100+75])
            
        v1 = i*100 + 50
        for j in range (1,2):
            temp.append([v1,j*100])
            temp.append([v1,j*100+25])
            temp.append([v1,j*100+50])
            temp.append([v1,j*100+75])
            
        v1 = i*100 + 75
        for j in range (1,2):
            temp.append([v1,j*100])
            temp.append([v1,j*100+25])
            temp.append([v1,j*100+50])
            temp.append([v1,j*100+75])

    return temp

def prepare_eval_out(values):
    temp = []
    for pair in values:
        angle = get_min_angle(pair[1])
        vel = get_best_velocity(pair[0],angle+0.1)
        temp.append([vel,angle])
    return temp
def load_model(path):
    return tf.keras.models.load_model(path)



model_eval = prepare_model_eval_data()
model_eval_out = prepare_eval_out(model_eval)
err = 0
model = load_model('model_CURRENT_BEST.h5')         #   3.4%
# keras.metrics.
# model.metrics.
model.evaluate(x=np.matrix(model_eval),y=np.matrix(model_eval_out))
