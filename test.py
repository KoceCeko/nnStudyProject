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
    y = v[0]*math.tan(v[1]) - (980.6/2) * (v[0]**2/(v[2]**2)*math.cos(v[1])**2)
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
    if abs(distance-best_dist) > 45.72/2 - 12:
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

def load_model(path):
    return tf.keras.models.load_model(path)



ev_range,ev_enemy = prepare_eval_data()

print(curve([700,0.485398,320]))

err = 0
model = load_model('model_CURRENT_BEST.h5')         #   3.4%
for val1 in ev_range :
    for val2 in ev_enemy:
        predictions = model.predict(np.matrix([val1,val2]))
        if not evaluate_data(predictions[0][1],predictions[0][0],val1,val2):
             # print('error -> ',val1,val2)
            err = err + 1
print('error % ->',err/(len(ev_range)*len(ev_enemy)))
tmp = err/(len(ev_range)*len(ev_enemy))

best = tmp
print('CURRENT BEST ->',best)
err = 0
session = 1
while False:
    print('session',session)
    session=session+1
    models = []
    # models.append(load_model('model_CURRENT_BEST.h5'))          #   3.4%
    models.append(load_model('final/model_MIN_LOSS.h5'))        #   5.6%
    models.append(load_model('final/model_MIN_VAL_LOSS.h5'))    #   4.3%
    models.append(load_model('final/model_MIN_LOSS_RATIO.h5'))  #   3.4%
    models.append(load_model('final/model_50.h5'))              #   3.7%
    # models.append(load_model('model_MIN_LOSS1.h5'))           #   4.3%
    # models.append(load_model('model3.h5'))                    #   8.2%

    err = 0
    for model in models :
        for val1 in ev_range :
            for val2 in ev_enemy:
                predictions = model.predict(np.matrix([val1,val2]))
                if not evaluate_data(predictions[0][1],predictions[0][0],val1,val2):
                    # print('error -> ',val1,val2)
                    err = err + 1
        print('error % ->',err/(len(ev_range)*len(ev_enemy)))
        tmp = err/(len(ev_range)*len(ev_enemy))
        if tmp < best:
            best = tmp
            model.save('model_CURRENT_BEST.h5')
        if (err/(len(ev_range)*len(ev_enemy))) < 0.01:
            model.save('model_BEST.h5')
        err = 0
