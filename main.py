import keras
from keras.models import Sequential
from keras.layers import Dense
import math
import random
import numpy as np
import os
import tensorflow as tf
import datetime

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TRAINING_SIZE = 20000
NUMBER_OF_EPOCHS = 10000
BATCH_SIZE = 64

# ulazi: distanca od kosa 'x', distanca protivnika 'p'
# izlaz: brzina izbacaja 'v', ugao izbacaja 'alfa' 

#scaled values: bascet_height= 55 cm, enemy_height=40 ,bascet_radius = 45.72 cm  

print('start')
#v[0] = x, v[1] = alfa, v[2] = velocity

def get_min_angle(enemy_distance):
    r = math.sqrt(enemy_distance**2+52**2)
    alfa = math.asin(40.0/r)
    return alfa


def curve(v):
    y = v[0]*math.tan(v[1]) - (980.6/2) * (v[0]**2/(v[2]**2)*math.cos(v[1])**2)
    return y

def get_best_velocity(distance,min_alfa):
    min_v = 0.1
    zero_val = curve([distance,min_alfa,min_v])
    fit = abs( 55.0 - zero_val)
    for v in range(500,3000):
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

def load_data():
    print('loading data')
    if os.path.isfile('input_values') and os.path.isfile('output_values'):

        in_v = np.loadtxt('input_values')
        out_v =  np.loadtxt('output_values')
        test_in = np.loadtxt('input_values_test')
        test_out = np.loadtxt('output_values_test')
        return in_v,out_v,test_in,test_out
    else:
        in_v,out_v = prepare_data()
        test_in,test_out = prepare_test_data() 
        return in_v,out_v,test_in,test_out

def prepare_data():
    in_values = []
    result = []

    for i in range(0,TRAINING_SIZE):
        robot_distance = random.random() * 1300.0 + 600.0
        enemy_distance = random.random() * 240+ 80

        min_alfa = get_min_angle(enemy_distance) + 0.1
        
        velocity = get_best_velocity(robot_distance,min_alfa)

        in_values.append([robot_distance,enemy_distance])
        result.append([velocity,min_alfa])
        
    # in_file = open('input_values','w')
    # out_file = open('output_values','w')
    np.savetxt('input_values',in_values)
    np.savetxt('output_values',result)

    return in_values,result

def prepare_test_data():
    in_values = []
    result = []

    for i in range(0,2000):
        robot_distance = random.random() * 1125.0 + 675.0
        enemy_distance = random.random() * 200 + 100

        min_alfa = get_min_angle(enemy_distance) + 0.1
        
        velocity = get_best_velocity(robot_distance,min_alfa)

        in_values.append([robot_distance,enemy_distance])
        result.append([velocity,min_alfa])
        
    # in_file = open('input_values','w')
    # out_file = open('output_values','w')
    np.savetxt('input_values_test',in_values)
    np.savetxt('output_values_test',result)

    return in_values,result

def prepare_eval_data():
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
        # ev_in_range.append(i*100)
        # ev_in_range.append(i*100+25)
        # ev_in_range.append(i*100+50)
        # ev_in_range.append(i*100+75)

    
    # for i in range (1,2):
    #     ev_in_enemy.append(i*100)
    #     ev_in_enemy.append(i*100+25)
    #     ev_in_enemy.append(i*100+50)
    #     ev_in_enemy.append(i*100+75)

    return temp

class CallBack(keras.callbacks.Callback):
    def __init__(self,model, optimizer):
        self.model = model
        self.opt = optimizer
        self.min_val_loss = 1000
        self.min_loss = 1000
        self.min_loss_ratio = self.min_val_loss*self.min_loss

    def on_epoch_end(self, batch, logs):
        if self.min_val_loss > logs.get('val_loss'):
            print(' NEW MIN VAL_LOSS ->',logs.get('val_loss'))
            self.min_val_loss=logs.get('val_loss')

            if batch > 100:
                self.model.save('final/model_MIN_VAL_LOSS.h5')

        if self.min_loss > logs.get('loss'):
            print(' NEW MIN LOSS ->',logs.get('loss'))
            self.min_loss=logs.get('loss')

            if batch > 100:
                self.model.save('final/model_MIN_LOSS.h5')

        if self.min_loss_ratio > logs.get('loss')*logs.get('val_loss'):
            print(' NEW MIN LOSS RATIO->',logs.get('loss')*logs.get('val_loss'))
            self.min_loss_ratio=logs.get('loss')*logs.get('val_loss')

            if batch > 100:
                self.model.save('final/model_MIN_LOSS_RATIO.h5')


        # if batch > 4000:
        #     self.opt.learning_rate = self.opt.learning_rate - self.opt.learning_rate/2
            
        # if batch > 7000:
        #     self.opt.learning_rate = self.opt.learning_rate - self.opt.learning_rate/2
        return


def train_model(in_v,out,in_test,out_test):


    model = Sequential()
    # keras.activations.sigmoid

    log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    model.add(Dense(64,input_shape=(2,),activation='relu'))
    model.add(Dense(256,activation='sigmoid'))
    # model.add(Dense(256,activation='relu'))
    model.add(Dense(2,activation='linear'))

    # print('input ',np.matrix(in_v))
    # print('output ',np.matrix(out))
    
    sgd = keras.optimizers.SGD()
    adx = keras.optimizers.Adamax()
    # current best for adamax
    model.compile(loss='mean_squared_error', optimizer=adx, metrics=['accuracy'])
    
    #model.optimizer.learning_rate = 0.01
    history = model.fit(x=np.matrix(in_v),y=np.matrix(out),epochs=NUMBER_OF_EPOCHS,batch_size=BATCH_SIZE,validation_data=[np.matrix(in_test),np.matrix(out_test)],callbacks=[CallBack(model,adx)],use_multiprocessing=True)
    print(history)

    # model.evaluate()
    return model

in_v,out,in_test,out_test = load_data()

print('training set: ',in_v,' out: ',out)
model = train_model(in_v,out,in_test,out_test)

test_distance = 800

predictions = model.predict(np.transpose(np.transpose([[test_distance,300]])))

print('prediction: ',predictions[0][0],predictions[0][1])
print('height: ',curve([test_distance,predictions[0][1],predictions[0][0]]))

distance = find_distance(predictions[0][1],predictions[0][0])

print('diference: ',test_distance-distance)

#if abs(test_distance - distance) < 46:
model.save('model3.h5')

angle = get_min_angle(300)

v = get_best_velocity(test_distance,angle)

print('best_values: ',v,angle)
print('best_result: ', curve([test_distance,angle,v]))