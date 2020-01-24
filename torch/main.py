import torch
import numpy as np
import random
import os
import math
import tensorflow.core
import torch.nn as nn
import torch.optim as optim

print('CUDA CORES: ',torch.cuda.is_available())

def load_data():
    print('loading data')
    if os.path.isfile('input_values') and os.path.isfile('output_values'):

        in_v = np.loadtxt('input_values')
        out_v =  np.loadtxt('output_values')
        test_in = np.loadtxt('input_values_test')
        test_out = np.loadtxt('output_values_test')
        return in_v,out_v,test_in,test_out
    else:
        return [prepare_data(),prepare_test_data()]


def prepare_data():
    in_values = []
    result = []

    for i in range(0,10000):
        robot_distance = random.random() * 1125.0 + 675.0
        enemy_distance = random.random() * 200 + 100

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

    for i in range(0,1200):
        robot_distance = random.random() * 1125.0 + 675.0
        enemy_distance = random.random() * 220 + 100

        min_alfa = get_min_angle(enemy_distance) + 0.1
        
        velocity = get_best_velocity(robot_distance,min_alfa)

        in_values.append([robot_distance,enemy_distance])
        result.append([velocity,min_alfa])
        
    # in_file = open('input_values','w')
    # out_file = open('output_values','w')
    np.savetxt('input_values_test',in_values)
    np.savetxt('output_values_test',result)

    return in_values,result

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

# input_tensor = torch.tensor(in_d)
# out_tensor = torch.tensor(out_d)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(2,64)
        self.hidden2 = torch.nn.ReLU()
        self.hidden3 = torch.nn.Linear(64,256)
        self.sig = torch.nn.Sigmoid()
        self.output = torch.nn.Linear(256,2)

    def forward(self, x):
        x = self.hidden(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.sig(x)
        x = self.output(x)

        return x



in_d,out_d,in_test,out_test = load_data()
net = Net()

crit = nn.MSELoss()
# opt = optim.SGD(params=net.parameters(),lr= 0.01)
opt = optim.Adamax(params = net.parameters(),lr=0.002,betas=(0.9,0.999))


for epoch in range(100):
    loss_value = 0.0
    
    i = 0

    for values in in_d:

        opt.zero_grad()

        temp = np.matrix(values)
        tensorIn = torch.from_numpy(temp).float()
        outs = net(tensorIn)
        tensorOut = torch.from_numpy(out_d[i]).float()
        loss = crit(outs,tensorOut)
        i = i+1

        loss.backward()

        opt.step()

        loss_value += loss.item()

        if i%2000 == 1999 :
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss_value / 2000))
            loss_value = 0.0


data = prepare_eval_data()

for v1 in data[0]:
    for v2 in data[1]:
        temp = torch.from_numpy(np.matrix([v1,v2])).float()

        out = net(temp)

        print('in values -> ',v1,v2,' out values -> ',out[0])

print('FINISHED')