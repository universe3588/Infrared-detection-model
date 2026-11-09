import numpy as np
import torch
from torch.backends import cudnn
import tqdm
import os
cudnn.benchmark = True
import argparse
from net import Net
import os
import time
from thop import profile
import torch


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

model_name = 'L2SKNet_UNet'
input_img = torch.rand(1,1,256,256).cuda()
net = Net(model_name).cuda()
net.eval()

repetitions = 300


print('warm up ...\n')
with torch.no_grad():
    for _ in range(100):
        _ = net(input_img)

torch.cuda.synchronize()

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

timings = np.zeros((repetitions, 1))

print('testing ...\n')
with torch.no_grad():
    for rep in tqdm.tqdm(range(repetitions)):
        starter.record()
        _ = net(input_img)
        ender.record()
        torch.cuda.synchronize() 
        curr_time = starter.elapsed_time(ender) 
        timings[rep] = curr_time

avg = timings.sum()/repetitions
print('\navg={}\n'.format(avg))