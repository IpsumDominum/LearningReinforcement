import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state','reward'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DQN(nn.Module):

    def __init__(self, h, w, outputs,frame_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(frame_size, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


def preprocess(obs):
    obs = cv2.resize(obs,(84,84))
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    ret,obs = cv2.threshold(obs,1,255,cv2.THRESH_BINARY)
    #obs = obs.transpose((2, 0, 1))
    obs = np.ascontiguousarray(obs, dtype=np.float32) / 255
    return obs

def to_tensor(obs):
    return torch.from_numpy(obs).type(torch.cuda.FloatTensor).to(device)

