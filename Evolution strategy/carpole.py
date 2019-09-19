import pdb
import gym
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
MAX_BATCH_EPISODES = 100
MAX_BATCH_STEPS = 20000
NOISE_STD = 0.1
LEARNING_RATE = 0.001
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class Net(nn.Module):
    def __init__(self, obs_size, action_size):
        super(Net, self).__init__()
        self.net = nn.Sequential(
        nn.Linear(obs_size, 128),
        nn.Sigmoid(),
        nn.Linear(128, 64),
        nn.Sigmoid(),
        nn.Linear(64, 32),
        nn.Sigmoid(),
        nn.Linear(32, action_size),
        nn.Softmax(dim=0)
        )
    def forward(self, x):
        return self.net(x)
def preprocess(frame):
    frame = cv2.resize(frame,(84,84))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret,frame = cv2.threshold(frame,1,255,cv2.THRESH_BINARY)
    #obs = obs.transpose((2, 0, 1))
    frame = np.ascontiguousarray(frame, dtype=np.float32) / 255
    return torch.from_numpy(np.array(frame.flatten())).type(torch.cuda.FloatTensor)
def evaluate(env, net,render=False):
    obs = env.reset()
    reward = 0.0
    steps = 0    
    while True:
        if render:
            env.render()
        obs_v = preprocess(obs)
        act_prob = net(obs_v)
        acts = act_prob.max(dim=0)[1]
        obs, r, done, _ = env.step(acts.detach().cpu().data.numpy())
        reward += r
        steps += 1
        if done:
            break
    return reward, steps
def sample_noise(net):
    pos = []
    neg = []
    for p in net.parameters():
        noise_t =torch.cuda.FloatTensor(np.random.normal(size=p.data.size()).astype(np.float32))
        pos.append(noise_t)
        neg.append(-noise_t)
    return pos, neg
def eval_with_noise(env, net, noise):
    old_params = net.state_dict()
    for p, p_n in zip(net.parameters(), noise):
        p.data += NOISE_STD * p_n
        r, s = evaluate(env, net)
        net.load_state_dict(old_params)
    return r, s
def train_step(net, batch_noise, batch_reward, step_idx):
    norm_reward = np.array(batch_reward)
    norm_reward -= np.mean(norm_reward)
    s = np.std(norm_reward)
    if abs(s) > 1e-6:
        norm_reward /= s
    weighted_noise = None
    for noise, reward in zip(batch_noise, norm_reward):
        if weighted_noise is None:
            weighted_noise = [reward * p_n for p_n in noise]
        else:
            for w_n, p_n in zip(weighted_noise, noise):
                w_n += reward * p_n
    m_updates = []
    for p, p_update in zip(net.parameters(), weighted_noise):
        update = p_update / (len(batch_reward) * NOISE_STD)
        p.data += LEARNING_RATE * update
        m_updates.append(torch.norm(update))
if __name__ == "__main__":
    env = gym.make("SpaceInvaders-v0")
    obs_shape = 84*84
    net = Net(obs_shape,env.action_space.n).to(device)
    load = False
    step_idx = 0 
    if load == True:
        net.load_state_dict(torch.load("weights"))
        net.eval()
    while True:
        t_start = time.time()
        batch_noise = []
        batch_reward = []
        batch_steps = 0
        for _ in range(MAX_BATCH_EPISODES):
            noise, neg_noise = sample_noise(net)
            batch_noise.append(noise)
            batch_noise.append(neg_noise)
            reward,steps = eval_with_noise(env,net,noise)
            batch_reward.append(reward)
            batch_steps += steps
            reward,steps = eval_with_noise(env,net,neg_noise)
            batch_reward.append(reward)
            batch_steps += steps
            if batch_steps >MAX_BATCH_STEPS:
                break
        step_idx += 1
        m_reward = np.mean(batch_reward)
        if m_reward > 1000:
            print("solved in %d steps" % step_idx)
            torch.save(net.state_dict(),"weights")
            r,s = evaluate(env,net,render=True)     
            break
        train_step(net,batch_noise,batch_reward,step_idx)
        speed = batch_steps / (time.time() - t_start)

        print("%d: reward=%.2f, speed=%.2f f/s" % (step_idx, m_reward, speed))
    env.close()