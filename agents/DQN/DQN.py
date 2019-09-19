from agents.DQN.network import *
from utils.Memory import PrioritizedMemory
#DQNAgent is inspired by the DQN tutorial, from Pytorch's official tutorials
#https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''implements
    weights_init()
    load_model()
    get_action()
    train_model()
    consider()        
    save_model()
    and various supporting methods etc.
'''
class DQNAgent():
    def __init__(self, state_size,frame_size,n_actions,load_path):
        self.NUM_EPOCHS = 1000
        self.BATCH_SIZE = 32
        self.DISCOUNT = 0.50
        self.UPDATE_FREQ = 10000
        self.GAMMA = 0.9
        self.EPS_END = 0.05
        self.EPS_START = 0.95
        self.EPS_DECAY = 10000
        self.STARTLIMIT = 1000
        self.LOAD_PATH = load_path
        self.use_greedy = True
        self.memory = PrioritizedMemory(10000)
        self.test_runs = 10
        self.count = 0
        self.last_action = None
        self.n_actions = n_actions
        self.policy_net = DQN(state_size, state_size, self.n_actions,frame_size).to(device)
        self.target_net = DQN(state_size, state_size, self.n_actions,frame_size).to(device)
        self.policy_net.apply(self.weights_init)
        self.optimizer = optim.RMSprop(self.policy_net.parameters(),lr=0.0001)
        self.update_target_model()
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform(m.weight)
    def update_target_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    def load_model(self):
        saved_model = torch.load(self.LOAD_PATH)
        self.policy_net.load_state_dict(saved_model['weights'])
        self.policy_net.eval()
    def get_action(self,current_state,env_count):
        if(self.use_greedy):
            sample = random.random()
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                            math.exp(-1. * env_count/ self.EPS_DECAY)
            if(sample>eps_threshold):
                with torch.no_grad():
                    action_pred = self.policy_net(to_tensor(np.array([current_state])))
                    action = torch.argmax(action_pred[0]).item()
            else:
                action = random.randrange(self.n_actions)
        else:
            action = torch.argmax(self.policy_net(to_tensor(np.array([current_state])))).item()
        return action
    def append_sample(self, error,state, action, reward, next_state,done):
        if(error==None):
            target = self.policy_net(to_tensor(np.array([state]))).detach()
            old_val = target[0][action]
            target_val = self.target_net(to_tensor(np.array([next_state]))).detach()
            if done:
                target[0][action] = reward
            else:
                target[0][action] = reward + self.DISCOUNT * torch.max(target_val)
            error = abs(old_val - target[0][action])
            self.memory.add(error.cpu(), state, np.array([action]), next_state,reward)
        else:
            self.memory.add(error, state, np.array([action]), next_state,reward)
    def train_model(self):
        try:
            sample,idx,_ = self.memory.sample(self.BATCH_SIZE)
            sample = Transition(*zip(*sample))
            s =  to_tensor(np.array(list(sample.state)))
            a =  to_tensor(np.array(list(sample.action))).type(torch.cuda.LongTensor)
            r =  to_tensor(np.array(list(sample.reward)))
            n_s = to_tensor(np.array([s for s in sample.next_state
                                    if s is not None]))
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    sample.next_state)), device=device, dtype=torch.uint8)
            pred = self.policy_net(s).gather(1,a).squeeze(1)
            next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
            target = self.target_net(n_s).detach()
            target = target[torch.arange(target.size(0)),self.policy_net(n_s).detach().max(1)[1]]
            next_state_values[non_final_mask] = target*self.GAMMA + r
            print(pred)
            next_state_values = next_state_values - next_state_values.mean()
            pred = pred - pred.mean()
            loss = F.smooth_l1_loss(pred, next_state_values)
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
                self.optimizer.step()
        except TypeError:
            #some times it samples only final states :( and things will fail
            pass
    def consider(self,env_count,previous_state,current_state,action,reward,done,train=True):
        action = self.get_action(current_state,env_count)
        if(self.last_action !=None):
            self.append_sample(None,previous_state,self.last_action,reward,current_state,done)
        self.last_action = action
        if(train==True and self.count >self.STARTLIMIT):
            self.train_model()
        self.count +=1
        if(self.count % self.UPDATE_FREQ ==0):
            self.update_target_model()
        return action
    def save_model(self):
        torch.save({"weights":self.policy_net.state_dict()},self.LOADPATH)


