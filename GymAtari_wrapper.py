import gym
from agents.DQN.network import *
from agents.DQN.DQN import DQNAgent
class GymAtari():
    def __init__(self,env_name,state_size,frame_size,agent="DQN",render=True,train=True,load_path="checkpoint.pth"):
        self.env_name = env_name
        self.frame_size = frame_size
        self.render = render
        self.train = train
        self.env = gym.make(self.env_name)
        if(agent=="DQN"):
            self.agent = DQNAgent(state_size,frame_size,self.env.action_space.n,load_path)
        else:
            raise Exception("Agent not Found")
        self.initialized = False
        self.epoch = 0
    def stack_frames(self):
        if(self.count <1):
            #if frames have not yet been stacked up, use the first observation as the whole state
            for i in range(self.frame_size):
                self.current_state[i] = self.obs
        else:
            #replacing the state one by one
            for i in reversed(range(self.frame_size-1)):
                self.current_state[i+1] = self.current_state[i]
                self.current_state[0] = self.obs
    def initialize_things(self):
        self.current_state = np.zeros((self.frame_size,84,84))
        self.previous_state = self.current_state.copy()
        self.cum_rewards = []
        self.action = 0
        self.last_action = 0
        self.count = 0
        self.cumulated_reward = 0
        self.initialized = True
        self.done = False
        self.env.reset()
    def env_step(self):
        if(self.render):self.env.render()
        obs,reward,done,_ = self.env.step(self.action)
        self.obs = preprocess(obs)
        self.stack_frames()
        self.reward = max(-0.99,min(reward,0.99))
        self.cumulated_reward += self.reward
        self.done = done
    def agent_step(self):
        self.action = self.agent.consider(self.count,self.previous_state,self.current_state,self.action,self.reward,self.done,self.train)
    def check_done(self):
        if(self.done):
            self.count = 0
            self.epoch +=1
            self.env.reset()
            self.cum_rewards.append(self.cumulated_reward)
            self.print_things()
            self.agent.update_target_model()
        else:
            self.count+=1
    def wrap_up(self):
        self.env.close()
        self.agent.save_model()
        self.initialized = False
        self.done = True
        print("Complete, Average reward is : {}".format(sum(self.cum_rewards)/len(self.cum_rewards)))
        import matplotlib.pyplot as plt
        plt.plot(self.cum_rewards)
        plt.show()
    def print_things(self):
        print(self.cumulated_reward)
        print(self.count)
        print("epoch{}".format(self.epoch))
    def main_loop(self,num_steps):
        ####################################
        for step in range(num_steps):
            self.initialize_things()
            while(self.done==False):
                self.env_step()
                self.agent_step()
                self.check_done()
        self.wrap_up()#wrap up when done
        ####################################


