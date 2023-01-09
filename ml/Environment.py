import gym
from gym import spaces
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class Environment(gym.Env):
    def __init__(self, n_actions, n_observations):
        super(Environment, self).__init__()
        
        # self.action_space = spaces.Discrete(n_actions) #start from 0
        # self.observation_space = spaces.Discrete(n_observations) #start from 0
        observations = np.array([i for i in range(7)])
        observations = observations.reshape(len(observations), 1)
        actions = np.array([j for j in range(4)])
        actions = actions.reshape(len(actions), 1)
        onehot_encoder = OneHotEncoder(sparse=False)
        self.observation_one_hot = onehot_encoder.fit_transform(observations, actions)
                
        self.front = 0 # 前
        self.top = 0 # 上
        self.down = 0 # 下

        self.length = -45 # 判斷上下

    def set_scene_info(self, scene_info):
        self.scene_info = scene_info
    
    def get_action(self):
        return self.action

    #設定observation
    def reset(self):        
        if self.scene_info["velocity"] < 10:
            self.length = -60
        position = ((self.scene_info["all_cars_pos"][0][1]-100)//50)+1   # 目前車在哪個賽道上

        observation = 0
        self.front = 0
        self.top = 0
        self.down = 0

        for i in range (1,len(self.scene_info["all_cars_pos"])):
            if self.scene_info["all_cars_pos"][i][1] == ((position-1)*50)+100+10:    # 同賽道有車
                if self.scene_info["all_cars_pos"][i][0] - self.scene_info["all_cars_pos"][0][0] < 210 and self.scene_info["all_cars_pos"][i][0] > self.scene_info["all_cars_pos"][0][0]:   # 前方有車
                    self.front = 1
                    break
            else:
                self.front = 0
    
        if self.front == 1:  # 如果前方有車
            for j in range(1,len(self.scene_info["all_cars_pos"])):
                    if (self.scene_info["all_cars_pos"][j][0] - self.scene_info["all_cars_pos"][0][0] < 180 and self.scene_info["all_cars_pos"][j][0] - self.scene_info["all_cars_pos"][0][0] > self.length) and self.scene_info["all_cars_pos"][j][1] == (position-2)*50+100+10: # 上方有車
                        self.top = 1
                    if (self.scene_info["all_cars_pos"][j][0] - self.scene_info["all_cars_pos"][0][0] < 180 and self.scene_info["all_cars_pos"][j][0] - self.scene_info["all_cars_pos"][0][0] > self.length) and self.scene_info["all_cars_pos"][j][1] == (position)*50+100+10: # 下方有車
                        self.down = 1

        if position == 1:
            self.top = 1
        if position == 9:
            self.down = 1

        if self.front == 0:          # 前方沒車
            if self.scene_info["all_cars_pos"][0][1] - (((position-1)*50)+100+10) > 5 and self.scene_info["all_cars_pos"][0][1] > ((position-1)*50)+100+10:     # 車子在路線下方
                observation = 2 
            elif self.scene_info["all_cars_pos"][0][1] - (((position-1)*50)+100+10) < -5 and self.scene_info["all_cars_pos"][0][1] < ((position-1)*50)+100+10:     # 車子在路線上方
                observation = 1 
            else:
                observation = 0  

        elif self.front == 1 :     # 前方有車
            if self.top == 1 and self.down == 1:   # 上下前都有車
                observation = 3 
            elif self.top == 1:       # 上前有車
                observation = 4 
            elif self.down == 1:     # 前下有車
                observation = 5 
            else:                    # 只有前面有車
                observation = 6 
        else:
            pass
        
        # reward, done, info can't be included
        return self.observation_one_hot[observation]

    #設reward
    def step(self, action):      
        reward = 0
        # 0: brake / 1: speed / 2: left / 3: right / 4: nothing

        observation = self.reset()

        if np.array_equal(observation, self.observation_one_hot[0,0]):      # 前方沒車
            reward += -10
        if np.array_equal(observation, self.observation_one_hot[0,1]):
            reward += 10
        if np.array_equal(observation, self.observation_one_hot[0,2]):
            reward += -10
        if np.array_equal(observation, self.observation_one_hot[0,3]):
            reward += -10
        if np.array_equal(observation, self.observation_one_hot[0,4]):
            reward += 0
            
        if np.array_equal(observation, self.observation_one_hot[1,0]):      # 前方沒車且車子在路線上方
            reward += -10
        if np.array_equal(observation, self.observation_one_hot[1,1]):
            reward += -10
        if np.array_equal(observation, self.observation_one_hot[1,2]):
            reward += -10
        if np.array_equal(observation, self.observation_one_hot[1,3]):
            reward += 10
        if np.array_equal(observation, self.observation_one_hot[1,4]):
            reward += 0
            
        if np.array_equal(observation, self.observation_one_hot[2,0]):      # 前方沒車且車子在路線下方
            reward += -10
        if np.array_equal(observation, self.observation_one_hot[2,1]):
            reward += -10
        if np.array_equal(observation, self.observation_one_hot[2,2]):
            reward += 10
        if np.array_equal(observation, self.observation_one_hot[2,3]):
            reward += -10
        if np.array_equal(observation, self.observation_one_hot[2,4]):
            reward += 0
            
        if np.array_equal(observation, self.observation_one_hot[3,0]):      # 上下前都有車
            reward += 10
        if np.array_equal(observation, self.observation_one_hot[3,1]):
            reward += -10
        if np.array_equal(observation, self.observation_one_hot[3,2]):
            reward += -10
        if np.array_equal(observation, self.observation_one_hot[3,3]):
            reward += -10
        if np.array_equal(observation, self.observation_one_hot[3,4]):
            reward += 0
        
        if np.array_equal(observation, self.observation_one_hot[4,0]):      # 前上有車
            reward += -10
        if np.array_equal(observation, self.observation_one_hot[4,1]): 
            reward += -10
        if np.array_equal(observation, self.observation_one_hot[4,2]): 
            reward += -10
        if np.array_equal(observation, self.observation_one_hot[4,3]): 
            reward += 10
        if np.array_equal(observation, self.observation_one_hot[4,4]): 
            reward += 0
                
        if np.array_equal(observation, self.observation_one_hot[5,0]):      # 前下有車
            reward += -10
        if np.array_equal(observation, self.observation_one_hot[5,1]):
            reward += -10
        if np.array_equal(observation, self.observation_one_hot[5,2]):
            reward += 10
        if np.array_equal(observation, self.observation_one_hot[5,3]):
            reward += -10
        if np.array_equal(observation, self.observation_one_hot[5,4]):
            reward += 0
        
        if np.array_equal(observation, self.observation_one_hot[6,0]):      # 只有前面有車
            reward += -10
        if np.array_equal(observation, self.observation_one_hot[6,1]):
            reward += -10
        if np.array_equal(observation, self.observation_one_hot[6,2]):
            reward += 10
        if np.array_equal(observation, self.observation_one_hot[6,3]):
            reward += 10
        if np.array_equal(observation, self.observation_one_hot[6,4]):
            reward += 0

        if self.scene_info["status"] != "GAME_ALIVE":
            done = 1
        else:
            done = 0

        info = {}

        return observation, reward, done, info