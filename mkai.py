import numpy as np
import random# to randomly implement experience replay
import os#to save the brain and reuse it later
import torch# neural network implementation
import torch.nn as nn #all tools to implement NT, take 5 state input and give action to play
import torch.nn.functional as F#cal. of loss  function
import torch.optim as optim #as optimiser to perform ST. Gradient descent
import torch.autograd as autograd #to convert varib look google
from torch.autograd import Variable # google it
 
class Network(nn.Module): #our Network class will inherit fromModule class..this is basically a child class 
    
    def __init__(self, input_size, nb_action):
        #input is 5 dimension & nb =output actoin are 3
        super(Network, self).__init__ ()#to use tools of parent class ie. Module class in nn 
        self.input_size = input_size # creating input layer  self.inpit means it is connected to object
        self.nb_action = nb_action #output layer
        
        #creating full connection btw layers like synapses 1 hidden layer with 2 connections
        # 1 with input and 1 output
        self.fc1=nn.Linear(input_size,60) #input_size is layers connected before and 30 is neurons connected'it is a second layer neurons connected to it
        #u can play with this arct. 30 is no. of neurons in 2nd layer..u can inc or dec it 
        self.fc2 = nn.Linear(60,60 ) #middle layer with 30 neurons connected to output layer 
        self.fc3=nn.Linear(60,nb_action)
    #now we will activate neurons..take procedure forward
    def forward(self, state):
        #x represent hidden neurons take input layer fc1 and apply activation function on it
        x= F.relu(self.fc1(state))
        y=F.relu(self.fc2(x))
        q_values = self.fc3(y)  #these are not actions now..these are Q values..this is output layer
        # above x is input of fc2 bcz its a layer before output layer
        #actions are obtained from this using argmax or softmax 
        return q_values
    #basic arct. done ..can add more hidden layers and neurons instead of 30

#implementing experience replay for our model to learn long term correlation
# have memory of last 100 replays it will be capacity
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity=capacity
        self.memory= []    #memory will be a list of transitions and will be appended with newones
    
    def push(self,event): #append an event or transition in memory list
    #event= tuple of 4 last state,new state, last reward, last action At
        self.memory.append(event)
        if len(self.memory) >self.capacity:
            del self.memory[0]
            
    def sample(self, batch_size):
        samples= zip(*random.sample(self.memory, batch_size)) #we take random samples from memory with fixed size of batch size
        # zip(*) funct is reshaping fun..will separate actions and rewards etc. for pytorch
        return map(lambda x: Variable (torch.cat(x,0)), samples) # x is sample, tensors ->torch variable in 1 dimension with both tensors and gradient 

#implementing deep q learning
class Dqn():
    def __init__(self,input_size, nb_action,gamma):#gamma=discount factor 
        self.gamma= gamma
        self.reward_window=[] #mean of rewards over time of last 100 rewards 
        self.model= Network(input_size, nb_action) #neural network made as model as an object of Network class
        #above created 1 deep q learning network
        self.memory=ReplayMemory(100000) #created an object of memory , 100000 transitions ie capacity passed to get small samples of transitions on which model will learn
        self.optimizer= optim.Adam(self.model.parameters(), lr= 0.001)# creating an object of Adam class (try dif.) , passing our model to optimizer & its learning rate 
        #to access parameters used parameters, make lr large to give time AI to learn properly
        self.last_state= torch.Tensor(input_size).unsqueeze(0)  #our state have 5 dimensions , converted in object of tensor class & making fake dimension     
        #in unsqueeze() 0 is the index of fake dimnension, which have to the 1st dimension of last_state so 0 index
        self.last_action=0
        self.last_reward=0
    
    def select_acton(self, state): #actons depends on output of NN which depent on input state 
        #using softmax we get final action to play
        probs= F.softmax(self.model(Variable(state, volatile=True))*100)#we get probability distribution by paasing model ie NN
        #doing volatile =true improves algo. , bcz gradient will not be included
        # 7 above is temperature parameter..increases the confidence of our action
        #if we dont want AI to work just put temp =0
        action = probs.multinomial() #random draw from probs , high Q value high prob. , 
        #probs.multinomial() returns pytorch vatiable with fake batch 
        return action.data[0,0] #return only 0,1,2 index
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action): # taking input self & transitions , they are all alligned to time by concatenation in 51 
        #we have to ake experiencex ie batches for our model to learn...with short memory our model cannot learn
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1).squeeze(1)) 
        next_output= self.model(batch_next_state).detach().max(1)[0] #for cal. of target we need next output
        #above: actons is index 1, states with 0, we get maximum of Q values of next state with actions 
        target =self.gamma * next_output+ batch_reward
        td_loss=F.smooth_l1_loss(outputs,target) #temporal difference loss, we apply the function 
        self.optimizer.zero_grad() #zero grad wwill reinitialize the loop 
        td_loss.backward(retain_variables= True) #back propagation , back propagates the loss in neural net.
        self.optimizer.step() #update the weights acc to loss
        #google that squueze and unsqueeze 
        
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0) #signal is a state but now its a list fo 5 elements ..signal 1,2,3,and orientations ..we converted it into torch tensor
        #unsqueeze to create fake dimension at an index 0
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100: # make AI learn , 1st memory is object of replaymemory class and 2nd is the attribute there only
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state # we have reached this state so it becomes last state
        self.last_reward = reward  
        self.reward_window.append(reward)# keep track of how our training is going 
        if len(self.reward_window) > 1000: #fixed size..evolution of rewards 
            del self.reward_window[0] #1st element deleted , find if mean of window is increasing or not.
        return action
    #update function not only updates value but also returns the last action played on reaching the last state
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1) # added 1 so that dr. is never 0
    
    #saving our model so that we can quit 
    def save(self):# will save our neural network ie model and optimizer that is related to weights which are then to actions
        torch.save({'state_dict':self.model.state_dict(),   # dictionary is made
                    'optimizer':self.optimizer.state_dict(),# state dict will pass parameters 
                     },'last_brain.pth' )# path where it is saved
    

  #to lead our brain back what we have saved
    def load(self):
        if os.path.isfile('last_brain.pth'):# checks if file exist
            print('loading ...')
            checkpoint= torch.load('last_brain.pth')
            # we have to now update our parameters of optmizer and weights according to last_brain 
            self.model.load_state_dict(checkpoint['state_dict']) #state_dict is our key that corresponds to our model , in save fun.
            self.optimizer.load_state_dict(checkpoint['optimizer']) #key corresponds to optimizer
            print('done')
            
        else:
            print('no file found')
            
        



