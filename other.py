import tensorflow as tf
import numpy as np
import random
from collections import deque

# Hyper Parameters for DQN
GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch

class OtherDQN():
  # DQN Agent
  def __init__(self,Query=False):
    # init experience replay
    self.replay_buffer = deque()
    self.switchs = deque()
    # init some parameters
    self.time_step = 0
    self.epsilon = INITIAL_EPSILON
    self.state_dim = 2
    self.action_dim = 4

    self.action_dim += 1            # stay

    # self.askAction_dim = self.action_dim + 1
    self.askAction_dim = 1
  
    self.useQuery = Query

    self.create_Q_network()
    self.create_training_method()
    
    # Init session
    self.session = tf.InteractiveSession()
    self.session.run(tf.global_variables_initializer())
    self.times = 0
    # print("dag")
    # tf.summary.FileWriter('odqnlogs', self.session.graph)
    
    # exit(0)

  def create_Q_network(self):
    # network weights
    W1 = self.weight_variable([self.state_dim,20])
    b1 = self.bias_variable([20])
    W2 = self.weight_variable([20,self.action_dim])
    b2 = self.bias_variable([self.action_dim])
    
    # input layer
    self.state_input = tf.placeholder("float",[None,self.state_dim])
    self.switchY = tf.placeholder(tf.float32,[None,2])
    # hidden layers
    h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)
    
    if self.useQuery :
      askQ = tf.layers.dense(h_layer,self.askAction_dim)
      Switcher = tf.layers.dense(self.state_input,2,activation=tf.nn.softmax,name="switch")
      moveQ = tf.layers.dense(h_layer,self.action_dim)
   
      moveQ = moveQ * tf.reshape(Switcher[:,0],[-1,1]) 
      askQ = askQ * tf.reshape(Switcher[:,1],[-1,1])
      self.Q_value = tf.concat([moveQ,askQ],1)
      self.switch = Switcher
    else:
      # Q Value layer
      self.Q_value = tf.matmul(h_layer,W2) + b2


  def create_training_method(self):
    if self.useQuery :
      self.action_input = tf.placeholder("float",[None,self.action_dim+self.askAction_dim]) # one hot presentation
    else:
      self.action_input = tf.placeholder("float",[None,self.action_dim]) # one hot presentation
          
    self.y_input = tf.placeholder("float",[None])
    Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),reduction_indices = 1)
    self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))

    variables = tf.trainable_variables()
    var1 = []
    var2 = []
    for v in variables :
      if "switch" in v.name :
        var1.append(v)
      else:
        var2.append(v)

    self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost,var_list=var2)
    if self.useQuery :
      self.switchloss = -tf.reduce_mean( tf.reduce_sum( tf.multiply(self.switchY , tf.log(self.switch) ) , 1) )
      self.switchOptimizer = tf.train.AdamOptimizer(0.0001).minimize(self.switchloss,var_list=var1)
  
  def addMemory(self,memory):
    self.perceive(*memory)

  def perceive(self,state,action,reward,next_state,done):
    if self.useQuery :
      one_hot_action = np.zeros(self.action_dim+self.askAction_dim)
    else:
      one_hot_action = np.zeros(self.action_dim)          
          
    one_hot_action[action] = 1
    self.replay_buffer.append((state,one_hot_action,reward,next_state,done))
    if len(self.replay_buffer) > REPLAY_SIZE:
      self.replay_buffer.popleft()

    if len(self.replay_buffer) > BATCH_SIZE:
      self.train_Q_network()

  def train_Q_network(self):
    self.time_step += 1
    # Step 1: obtain random minibatch from replay memory
    minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
    state_batch = [data[0] for data in minibatch]
    action_batch = [data[1] for data in minibatch]
    reward_batch = [data[2] for data in minibatch]
    next_state_batch = [data[3] for data in minibatch]

    # Step 2: calculate y
    y_batch = []
    Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})
    for i in range(0,BATCH_SIZE):
      done = minibatch[i][4]
      if done:
        y_batch.append(reward_batch[i])
      else :
        y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

    # if self.times % 100 == 0 :
    #     print(self.cost.eval(feed_dict={
    #         self.y_input:y_batch,
    #         self.action_input:action_batch,
    #         self.state_input:state_batch
    #     }))
    # self.times += 1
    
    self.optimizer.run(feed_dict={
      self.y_input:y_batch,
      self.action_input:action_batch,
      self.state_input:state_batch
      })

  

  def egreedy_action(self,state):
    if not self.useQuery :
      Q_value = self.Q_value.eval(feed_dict = {
        self.state_input:[state]
        })[0]
    else:  
      Q_value , switch = self.session.run((self.Q_value,self.switch),feed_dict = {
        self.state_input:[state]
        })
      Q_value = Q_value[0]
    # print(Q_value,switch)
    if random.random() <= 0.46:
      if self.useQuery :
        return random.randint(0,self.action_dim + self.askAction_dim - 1) , True
      else:
        return random.randint(0,self.action_dim - 1) , True
            
    else:
      # if switch > 0.5 :
      #   return np.argmax(Q_value[0:self.action_dim])
      # else:
      #   return np.argmax(Q_value[self.action_dim:]) + self.action_dim
            
      return np.argmax(Q_value) , False

    self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/10000

  def action(self,state):
    return np.argmax(self.Q_value.eval(feed_dict = {
      self.state_input:[state]
      })[0])

  def addSwitchMemory(self,switch):
    self.switchs.append(switch)       
    if len(self.switchs) > REPLAY_SIZE:
      self.switchs.popleft() 
  
    #train switch
    batch_size = 16
    for i in range(5) :
      if self.useQuery and len(self.switchs) :
        if len(self.switchs) > batch_size :
          minibatch = random.sample(self.switchs,batch_size)
        else:
          minibatch = random.sample(self.switchs,len(self.switchs))
              
        state_batch = [data[0] for data in minibatch]
        switch_batch = [data[1] for data in minibatch]

        self.switchOptimizer.run(feed_dict={
          self.switchY:switch_batch,
          self.state_input:state_batch
          })

  def weight_variable(self,shape):
    initial = tf.truncated_normal(shape)
    return tf.Variable(initial)

  def bias_variable(self,shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)


