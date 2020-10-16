import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.initializers import glorot_normal as x_init
from NoiseDense import NoisyDense

class DuelingDQN(keras.Model):
    def __init__(self, n_actions, first_layer_dim=10, second_layer_dim=5, lstm_cells = 20, activation="relu", normalize=False, lstm=True):
        super(DuelingDQN, self).__init__()
        self.lstm = keras.layers.LSTM(units=lstm_cells,
                                        recurrent_activation="sigmoid", activation="tanh",
                                        use_bias=True,kernel_initializer=x_init, return_sequences=False,
                                        stateful=False)
        self.norm_layer = keras.layers.BatchNormalization()
        self.dense1 = keras.layers.Dense(first_layer_dim, activation=activation)
        self.dense2 = keras.layers.Dense(second_layer_dim, activation=activation)
        self.noise = keras.layers.GaussianNoise(stddev=0.02)
        #self.noisy_dense = NoisyDense(second_layer_dim, activation=activation)
        self.A = keras.layers.Dense(n_actions, activation=None)
        self.V = keras.layers.Dense(1, activation=None)
        
        
        self.norm  = normalize
        self.run_lstm = lstm
       
    def call(self, state):
        if self.run_lstm:
            state = tf.expand_dims(state, axis=1)
            x = self.lstm(state)
            adv = self.dense1(x)
        else:
            adv = self.dense1(state)
        #adv = self.noise(adv)
        adv = self.dense2(adv)
        #adv = self.noisy_dense(adv)
        adv = self.A(adv)
        if self.norm:
            A = self.norm_layer(adv)
        else:
            A = adv
        
        if self.run_lstm:
            v = self.dense1(x)
        else:
            v = self.dense1(state)
        
        v = self.dense2(v)
        V = self.V(v)
        
        Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))
        return Q
    
    def advantage(self, state):
        if self.run_lstm:
            state = tf.expand_dims(state, axis=1)
            x = self.lstm(state)
            adv = self.dense1(x)
        else:
            adv = self.dense1(state)
        #adv = self.noise(adv)
        adv = self.dense2(adv)
        #adv = self.noisy_dense(adv)
        adv = self.A(adv)
        if self.norm:
            A = self.norm_layer(adv)
        else:
            A = adv  
        
        return A
    
class ReplayBuffer(object):
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        
        self.state_memory = np.zeros((self.mem_size, *input_shape), 
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), 
                                     dtype=np.float32) 
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = np.array(action)
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        
        self.mem_cntr += 1
        
    def sample_buufer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        
        return states, actions, rewards, states_, dones
    
class Agent(object):
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, 
                 input_dims, epsilon_dec = 1e-3, epsilon_end=0.001,
                 mem_size = 1000000, first_layer_dim=256, second_layer_dim=128, 
                 lstm_cells = 64, activation="relu", normalize=False, lstm=True,
                 replace=100):
        
        self.actions_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.replace = replace
        self.learning_rate = lr
        
        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = DuelingDQN(n_actions, first_layer_dim, second_layer_dim, 
                                lstm_cells, activation, normalize, lstm)
        self.q_next = DuelingDQN(n_actions, first_layer_dim, second_layer_dim, 
                                lstm_cells, activation, normalize, lstm)
        
        self.q_eval.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mean_squared_error")
        self.q_next.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mean_squared_error")
        
    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        
    def choose_action(self, observation):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.actions_space)
        else:
            state = np.array([observation])
            actions = self.q_eval.advantage(state)
            action = tf.argmax(actions, axis=1).numpy()[0]
        return action
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
    
        if self.learn_step_counter % self.replace == 0:
            self.q_next.set_weights(self.q_eval.get_weights())
            
        states, actions, rewards, states_, dones = self.memory.sample_buufer(self.batch_size)
            
        
        q_pred = self.q_eval(states)
        q_next = tf.math.reduce_max(self.q_next(states_), axis=1, keepdims=True).numpy()
        q_target = np.copy(q_pred)
        #max_actions = tf.math.argmax(self.q_eval(states_), axis=1)
        
        ## trainig loop:
        for idx, terminal in enumerate(dones):
            if terminal:
                q_next[idx] = 0.0
            q_target[idx, actions[idx]] = rewards[idx] + self.gamma*q_next[idx]
            
        self.q_eval.train_on_batch(states, q_target)
        
        self.epsilon = self.epsilon-self.epsilon_dec if self.epsilon>self.epsilon_end else self.epsilon_end
        
        self.learn_step_counter +=1  
                                                 
                                                
        
            
    
            
    
        
    
        
        