#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-player Alpha go
@author: Tuan Dam, IAS TU Darmstadt
"""

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import argparse
import time
import copy
import os
from numpy.random import rand
from gym import wrappers
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from helpers import (argmax,check_space,is_atari_game,copy_atari_state,store_safely,
restore_atari_state,stable_normalizer,smooth,symmetric_remove,Database,power)
from rl.make_game import make_game

TAU = 1

#### Neural Networks ##
class Model():
    
    def __init__(self,Env,lr,n_hidden_layers,n_hidden_units):
        # Check the Gym environment
        self.action_dim, self.action_discrete  = check_space(Env.action_space)
        self.state_dim, self.state_discrete  = check_space(Env.observation_space)
        if not self.action_discrete: 
            raise ValueError('Continuous action space not implemented')
        
        # Placeholders
        if not self.state_discrete:
            self.x = x = tf.placeholder("float32", shape=np.append(None,self.state_dim),name='x') # state  
        else:
            self.x = x = tf.placeholder("int32", shape=np.append(None,1), name='x') # state
            x = tf.squeeze(tf.one_hot(x,self.state_dim,axis=1),axis=2)
        
        # Feedforward: Can be modified to any representation function, e.g. convolutions, residual networks, etc. 
        for i in range(n_hidden_layers):
            x = slim.fully_connected(x,n_hidden_units,activation_fn=tf.nn.elu)
            
        # Output
        log_pi_hat = slim.fully_connected(x,self.action_dim,activation_fn=None) 
        self.pi_hat = tf.nn.softmax(log_pi_hat) # policy head           
        self.V_hat = slim.fully_connected(x,1,activation_fn=None) # value head

        # Loss
        self.V = tf.placeholder("float32", shape=[None,1],name='V')
        self.pi = tf.placeholder("float32", shape=[None,self.action_dim],name='pi')
        self.V_loss = tf.losses.mean_squared_error(labels=self.V,predictions=self.V_hat)
        self.pi_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.pi,logits=log_pi_hat)
        self.loss = self.V_loss + tf.reduce_mean(self.pi_loss)
        
        self.lr = tf.Variable(lr,name="learning_rate",trainable=False)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
        self.train_op = optimizer.minimize(self.loss)
    
    def train(self,sb,Vb,pib):
        self.sess.run(self.train_op,feed_dict={self.x:sb,
                                          self.V:Vb,
                                          self.pi:pib})
    
    def predict_V(self,s):
        return self.sess.run(self.V_hat,feed_dict={self.x:s})
        
    def predict_pi(self,s):
        return self.sess.run(self.pi_hat,feed_dict={self.x:s})
   
##### MCTS functions #####
      
class Action():
    ''' Action object '''
    def __init__(self,index,parent_state,Q_init=0.0,epsilon=0.0,algorithm='uct',p=1.0):
        self.index = index
        self.parent_state = parent_state
        self.W = 0.0
        self.n = 1
        self.Q = Q_init
        self.algorithm = algorithm
        self.p = p
        self.epsilon = epsilon
                
    def add_child_state(self,s1,r,terminal,model):
        self.child_state = State(s1,r,terminal,self,self.parent_state.na,model,epsilon=self.epsilon,
                                 algorithm=self.algorithm,p=self.p)
        return self.child_state
        
    def update(self,R):
        self.n += 1
        if self.algorithm == 'uct' or self.algorithm == 'power-uct':
            self.W += R
            self.Q = self.W/self.n
        else:
            self.Q = R

class State():
    ''' State object '''

    def __init__(self,index,r,terminal,parent_action,na,model,epsilon,algorithm,p=1.0):
        ''' Initialize a new state '''
        self.index = index # state
        self.r = r # reward upon arriving in this state
        self.terminal = terminal # whether the domain terminated in this state
        self.parent_action = parent_action
        self.n = 1
        self.model = model
        self.epsilon = epsilon
        self.algorithm = algorithm
        self.p = p
        
        self.evaluate()
        # Child actions
        self.na = na
        self.child_actions = [Action(a,parent_state=self,Q_init=self.V,epsilon=self.epsilon,algorithm=algorithm,p=p)
                              for a in range(na)]
        if not self.model.state_discrete:
            self.priors = model.predict_pi(index[None,]).flatten()
        else:
            index = np.reshape(index, (1, -1))
            self.priors = model.predict_pi(index).flatten()
    
    def select(self,c=1.5):
        ''' Select one of the child actions based on UCT rule '''
        winner = 0
        if self.algorithm == 'uct' or self.algorithm == 'power-uct':
            UCT = np.array([child_action.Q + prior * c * (np.sqrt(self.n + 1)/(child_action.n + 1)) for child_action,prior in zip(self.child_actions,self.priors)])
            winner = argmax(UCT)
        elif self.algorithm == 'rents':
            random = rand()
            Qs = np.array([child_action.Q for child_action in self.child_actions])
            max_Q = np.max(Qs)
            UCT = np.array([prior * np.exp((child_action.Q - max_Q) / TAU) for child_action, prior in
                            zip(self.child_actions, self.priors)])
            UCT = UCT / np.sum(UCT)
            para_lambda = self.epsilon * self.na / np.log(self.n + 1)
            if random > para_lambda:
                winner = np.random.choice(len(self.child_actions), p=UCT)
            else:
                winner = np.random.randint(self.na)
        elif self.algorithm == 'ments':
            random = rand()
            Qs = [child_action.Q for child_action in self.child_actions]
            max_Q = np.max(Qs)
            UCT = [np.exp((child_action.Q - max_Q) / self.p_tau) for child_action in self.child_actions]
            UCT = np.squeeze(UCT)
            UCT = UCT / np.sum(UCT)
            para_lambda = self.epsilon * (self.na / np.log(self.n + 1))
            if random > para_lambda:
                winner = np.random.choice(len(self.child_actions), p=UCT)
            else:
                winner = np.random.randint(self.na)

        return self.child_actions[winner]

    def evaluate(self):
        ''' Bootstrap the state value '''
        if not self.model.state_discrete:
            self.V = np.squeeze(self.model.predict_V(self.index[None,])) if not self.terminal else np.array(0.0)
        else:
            index = np.reshape(self.index, (1, -1))
            self.V = np.squeeze(self.model.predict_V(index)) if not self.terminal else np.array(0.0)

    def update(self):
        ''' update count on backward pass '''
        self.n += 1
        counts = np.array([child_action.n for child_action in self.child_actions])
        sum = np.sum(counts)
        if self.algorithm == 'uct':
            UCT = np.array([child_action.Q for child_action in self.child_actions])
            self.V = np.sum((counts/sum)*UCT)
        elif self.algorithm == 'power-uct':
            self.V = 0
            for child_action in self.child_actions:
                self.V += (child_action.n / sum) * power(child_action.Q, self.p)
            self.V = power(self.V, 1 / self.p)
        elif self.algorithm == 'rents':
            self.Qs = np.array([child_action.Q for child_action in self.child_actions])
            max_Q = np.max(self.Qs)
            self.priors = self.model.predict_pi(self.index[None,]).flatten()
            UCT = np.array([prior * np.exp((child_action.Q - max_Q) / TAU) for child_action, prior in
                            zip(self.child_actions, self.priors)])
            self.V = max_Q + TAU * np.log(np.sum(UCT))
        elif self.algorithm == 'ments':
            self.Qs = np.array([child_action.Q for child_action in self.child_actions])
            max_Q = np.max(self.Qs)
            Q_exp = np.array([np.exp((child_action.Q - max_Q) / TAU) for child_action, prior in
                              zip(self.child_actions, self.priors)])
            self.V = max_Q + TAU * np.log(np.sum(Q_exp))

        
class MCTS():
    ''' MCTS object '''

    def __init__(self,root,root_index,model,na,gamma,epsilon,algorithm,p=1.0):
        self.root = None
        self.root_index = root_index
        self.model = model
        self.na = na
        self.gamma = gamma
        self.epsilon = epsilon
        self.algorithm=algorithm
        self.p = p
    
    def search(self,n_mcts,c,Env,mcts_env):
        ''' Perform the MCTS search from the root '''
        if self.root is None:
            self.root = State(self.root_index,r=0.0,terminal=False,parent_action=None,na=self.na,model=self.model,
                              epsilon=self.epsilon,algorithm=self.algorithm,p=self.p) # initialize new root
        else:
            self.root.parent_action = None # continue from current root
        if self.root.terminal:
            raise(ValueError("Can't do tree search from a terminal state"))

        is_atari = is_atari_game(Env)
        if is_atari:
            snapshot = copy_atari_state(Env) # for Atari: snapshot the root at the beginning     
        
        for i in range(n_mcts):     
            state = self.root # reset to root for new trace
            if not is_atari:
                mcts_env = copy.deepcopy(Env) # copy original Env to rollout from
            else:
                restore_atari_state(mcts_env,snapshot)            
            
            while not state.terminal: 
                action = state.select(c=c)
                s1,r,t,_ = mcts_env.step(action.index)
                if hasattr(action,'child_state'):
                    state = action.child_state # select
                    continue
                else:
                    state = action.add_child_state(s1,r,t,self.model) # expand
                    break

            # Back-up 
            R = state.V         
            while state.parent_action is not None: # loop back-up until root is reached
                if self.algorithm == 'rents' or self.algorithm == 'ments':
                    R = state.V
                R = state.r + self.gamma * R
                action = state.parent_action
                action.update(R)
                state = action.parent_state
                state.update()                
    
    def return_results(self,temp):
        ''' Process the output at the root node '''
        counts = np.array([child_action.n for child_action in self.root.child_actions])
        pi_target = stable_normalizer(counts, temp)
        Q = np.array([child_action.Q for child_action in self.root.child_actions])
        if self.algorithm == 'rents':
            max_Q = np.max(Q)
            if np.sum(self.root.priors) == 0:
                Q = [np.exp((child_action.Q - max_Q) / TAU) for child_action in self.root.child_actions]
            else:
                Q = [(prior * np.exp((child_action.Q - max_Q) / TAU)) for child_action, prior in
                     zip(self.root.child_actions, self.root.priors)]

            priors = self.root.getPriors()
            V_target = max_Q + TAU * np.log(np.sum(priors * np.exp(Q - max_Q/TAU)))[None]
        elif self.algorithm == 'ments':
            max_Q = np.max(Q)
            V_target = max_Q + TAU * np.log(np.sum(np.exp(Q - max_Q / TAU)))[None]
        elif self.algorithm == 'uct':
            V_target = np.sum((counts/np.sum(counts))*Q)[None]
        elif self.algorithm == 'power-uct':
            V_target = power(np.sum((counts / np.sum(counts)) * power(Q, self.p)), 1/self.p)[None]

        return self.root.index,pi_target,V_target
    
    def forward(self,a,s1):
        ''' Move the root forward '''
        if not hasattr(self.root.child_actions[a],'child_state'):
            self.root = None
            self.root_index = s1
        elif np.linalg.norm(self.root.child_actions[a].child_state.index - s1) > 0.01:
            # print('Warning: this domain seems stochastic. Not re-using the subtree for next search. '+
            #       'To deal with stochastic environments, implement progressive widening.')
            self.root = None
            self.root_index = s1            
        else:
            self.root = self.root.child_actions[a].child_state

#### Agent ##
def agent(algorithm, game,n_ep,n_mcts,max_ep_len,lr,c,p,gamma,data_size,batch_size,temp,n_hidden_layers,n_hidden_units,
          epsilon):
    ''' Outer training loop '''
    #tf.reset_default_graph()
    episode_returns = [] # storage
    timepoints = []
    # Environments
    Env = make_game(game)
    is_atari = is_atari_game(Env)
    mcts_env = make_game(game) if is_atari else None

    D = Database(max_size=data_size,batch_size=batch_size)        
    model = Model(Env=Env,lr=lr,n_hidden_layers=n_hidden_layers,n_hidden_units=n_hidden_units)  
    t_total = 0 # total steps   
    R_best = -np.Inf
 
    with tf.Session() as sess:
        model.sess = sess
        sess.run(tf.global_variables_initializer())
        for ep in range(n_ep):    
            start = time.time()
            s = Env.reset()
            R = 0.0 # Total return counter
            a_store = []
            seed = np.random.randint(1e7) # draw some Env seed
            Env.seed(seed)      
            if is_atari: 
                mcts_env.reset()
                mcts_env.seed(seed)                                

            mcts = MCTS(root_index=s,root=None,model=model,na=model.action_dim,gamma=gamma,
                        epsilon=epsilon,algorithm=algorithm,p=p) # the object responsible for MCTS searches
            for t in range(max_ep_len):
                # MCTS step
                mcts.search(n_mcts=n_mcts,c=c,Env=Env,mcts_env=mcts_env) # perform a forward search
                state,pi,V = mcts.return_results(temp) # extract the root output
                D.store((state,V,pi))

                # Make the true step
                a = np.random.choice(len(pi),p=pi)
                a_store.append(a)
                s1,r,terminal,_ = Env.step(a)
                R += r
                t_total += n_mcts # total number of environment steps (counts the mcts steps)                

                if terminal:
                    break
                else:
                    mcts.forward(a,s1)
            
            # Finished episode
            episode_returns.append(R) # store the total episode return
            timepoints.append(t_total) # store the timestep count of the episode return
            store_safely(os.getcwd(),'result',{'R':episode_returns,'t':timepoints})  

            if R > R_best:
                a_best = a_store
                seed_best = seed
                R_best = R
            print('Finished episode {}, total return: {}, total time: {} sec'.format(ep,np.round(R,2),np.round((time.time()-start),1)))
            # Train
            D.reshuffle()
            for epoch in range(1):
                for sb,Vb,pib in D:
                    model.train(sb,Vb,pib)
    # Return results
    return episode_returns, timepoints, a_best, seed_best, R_best

#### Command line call, parsing and plotting ##
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', default='uct',help='uct/power-uct/rents/ments')
    parser.add_argument('--game', default='CartPole-v1', help='Training environment')
    parser.add_argument('--n_ep', type=int, default=500, help='Number of episodes')
    parser.add_argument('--n_mcts', type=int, default=512, help='Number of MCTS traces per step')
    parser.add_argument('--max_ep_len', type=int, default=300, help='Maximum number of steps per episode')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--c', type=float, default=1.5, help='UCT constant')
    parser.add_argument('--epsilon', type=float, default=.0, help='Epsilon')
    parser.add_argument('--p', type=float, default=1.0, help='Power constant')
    parser.add_argument('--temp', type=float, default=1.0, help='Temperature in normalization of counts to policy target')
    parser.add_argument('--gamma', type=float, default=1., help='Discount parameter')
    parser.add_argument('--data_size', type=int, default=1000, help='Dataset size (FIFO)')
    parser.add_argument('--batch_size', type=int, default=32, help='Minibatch size')
    parser.add_argument('--window', type=int, default=25, help='Smoothing window for visualization')

    parser.add_argument('--number', type=int, default=1, help='Iteration number')

    parser.add_argument('--n_hidden_layers', type=int, default=2, help='Number of hidden layers in NN')
    parser.add_argument('--n_hidden_units', type=int, default=128, help='Number of units per hidden layers in NN')

    
    args = parser.parse_args()
    episode_returns,timepoints,a_best,seed_best,R_best = agent(algorithm=args.algorithm,game=args.game,n_ep=args.n_ep,n_mcts=args.n_mcts,
                                        max_ep_len=args.max_ep_len,lr=args.lr,c=args.c,p=args.p,gamma=args.gamma,
                                        data_size=args.data_size,batch_size=args.batch_size,temp=args.temp,
                                        n_hidden_layers=args.n_hidden_layers,n_hidden_units=args.n_hidden_units,epsilon=args.epsilon)

    # Finished training
    filename = os.getcwd() +  '/' + args.game + '_puct.txt_' + str(args.number)
    file = open(filename,"w+")

    for reward in episode_returns:
        file.write(str(reward) + "\n")

    file.close()
