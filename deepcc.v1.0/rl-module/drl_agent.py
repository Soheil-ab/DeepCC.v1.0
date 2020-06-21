'''
MIT License
Copyright (c) 2019 - Chen-Yu Yen and Soheil Abbasloo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''

import numpy as np
import tensorflow as tf
import signal
import sys
import pickle

from utils import configure_logging

#import gym
from models import *
from memory import Memory
import time
import os
import sysv_ipc
import collections
from time import sleep
import argparse
#import random
tf.logging.set_verbosity(tf.logging.ERROR)
FORCE_ALPHA_INIT = 2     #if not "0",will force action to be FORCE_ALPHA*100

state_dim = 6
state_dim_extra = 1

final_state_dim = 5
# Number of inputs coming from rl-server.cc
input_dim = 7

rec_dim = 20
action_dim = 1
action_bound = 1

TARGET_MARGIN = 1

TEST_MEM_READ_WRITE = 0
MAX_EPISODES = 300

MAX_EP_STEPS = 5
MVWIN = 2
#MAX_EP_STEPS

PMWSIZE = 20
RENDER = 0
USEGYM = 1
MEMSIZE = 16*1e4 #1e4
RESTORE = 0
EVAL = 0
BATCHSIZE = 512
ZERO_DELAY = 10000000
ZERO_THRPT = 0.001

GAMMA = 0.995
TAU = 0.001
LR_A = 0.0001
LR_C = 0.001

# Noise_TYPE 0: OU noise (original version), whatever explore step
# Noise_TYPE 1: OU noise with decay by explore step, low noise after explore step
# Noise_TYPE 2: Gaussian noise with decay by explore step, no noise after explore step
# Noise_TYPE 3: Gaussian noise without decay
# Noise_TYPE 4: Gaussian noise with stepwise decay: after EXPLORE steps: sigma=NSTEP*sigma
# Noise_type 5: None

NOISE_TYPE = 3
EXPLORE = 4000
STDDEV = 0.1
NSTEP = 0.3
#memory1 = sysv_ipc.SharedMemory(123456)
#memory2 = sysv_ipc.SharedMemory(12345)

def log_parameters():
    logger.info("------------RL Training Hyper Parameters--------------")
    logger.info("LR_A: {}".format(LR_A))
    logger.info("LR_C: {}".format(LR_C))
    logger.info("tau: {}".format(TAU))
    logger.info("MAX_EP_STEP: {}".format(MAX_EP_STEPS))
    logger.info("Noise STDDEV: {}".format(STDDEV))
    logger.info("MEMSIZE: {}".format(MEMSIZE))
    logger.info("Batch_Size: {}".format(BATCHSIZE))
    logger.info("-------------------------------------------------------")
#a_glo = 13

def handler_term(signum, frame):
    # handle: pkill -15 -f p1.py
    #print(signum, frame)
    #if signum ==signal.SIGUSR1:

    print("python program terminated usking Kill -15")
    #logger.info("python program terminated usking Kill -15")
    # func()
    if not config.eval:
        terminated_save()
        normalizer.save_stats()
    #sess.close()
    sys.exit(1)


def handler_ctrlc(signum, frame):
    # handle: ctrl + c
    print("python program terminated using Ctrl+c")
    #logger.info("python program terminated using Ctrl+c")
    # func()
    if not config.eval:
        terminated_save()
        normalizer.save_stats()
    sess.close()
    sys.exit(1)


def terminated_save():
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(train_dir, 'model'))
    print("--------save checkpoint model at{}--------".format(train_dir))
    logger.info("--------save checkpoint model at{}--------".format(train_dir))
    ## save replay buffer
    with open(os.path.join(train_dir, "replay_memory.pkl"), "wb") as fp:
        pickle.dump(replay_memory, fp)
    print("--------save replay memory at{}---------".format(train_dir))
    logger.info("--------save replay memory at{}---------".format(train_dir))

signal.signal(signal.SIGTERM, handler_term)
signal.signal(signal.SIGINT, handler_ctrlc)

def eval_policy(env, learner):
    s0 = env.reset()
    ep_r = 0
    print("=======EVALUATION=========")
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()
        # a = learner.get_actor_out(s0)
        a, _ = learner.actor_step(s0)
        a = a[0]
        s1, r, done, _ = env.step(a)
        s0 = s1
        ep_r += r

    print("Total return in an episode:", ep_r)


def map_action_reverse(a):
    out =  math.log(a/100,2)
    return out

def map_action(a):
    out = math.pow(2,a)
    out *= 100
    out = int(out)
    return out
'''
def map_action(a):
    out =  (a - (-1))/2.0 * 9 + 1.0
    out *= 100
    out = int(out)
    return out
'''
class state():
    def __init__(self):
        self.reset()

    def reset(self):
        self.pre_samples = 0.0
        self.new_samples = 0.0
        self.avg_delay = 0.0
        self.avg_thr = 0.0
        self.thr_ = 0.0
        self.del_ = 0.0
        self.norm=Normalizer(1)
        self.del_moving_win = Moving_Win(MVWIN)
        self.thr_moving_win = Moving_Win(MVWIN)

    def get_state(self,memory, prev_rid,target,normalizer,evaluation=False):
        # [Output] delay: s0[0], thrpt: s0[1]
        succeed = False
        error_cnt=0
        #for cnt_ in range(100):
        while(1):
        # Read value from shared memory
            try:
                memory_value = memory.read()

            except sysv_ipc.ExistentialError:
                print("No shared memory Now, python ends gracefully :)")
                logger.info("No shared memory Now, python ends gracefully :)")
                sess.close()
                exit(1)

            memory_value = memory_value.decode('unicode_escape')

            #print("memory_value", memory_value)
            # Find the 'end' of the string and strip
            i = memory_value.find('\0')

            if i != -1:

                memory_value = memory_value[:i]
                #print("i:{}, memory_value{} ".format(i,memory_value))
                readstate = np.fromstring(memory_value, dtype=float, sep=' ')
                try:
                    rid = readstate[0]
                except :
                    rid = prev_rid
    #                print("rid waring")
                    continue
                try:
                    s0 = readstate[1:]
                except :
                    print("s0 waring")
                    continue

                #print(prev_rid, rid)

                if rid != prev_rid:
                    #prev_rid = rid
                    succeed = True
                    #print("Got the new state! finish reading "+str(rid))
                    break
                else:
                    wwwwww=""
        #            print("SAME ID, wait and read again "+str(rid))
            error_cnt=error_cnt+1
            if error_cnt > 1000:
#                print ("no new state given rid: "+str(rid)+" prev_rid: "+str(prev_rid)+" ****************** \n")
                error_cnt=0
            sleep(0.01)

        error_cnt=0
        if succeed == False:
            raise ValueError('read Nothing new from shrmem for a long time')
        reward=0
        state=np.zeros(1)
        w=s0
        if len(s0) == (input_dim):
#            if evaluation==True:
#                s0[0]=s0[0]*target/50.0
            d=s0[0]
            thr=s0[1]
            samples=s0[2]
            delta_t=s0[3]
            target_=s0[4]
            cwnd=s0[5]
            pacing_rate=s0[6]

            if evaluation!=True:
                normalizer.observe(s0)
            s0 = normalizer.normalize(s0)

            ############# Reward:
            min_ = normalizer.stats()
#            d_n=d
            d_n=s0[0]-min_[0]
            thr_n=s0[1]
            thr_n_min=s0[1]-min_[1]
            samples_n=s0[2]
            samples_n_min=s0[2]-min_[2]
            delta_t_n=s0[3]
            delta_t_n_min=s0[3]-min_[3]

            cwnd_n_min=s0[5]-min_[5]
            pacing_rate_n_min=s0[6]-min_[6]

            target_n_min=(normalizer.normalize_delay(target_*TARGET_MARGIN)-min_[0])

            if target_n_min!=0:
                delay_ratio=d_n/target_n_min
            else:
                delay_ratio=d_n

            ############# Reward3:
            self.pre_avg_delay = self.avg_delay
            self.del_moving_win.push(delay_ratio,samples)
            self.avg_delay = self.del_moving_win.get_avg()
            self.pre_thr = self.avg_thr
#            self.thr_moving_win.push(thr_n_min,1)
#            self.avg_thr = self.thr_moving_win.get_avg()
#            self.avg_thr = thr_n_min
            self.avg_thr = samples_n_min

            reward = self.avg_thr*thr_n_min*(delay_ratio)
            if reward>100.0:
                reward=100.0
            sign=1
            if self.avg_delay>1:
                reward= -reward
                sign=0

            state[0]=delay_ratio*(1-sign)
            state=np.append(state,[(1-delay_ratio)*sign])
            state=np.append(state,[samples_n_min*sign,thr_n_min*sign])
            state=np.append(state,[cwnd_n_min])
            #print ("d/T: "+str(state[0])+" avg-del: "+str(self.avg_delay)+" reward: "+str(reward)+" \n w[]: "+str(w)+"\n")

            return rid, state, d, reward, True
        else:
            return rid, state, 0.0, reward, False

class Moving_Win():
    def __init__(self,win_size):
        self.queue_main = collections.deque(maxlen=win_size)
        self.queue_aux = collections.deque(maxlen=win_size)
        self.length = 0
        self.avg = 0.0
        self.size = win_size
        self.total_samples=0

    def push(self,sample_value,sample_num):
        if self.length<self.size:
            self.queue_main.append(sample_value)
            self.queue_aux.append(sample_num)
            self.length=self.length+1
            self.avg=(self.avg*self.total_samples+sample_value*sample_num)
            self.total_samples+=sample_num
            if self.total_samples>0:
                self.avg=self.avg/self.total_samples
            else:
                self.avg=0.0
        else:
            pop_value=self.queue_main.popleft()
            pop_num=self.queue_aux.popleft()
            self.queue_main.append(sample_value)
            self.queue_aux.append(sample_num)
            self.avg=(self.avg*self.total_samples+sample_value*sample_num-pop_value*pop_num)
            self.total_samples=self.total_samples+(sample_num-pop_num)
            if self.total_samples>0:
                self.avg=self.avg/self.total_samples
            else:
                self.avg=0.0

    def get_avg(self):
        return self.avg

    def get_length(self):
        return self.length

class Normalizer():
    def __init__(self, num_inputs):
        self.n = 1e-5 #np.zeros(num_inputs)
        self.mean = np.zeros(num_inputs)
        self.mean_diff = np.zeros(num_inputs)
        self.var = np.zeros(num_inputs)
        self.dim = num_inputs
        self.min = np.zeros(num_inputs)

    def observe(self, x):
        self.n += 1
        last_mean = np.copy(self.mean)
        self.mean += (x-self.mean)/self.n
        self.mean_diff += (x-last_mean)*(x-self.mean)
        self.var = self.mean_diff/self.n

    def normalize(self, inputs):
        obs_std = np.sqrt(self.var)
        a=np.zeros(self.dim)
        if self.n > 2:
            a=(inputs - self.mean)/obs_std
            for i in range(0,self.dim):
                if a[i] < self.min[i]:
                    self.min[i] = a[i]
            return a
        else:
            return np.zeros(self.dim)

    def normalize_delay(self,delay):
        obs_std = math.sqrt(self.var[0])
        if self.n > 2:
            return (delay - self.mean[0])/obs_std
        else:
            return 0

    def stats(self):
        return self.min

    def save_stats(self):
        dic={}
        dic['n']=self.n
        dic['mean'] = self.mean.tolist()
        dic['mean_diff'] = self.mean_diff.tolist()
        dic['var'] = self.var.tolist()
        dic['min'] = self.min.tolist()
        import json
        with open(os.path.join(train_dir, 'stats.json'), 'w') as fp:
             json.dump(dic, fp)

        print("--------save stats at{}--------".format(train_dir))
        logger.info("--------save stats at{}--------".format(train_dir))



    def load_stats(self, file='stats.json'):
        import json

        with open(os.path.join(train_dir, file), 'r') as fp:
            history_stats = json.load(fp)
            #print(history_stats)
        self.n = history_stats['n']
        self.mean = np.asarray(history_stats['mean'])
        self.mean_diff = np.asarray(history_stats['mean_diff'])
        self.var = np.asarray(history_stats['var'])
        self.min = np.asarray(history_stats['min'])

    def adjust_state(self,target):
        self.mean[0] = self.mean[0]*50.0/target
        self.mean_diff[0] = self.mean_diff[0]*50.0/target
        self.var[0] = self.var[0]*50.0/target
        self.min[0] = self.min[0]*50.0/target
        print("Adjusting normalizer's values for D/Target by a="+str(50.0/target)+"\n")

def write_action(memory2, aciton, wid):
    #alpha = 13525
    msg = str(wid)+" "+str(aciton)+"\0"
    memory2.write(msg)

def main_tcp():
    global memory1
    global memory2
    #memory1 = sysv_ipc.SharedMemory(123456)
    #memory2 = sysv_ipc.SharedMemory(12345)
    global train_dir
    global sess
    global config
    global logger
    global normalizer
    global replay_memory

    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=float, default=1)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--tb_interval', type=int, default=10)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--scheme', type=str, default=None)
    parser.add_argument('--train_dir', type=str, default=None)
    parser.add_argument('--mem_r', type=int, default = 123456)
    parser.add_argument('--mem_w', type=int, default = 12345)

    config = parser.parse_args()

    if config.scheme is None:
        sys.exit('***********************Error: No Valid TCP is given *****************')

    if config.train_dir is None:
        sys.exit('***********************Error: Where is train_dir?! *****************')
    #print(config.target)

    memory1 = sysv_ipc.SharedMemory(config.mem_r)
    memory2 = sysv_ipc.SharedMemory(config.mem_w)


    # TCP env parameters:
    action_range = [-1.0, 1.0]
    prefix = 'v0'
    train_dir = './train_dir/%s-%s' % (
        prefix,
        time.strftime("%m%d-%H%M%S")
    )
    train_dir = str(config.train_dir)+'/train_dir-'+str(config.scheme)

    sess = tf.Session()
    global_step = tf.train.get_or_create_global_step(graph=None)
    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

    logger = configure_logging(train_dir)
    #logger.info("--------------------------------------------------------------------------------")
    #logger.info("--------------------------------------------------------------------------------")
    #logger.info("--------------------------------------------------------------------------------")
    #logger.info("--------------------------------------------------------------------------------")
    #logger.info("------------------------------Start Training !!!!!------------------------------")
    #logger.info("--------------------------------------------------------------------------------")

    actor = Actor(final_state_dim*rec_dim, action_dim, action_bound, "actor", use_gym=False)
    critic = Critic(final_state_dim*rec_dim, action_dim, action_bound, "critic", use_gym=False)
    #replay_memory = Memory(limit=int(MEMSIZE), action_shape=(action_dim,),
    #                observation_shape=(final_state_dim*rec_dim,))

    if config.load is not None and config.eval==False:
        if os.path.isfile(os.path.join(train_dir, "replay_memory.pkl")):
            with open(os.path.join(train_dir, "replay_memory.pkl"), 'rb') as fp:
                replay_memory = pickle.load(fp)
            print("--------load replay memory at{}---------".format(train_dir))
            logger.info("--------load replay memory at{}---------".format(train_dir))
        else:
            replay_memory = Memory(limit=int(MEMSIZE), action_shape=(action_dim,), observation_shape=(final_state_dim*rec_dim,))
    else:
        replay_memory = Memory(limit=int(MEMSIZE), action_shape=(action_dim,), observation_shape=(final_state_dim*rec_dim,))

    if NOISE_TYPE == 1:
        action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim), sigma=float(STDDEV) * np.ones(action_dim),dt=1,exp=EXPLORE)
    elif NOISE_TYPE == 2:
        ## Gaussian with gradually decay
        action_noise = GaussianActionNoise(mu=np.zeros(action_dim), sigma=float(STDDEV) * np.ones(action_dim), explore = EXPLORE)
    elif NOISE_TYPE == 3:
        ## Gaussian without gradually decay
        action_noise = GaussianActionNoise(mu=np.zeros(action_dim), sigma=float(STDDEV) * np.ones(action_dim), explore = None,theta=0.1)
    elif NOISE_TYPE == 4:
    ## Gaussian without gradually decay
        action_noise = GaussianActionNoise(mu=np.zeros(action_dim), sigma=float(STDDEV) * np.ones(action_dim), explore = EXPLORE,theta=0.1,mode="step",step=NSTEP)
    elif NOISE_TYPE == 5:
        action_noise = None
    else:
        action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim), sigma=float(STDDEV) * np.ones(action_dim),dt=0.5)
#        action_noise = None

    learner = Learner(sess, actor, critic, replay_memory, final_state_dim*rec_dim, 1, action_noise, batch_size=BATCHSIZE,
                    gamma=GAMMA, tau=TAU, lr_a=LR_A, lr_c=LR_C,
                    action_range=action_range, summary_writer=summary_writer)
    learner.initialize()

    saver = tf.train.Saver()

    summary_writer.add_graph(sess.graph)
    normalizer = Normalizer(input_dim)
    state_ = state()

    if config.load is not None:
        saver.restore(sess, os.path.join(train_dir, 'model'))
        #print("loaded from checkpoint!")
        #logger.info("load model from checkpoint!")

        #print("loaded stats from json")
        #logger.info("load stats from json!")
        normalizer.load_stats()

    #if EVAL:
    #    eval_policy(env, learner)
    #    return None
    #sleep(5)
    prev_rid = 99999
    target = config.target
    p_max_window = collections.deque(maxlen=PMWSIZE)
    p_max_window.append(0.0)
    wid = 23

    # start signal
    memory2.write(str(99999) + " " + str(99999) + "\0")
    #logger.info("ID 99999-------- RL module is ready")
    #logger.info("target:{}, PMWSIZE:{}".format(target, PMWSIZE))
    #logger.info("action, delay, throughput, reward")

    zero_delay_counter = 0

    if config.eval==True:
        learner.action_noise = None
#        normalizer.adjust_state(target)

    s0_rec_buffer = np.zeros([final_state_dim*rec_dim])
    s1_rec_buffer = np.zeros([final_state_dim*rec_dim])
    prev_rid, s0, delay_,rew0,error_code =state_.get_state(memory1, prev_rid,target,normalizer)
    s0_rec_buffer[-1*final_state_dim:] = s0
    if error_code == True:
        a0, _ = learner.actor_step(s0_rec_buffer)
        a0 = a0[0]
        map_a0 = map_action(a0)
    else:
        #Using previous action:
        map_a0 = 10
    write_action(memory2, map_a0, wid)
    wid = (wid + 1) % 1000


    step_counter = np.int64(0)
    episode_counter = np.int64(0)

#### evaluation ####
# Evaluation the trained model
# usage: --load=1 --eval
    while(config.eval==True):
        #logger.info("---------Evaluation--------------------------")
        episode_counter += 1
        start_time = time.time()

        done = False
        ep_r = 0.
        log_buffer = []

        for j in range(MAX_EP_STEPS):
            step_counter += 1

            if j == MAX_EP_STEPS-1:
                done = True

            prev_rid, s1,delay_,rew0,error_code = state_.get_state(memory1, prev_rid,target,normalizer,config.eval)
            if error_code == True:
                s1_rec_buffer = np.concatenate( (s0_rec_buffer[final_state_dim:], s1) )
                a1, _ = learner.actor_step(s1_rec_buffer)
            else:
                #Using previous action:
                wid = (wid + 1) % 1000
                write_action(memory2, map_a0, wid)
                continue

            a1 = a1[0]
            map_a1 = map_action(a1)

            write_action(memory2, map_a1, wid)

            if (j+1) % config.tb_interval == 0:
                # tensorboard
                act_summary = tf.Summary()
                act_summary.value.add(tag='Step/0-Actions-Eval', simple_value=map_a0)
                for i in range(0,len(s1)):
                    act_summary.value.add(tag='Step/1-Input-Eval'+str(i), simple_value=s1[i])
                act_summary.value.add(tag='Step/2-Reward-Eval', simple_value=rew0)
                if learner.action_noise!=None:
                    act_summary.value.add(tag='Step/3-Noise-Eval', simple_value=map_action(learner.action_noise.show()))
                summary_writer.add_summary(act_summary, step_counter)

            if 1:  #(j+1)%10 == 0:
                log_buffer = []

            a0 = a1
            s0 = s1
            s0_rec_buffer = s1_rec_buffer

            map_a0 = map_a1
            wid = (wid + 1) % 1000

            ep_r += rew0

            if done:
                ret_summary = tf.Summary()
                ret_summary.value.add(tag='Performance/ReturnInEpisode-Eval', simple_value=ep_r)
                summary_writer.add_summary(ret_summary, episode_counter)
                break

        duration = time.time() - start_time

##### training####
    log_parameters()
    if config.load==None:
        FORCE_ALPHA=FORCE_ALPHA_INIT
    else:
        FORCE_ALPHA=0

    while 1:
        episode_counter += 1
        start_time = time.time()



        done = False
        ep_r = 0.
        log_buffer = []

        for j in range(MAX_EP_STEPS):
            step_counter += 1

            if j == MAX_EP_STEPS-1:
                done = True

            prev_rid, s1,delay_,rew0,error_code = state_.get_state(memory1, prev_rid,target,normalizer,config.eval)
            if error_code == True:
                s1_rec_buffer = np.concatenate( (s0_rec_buffer[final_state_dim:], s1))
                a1, _ = learner.actor_step(s1_rec_buffer)
            else:
                #Using previous action:
                wid = (wid + 1) % 1000
                write_action(memory2, map_a0, wid)
                continue

            a1 = a1[0]
            map_a1 = map_action(a1)
            if FORCE_ALPHA>0:
                if (step_counter%1000)==0:
                    FORCE_ALPHA-=0.1
                if FORCE_ALPHA<0.5:
                    FORCE_ALPHA=0
                else:
                    map_a1=FORCE_ALPHA*100
                    a1 = map_action_reverse(map_a1)

            write_action(memory2, map_a1, wid)
            learner.store_transition(s0_rec_buffer, a0, rew0, s1_rec_buffer, done)

            if (j+1) % config.tb_interval == 0:
                # tensorboard
                act_summary = tf.Summary()
                act_summary.value.add(tag='Step/0-Actions', simple_value=map_a0)
                for i in range(0,len(s1)):
                    act_summary.value.add(tag='Step/1-Input'+str(i), simple_value=s1[i])
                act_summary.value.add(tag='Step/2-Reward', simple_value=rew0)
                if learner.action_noise!=None:
                    act_summary.value.add(tag='Step/3-Noise', simple_value=map_action(learner.action_noise.show()))
                summary_writer.add_summary(act_summary, step_counter)

            if 1:  #(j+1)%10 == 0:
                log_buffer = []

            if episode_counter >= 0:
                learner.train_step()
                learner.update_target()


            a0 = a1
            s0 = s1
            s0_rec_buffer = s1_rec_buffer

            map_a0 = map_a1
            wid = (wid + 1) % 1000

            ep_r += rew0

            if done:
                ret_summary = tf.Summary()
                ret_summary.value.add(tag='Performance/ReturnInEpisode', simple_value=ep_r)
                summary_writer.add_summary(ret_summary, episode_counter)
                break

        duration = time.time() - start_time
    terminated_save()

if __name__ == '__main__':
    main_tcp()
