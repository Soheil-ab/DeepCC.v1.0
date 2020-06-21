'''
MIT License
Copyright (c) 2019 - Chen-Yu Yen

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

##
# Define Actor and Critic class here
#
##

from copy import copy
import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
import math
H1_SHAPE = 1000
H2_SHAPE = 1000


def create_input_op_shape(obs, tensor):
    input_shape = [x or -1 for x in tensor.shape.as_list()]
    return np.reshape(obs, input_shape)


def fc(input, output_shape, name='fc'):
    with tf.variable_scope(name):
        in_shape = input.get_shape()[-1]
        w = tf.get_variable('w', [in_shape, output_shape],
                            initializer=tf.contrib.layers.variance_scaling_initializer())
        # relu: tf.contrib.layers.variance_scaling_initializer()
        # tanh: tf.contrib.layers.xavier_initializer()
        b = tf.get_variable('b', [output_shape], initializer=tf.constant_initializer(0.0))
        return tf.matmul(input, w) + b

def target_init(target, vars):

    return [tf.assign(target[i], vars[i]) for i in range(len(vars))]

def target_update(target, vars, tau):
    # update target network
    return [tf.assign(target[i], vars[i]*tau + target[i]*(1-tau)) for i in range(len(vars))]



class Learner(object):
    def __init__(self, sess, actor, critic, memory, state_dim, action_dim, action_noise, batch_size=32,
#                 gamma=0.99, tau=0.01, lr_a=0.001, lr_c=0.002, action_range=(-1., 1.), summary_writer=None):
#   actor'a rl is 10x smaller in the paper (0.001)
                  gamma=0.9, tau=0.1, lr_a=0.01, lr_c=0.01, action_range=(-1., 1.), summary_writer=None):
        self.sess = sess
        self.s_dim = state_dim
        self.actor = actor
        self.critic = critic
        self.memory = memory
        self.tau = tau
        self.batch_size = batch_size
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.training_steps = 0
        self.action_noise = action_noise
        self.action_range = action_range
        self.summary_writer = summary_writer
        self.s0 = tf.placeholder(tf.float32, shape=[None, self.s_dim], name='s0')
        self.s1 = tf.placeholder(tf.float32, shape=[None, self.s_dim], name='s1')
        self.reward = tf.placeholder(tf.float32, shape=[None, 1], name='reward')
        self.action = tf.placeholder(tf.float32, shape=[None, action_dim], name='action')
        self.is_terminal = tf.placeholder(tf.float32, shape=[None, 1], name='is_terminal')
        self.critic_target = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target')

        self.actor_optimizer = tf.train.AdamOptimizer(self.lr_a)
        self.critic_optimizer = tf.train.AdamOptimizer(self.lr_c)
        self.is_training = tf.placeholder(tf.bool, name='Actor_is_training')
        self.is_training_c = tf.placeholder(tf.bool, name='Critic_is_training')
        # Actor
        target_actor = copy(actor)
        self.target_actor = target_actor
        self.target_actor.name = 'target_actor'
        #self.target_actor.training = False

        self.actor_out, self.scaled_actor_out = actor.create_actor_network(self.s0, self.is_training)

        # create target network
        self.target_actor_out_s1, self.target_scaled_actor_out_s1 = self.target_actor.create_actor_network(self.s1, self.is_training)

        self.target_actor_update_init = target_init(self.target_actor.trainable_vars(), self.actor.trainable_vars())
        self.target_actor_update = target_update(self.target_actor.trainable_vars(), self.actor.trainable_vars(), self.tau)


        ###
        # Critic
        # Q(s, action)
        target_critic = copy(critic)
        self.target_critic = target_critic
        self.target_critic.name = 'target_critic'

        self.critic_out = critic.create_critic_network(self.s0, self.action, self.is_training_c)
        # Q(s, actor)
        self.critic_actor = critic.create_critic_network(self.s0, self.scaled_actor_out, self.is_training, reuse=True)

        self.q_s1 = self.target_critic.create_critic_network(self.s1, self.target_scaled_actor_out_s1, self.is_training_c)
        self.target_critic_update_init = target_init(self.target_critic.trainable_vars(), self.critic.trainable_vars())
        self.target_critic_update = target_update(self.target_critic.trainable_vars(), self.critic.trainable_vars(),
                                                 self.tau)

        self.target_q = self.reward + (1. - self.is_terminal) * gamma * self.q_s1
        self.target_q_debug =  gamma * self.q_s1

        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        #with tf.control_dependencies(self.extra_update_ops):
        self.actor_train_op = self.build_actor_train_op()
        self.critic_train_op = self.build_critic_train_op()

        if self.summary_writer is not None:
            # All tf.summaries should have been defined prior to running this.
            self._merged_summaries = tf.summary.merge_all()


    def get_target_actor_out(self, obs):
        return self.sess.run(self.target_scaled_actor_out_s1,
                             feed_dict={self.s1: create_input_op_shape(obs, self.s1)})

    def update_target(self):
        self.sess.run([self.target_actor_update, self.target_critic_update])


    def build_actor_train_op(self):


        self.a_loss = -tf.reduce_mean(self.critic_actor)

        #self.a_gradients = tf.gradients(self.a_loss, self.actor.trainable_vars() )
        self.a_gradients = self.actor_optimizer.compute_gradients(self.a_loss, var_list=self.actor.trainable_vars())
        #print(self.actor.trainable_vars())
        if self.summary_writer is not None:
            with tf.variable_scope('Losses'):
                a_loss_summary = tf.summary.scalar('ActorLoss', self.a_loss)
                for index, grad in enumerate(self.a_gradients):
                    a_grad_summary = tf.summary.histogram("{}-grad".format(self.a_gradients[index][1].name), self.a_gradients[index])

                self._merged_summaries_actor = tf.summary.merge([a_loss_summary, a_grad_summary])
#                tf.summary.histogram('Actor_grad', self.a_gradients)

        #self.a_gradients = list(zip(grads, self.actor.trainable_vars()))
        #return self.actor_optimizer.minimize(self.a_loss, var_list=self.actor.trainable_vars())
        a_op = self.actor_optimizer.apply_gradients(self.a_gradients)
        return a_op

    '''
    def build_critic_train_op(self):
        self.c_loss = tf.reduce_mean(tf.square(self.critic_out - self.critic_target))

        if self.summary_writer is not None:
            with tf.variable_scope('Losses'):
                tf.summary.scalar('CriticLoss', self.c_loss)

        return tf.train.AdamOptimizer(self.lr_c).minimize(self.c_loss, var_list=self.critic.trainable_vars())
    '''

    def build_critic_train_op(self):
        # L2 regularization
        #[print(x) for x in self.critic.trainable_vars()]
        #[print(x) for x in self.critic.trainable_vars() if (x.name.endswith('w:0') or x.name.endswith('kernel:0')) ]
        #with tf.control_dependencies(self.extra_update_ops):
        self.c_loss = tf.reduce_mean(tf.square(self.critic_out - self.target_q))

        reg_var = [x for x in  self.critic.trainable_vars() if (x.name.endswith('w:0') or x.name.endswith('kernel:0'))]
        reg_loss =  tc.layers.apply_regularization( tc.layers.l2_regularizer(1e-2), weights_list = reg_var  )
        self.c_loss += reg_loss

        if self.summary_writer is not None:
            with tf.variable_scope('Losses'):
                c_loss_summary = tf.summary.scalar('CriticLoss', self.c_loss)
            self._merged_summaries_critic = tf.summary.merge([c_loss_summary])

        self.c_gradients = self.critic_optimizer.compute_gradients(self.c_loss, var_list=self.critic.trainable_vars())
        #for i, (grad, var) in enumerate(self.c_gradients):
        #    if grad is not None:
        #        #print(grad)
        #        self.c_gradients[i] = (tf.clip_by_norm(grad, 1), var)
        #self.c_gradients = tf.gradients(self.c_loss, self.critic.trainable_vars())
        #return self.critic_optimizer.minimize(self.c_loss, var_list=self.critic.trainable_vars())
        return self.critic_optimizer.apply_gradients(self.c_gradients)

    def initialize(self):

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.target_actor_update_init)
        self.sess.run(self.target_critic_update_init)

        #self.sess.run(extra_update_ops)

    def get_actor_out(self, obs):
        return  self.sess.run(self.scaled_actor_out,
                            feed_dict={self.s0: create_input_op_shape(obs, self.s0)})


    #def get_target_actor_out(self, obs):
    #    out = self.sess.run(self.target_scaled_actor_out,
    #                        feed_dict={self.s0: create_input_op_shape(obs, self.s0)})
    #    return out

    def get_critic_out(self, obs, action):
        return self.sess.run(self.critic_out, feed_dict={self.s0: create_input_op_shape(obs, self.s0),
                                                           self.action: create_input_op_shape(action, self.action)})

    def get_critic_actor_out(self, obs):
        return self.sess.run(self.critic_actor, feed_dict={self.s0: create_input_op_shape(obs, self.s0),  self.is_training:False, self.is_training_c:False})

    def get_q_and_target_q(self, s0, action, r, s1, is_terminal):

        target_Q, target_Q_debug = self.sess.run([self.target_q, self.target_q_debug], feed_dict={
            self.s1: create_input_op_shape(s1, self.s1),
            self.reward: create_input_op_shape(r, self.reward),
            self.is_terminal: create_input_op_shape(is_terminal, self.is_terminal)
        })
        q_sa = self.get_critic_out(s0, action)

        return q_sa, target_Q, target_Q_debug

    def store_transition(self, s0, action, reward, s1, is_terminal):

        self.memory.append(s0, action, reward, s1, is_terminal)
        #for b in range(B):
        #    self.memory.append(s0[b], action[b], reward[b], s1[b], is_terminal[b])

    def train_step(self):
        #print("Hello:", tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        extra_update_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if "actor" in v.name and "target" not in v.name]
        #extra_update_ops_c = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if "critic/" in v.name and "target" not in v.name]
        #extra_update_ops_c1 = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if "critic_1" in v.name and "target" not in v.name]
        #extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #print("extra_update_ops: ", extra_update_ops)
        #print("extra_update_ops_c: ", extra_update_ops_c)
        batch = self.memory.sample(batch_size=self.batch_size)

        #a_gradients , c_gradients, summary = self.sess.run([self.a_gradients, self.c_gradients, self._merged_summaries], feed_dict={self.s0: batch['obs0'], self.action: batch['actions'], self.reward: batch['rewards'], self.s1: batch['obs1'], self.is_terminal:batch['terminals1']})

        #x = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        #for x in x:
        #    print(x)
        #input("kasjdlksa")

        #a_gradients = self.actor_optimizer.compute_gradients(a_loss, var_list=self.actor.trainable_vars())
        #c_gradients = self.critic_optimizer.compute_gradients(c_loss, var_list=self.critic.trainable_vars())
        #self.actor_optimizer.apply_gradients(a_gradients)
        #self.critic_optimizer.apply_gradients(c_gradients)

        _,  summary_critic = self.sess.run([self.critic_train_op,  self._merged_summaries_critic], feed_dict={self.s0: batch['obs0'], self.action: batch['actions'], self.reward:batch['rewards'], self.s1: batch['obs1'], self.is_terminal:batch['terminals1'],  self.is_training:True, self.is_training_c:True})
        _, _, summary_actor = self.sess.run([ self.actor_train_op, extra_update_ops, self._merged_summaries_actor ], feed_dict={self.s0:batch['obs0'], self.is_training:True, self.is_training_c:True })
        #summary = self.sess.run([self._merged_summaries], feed_dict={self.s0: batch['obs0'], self.action: batch['actions'], self.reward:batch['rewards'], self.s1: batch['obs1'], self.is_terminal:batch['terminals1']})
        #_ = self.sess.run([ self.actor_train_op ], feed_dict={self.s0:batch['obs0']})

        #_,_, summary = self.sess.run([self.critic_train_op, self.actor_train_op, self._merged_summaries], feed_dict={self.s0: batch['obs0'], self.action: batch['actions'], self.reward: batch['rewards'], self.s1: batch['obs1'], self.is_terminal: batch['terminals1']})
        #print(summary)
        #if self.summary_writer is not None:
        #    with tf.variable_scope('Losses'):
        #        tf.summary.scalar('CriticLoss', c_loss)


        '''
        target_Q = self.sess.run(self.target_q, feed_dict={
            self.s1: batch['obs1'],
            self.reward: batch['rewards'],
            self.is_terminal: batch['terminals1'].astype('float32'),
        })
        _, _, summary = self.sess.run([self.critic_train_op, self.actor_train_op, self._merged_summaries],
                      feed_dict={self.s0: batch['obs0'], self.action: batch['actions'], self.critic_target:target_Q, self.s1: batch['obs1']})
        '''

        self.training_steps += 1

        #summary = self.sess.run(self._merged_summaries)
        self.summary_writer.add_summary(summary_critic, self.training_steps)
        self.summary_writer.add_summary(summary_actor, self.training_steps)



    def actor_step(self, obs):
        action = self.sess.run(self.scaled_actor_out,
                             feed_dict={self.s0: create_input_op_shape(obs, self.s0), self.is_training:False})
        #print("action:", action)
        if self.action_noise is not None:
            noise = self.action_noise(action[0])
            assert noise.shape == action[0].shape
            action += noise
        action = np.clip(action, self.action_range[0], self.action_range[1])
        #print("actionoise:", action)
        q_sa = self.get_critic_actor_out(obs)
        return action, q_sa

    def reset(self):
        if self.action_noise is not None:
            self.action_noise.reset()

class Actor(object):
    def __init__(self, state_dim, action_dim, action_bound=1, name=None, training=True, use_gym=1):
        #self.sess = sess
        self.s_dim = state_dim
        self.action_dim = action_dim
        self.name = name
        self.learning_rate = 0.001
        self.action_bound = action_bound
        self.use_gym = use_gym
        self.training = training
        #self.obs0 = tf.placeholder(tf.float32, shape=[None, self.s_dim], name='obs0')

    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    def create_actor_network(self, obs, is_training, reuse=False):
        #inputs = tf.placeholder(tf.float32, [None, self.s_dim])

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            h1 = fc(obs, H1_SHAPE, name='fc1')
            h1 = tf.layers.batch_normalization(h1, training = is_training, scale=False)
            h1 = tf.nn.relu(h1)
            h2 = fc(h1, H2_SHAPE, name='fc2')
            h2 = tf.layers.batch_normalization(h2, training = is_training, scale=False)
            h2 = tf.nn.relu(h2)

            #out = tf.layers.dense(h2, self.action_dim, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            #out = tf.layers.dense(h2, self.action_dim, kernel_initializer=tf.contrib.layers.xavier_initializer())
            out = tf.layers.dense(h2, self.action_dim)#, kernel_initializer=tf.contrib.layers.xavier_initializer())
            #out = tf.layers.batch_normalization(out, training = is_training)
            out = tf.nn.tanh(out)

            # scale the output here, [-action_bound, action_bound]
            scaled_out = tf.multiply(out, self.action_bound)

        return out, scaled_out


class Critic(object):
    def __init__(self, state_dim, action_dim, action_bound=1, name=None, training=True, use_gym=1):
        self.s_dim = state_dim
        self.action_dim = action_dim
        self.name = name
        self.action_bound = action_bound
        self.use_gym = use_gym
        self.training = training

        # self.obs0 = tf.placeholder(tf.float32, shape=[None, self.s_dim], name='obs0')

    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    def create_critic_network(self, obs, action, is_training, reuse=False):
        # inputs = tf.placeholder(tf.float32, [None, self.s_dim])

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            '''
            h1 = fc(tf.concat([obs, action], axis=-1), H1_SHAPE, name='fc1')
            h1 = tf.nn.relu(h1)
            #tf.summary.histogram(tf.get_variable_scope().name, h1)
            h2 = fc(h1, H2_SHAPE, name='fc2')
            h2 = tf.nn.relu(h2)
            '''
            h1 = fc(obs, H1_SHAPE, name='fc1')
            #h1 = tf.layers.batch_normalization(h1, training = is_training, scale=False)
            #h1 = tf.contrib.layers.layer_norm(h1)
            h1 = tf.nn.relu(h1)

            h2 = fc(tf.concat([h1, action], axis=-1), H2_SHAPE, name='fc2')


            h2 = tf.nn.relu(h2)

            #w2_s = tf.get_variable('w2_s_w', [H1_SHAPE, H2_SHAPE])
            #w2_a = tf.get_variable('w2_a_w', [1, H2_SHAPE])
            #b2 = tf.get_variable('b', [H2_SHAPE])
            #h2 = tf.nn.relu(tf.matmul(h1, w2_s) + tf.matmul(action, w2_a) + b2)

            #out = tf.layers.dense(h2, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            out = tf.layers.dense(h2, 1)
        return out


class OrnsteinUhlenbeckActionNoise(object):
#    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
    def __init__(self, mu, sigma, theta=.15, dt=0.01, x0=None,exp=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.eps = 1.0
        self.exp = exp
        self.reset()

    def show(self):
        return self.x_prev

    def __call__(self,point):
        if self.exp!=None:
            self.dt -= 1/self.exp
            if self.dt<=0.01:
                self.dt=0.01
            self.sigma -= 1/self.exp
            if self.sigma<=0.3:
                self.sigma=0.3
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
#        x = x*self.eps
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class GaussianActionNoise(object):
    def __init__(self, mu , sigma, explore=40000,theta=0.1,mu2=0.0,mode="exp",eps=1.0,step=0.3):
        self.epsilon = eps
        self.mu = mu
        self.explore = explore
        self.sigma = sigma
        self.mu2 = mu2
        self.theta = theta
        self.noise = 0
        self.cnt = 0
        self.step = step
        self.mode = mode

    def show(self):
        return self.noise

    def __call__(self,point):
        if self.explore!=None:
            if self.mode=="exp":
                if self.epsilon <= 0:
                    self.noise=np.zeros_like(self.mu)
                else:
                    self.epsilon -= 1/self.explore
#                    noise = self.epsilon * (self.theta*(self.mu2-point)+self.sigma * np.random.randn(1))
                    noise = self.epsilon * (self.sigma * np.random.randn(1))
                    self.noise = noise
            else:
                self.cnt += 1
                if self.cnt >=self.explore:
                    self.sigma -= self.step*self.sigma
                    self.cnt = 0
                if self.sigma <= 0.2:
                    self.segma = 0.2
#                noise = self.theta*(self.mu2-point)+self.sigma*np.random.randn(1)
                noise = self.sigma*np.random.randn(1)
                self.noise = noise
        else:
#            noise = (self.theta*(self.mu2-point)+self.sigma * np.random.randn(1))
            noise = (self.sigma * np.random.randn(1))
            self.noise = noise

        return self.noise

    def reset(self):
        pass

