import tensorflow as tf
import cv2
import numpy as np
import random
import config
from dqn.memory import Memory
from functools import reduce


class Agent:
    def __init__(self, env, sess):
        self.env = env
        self.lr = config.learning_rate
        self.gamma = config.discount
        self.memory_size = config.memory_size
        self.batch_size = config.batch_size
        self.initial_epsilon = config.initial_epsilon
        self.final_epsilon = config.final_epsilon
        self.tn_update_freq = config.tn_update_freq
        self.epsilon = self.initial_epsilon
        self.learn_start = config.learn_start
        self.explore = config.explore
        self.memory = Memory()
        self.history_length = config.history_length
        self.step = 0
        self.episode = 1
        self.episode_reward = 0
        self.sl_update_count = 0
        self.frame_per_action = config.frame_per_action
        self.sn_frequency = config.save_network_frequency
        self.sv_frequency = config.sv_frequency
        self.sl_frequency = config.sl_frequency
        self.learning_rate = config.learning_rate
        self.learning_rate_minimum = config.learning_rate_minimum
        self.learning_rate_decay = config.learning_rate_decay
        self.learning_rate_decay_step = config.learning_rate_decay_step
        self.gradient_momentum = config.gradient_momentum
        self.proceed_train = config.proceed_train
        self.train_frequency = config.train_frequency
        self.mergelist = []
        self.s, self.q_value, self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w_fc1, self.b_fc1, self.w_fc2, self.b_fc2 = self.buildNetwork()
        self.sT, self.q_valueT, self.w1T, self.b1T, self.w2T, self.b2T, self.w3T, self.b3T, self.w_fc1T, self.b_fc1T, self.w_fc2T, self.b_fc2T = self.buildNetwork(isTargetNetwork=True)
        self.copyTargetQNetworkOperation = [self.w1T.assign(self.w1), self.b1T.assign(self.b1),
                                            self.w2T.assign(self.w2), self.b2T.assign(self.b2),
                                            self.w3T.assign(self.w3), self.b3T.assign(self.b3),
                                            self.w_fc1T.assign(self.w_fc1), self.b_fc1T.assign(self.b_fc1),
                                            self.w_fc2T.assign(self.w_fc2), self.b_fc2T.assign(self.b_fc2)]
        self.sess = sess
        self.minreward = 0
        self.maxreward = 0
        self.sumOfreward = 0
        self.sumOfloss = 0
        self.update_reward_freq = config.update_reward_freq
        with tf.variable_scope("Reward"):
            self.VarofAvgReward = tf.placeholder(tf.float32)
            self.summaryAvgReward = tf.summary.scalar("AvgReward", self.VarofAvgReward)

            self.VarofMaxReward = tf.placeholder(tf.int32)
            self.summaryMaxReward = tf.summary.scalar("MaxReward", self.VarofMaxReward)

            self.VarofMinReward = tf.placeholder(tf.int32)
            self.summaryMinReward = tf.summary.scalar("MinReward", self.VarofMinReward)

        with tf.variable_scope("Epsilon"):
            self.VarOfEpsilon = tf.placeholder(tf.float32)
            self.SummaryEpsilon = tf.summary.scalar("Epsilon", self.VarOfEpsilon)

        with tf.variable_scope("Action"):
            self.VarOfAction = tf.placeholder(tf.int16)
            self.SummaryAction = tf.summary.histogram("Action", self.VarOfAction)

        self.CreateTrainingMethod()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge(self.mergelist)

        self.train_writer = tf.summary.FileWriter('Data/', self.sess.graph)
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        self.saver = tf.train.Saver(max_to_keep=1)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded")
        else:
            print("Could not find old network weights")

        self.copyTargetQNetworkOperation = [self.w1T.assign(self.w1), self.b1T.assign(self.b1),
                                            self.w2T.assign(self.w2), self.b2T.assign(self.b2),
                                            self.w3T.assign(self.w3), self.b3T.assign(self.b3),
                                            self.w_fc1T.assign(self.w_fc1), self.b_fc1T.assign(self.b_fc1),
                                            self.w_fc2T.assign(self.w_fc2), self.b_fc2T.assign(self.b_fc2)]

    def buildNetwork(self, isTargetNetwork=False):
        width = self.env.screen_width
        height = self.env.screen_height
        #NCHW
        s = tf.placeholder(tf.float32, [None, self.history_length, height, width], name='s')
        mergelist = []
        if (isTargetNetwork):
            name = 'TargetNetwork'
        else:
            name = 'Q-Network'
        with tf.variable_scope(name):
            with tf.variable_scope('l1'):
                w1 = self.weight_variable([8, 8, 4, 32])
                b1 = self.bias_variable([32])
                l1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(s, w1, strides=[1, 1, 4, 4], data_format='NCHW',padding='VALID'),b1,data_format='NCHW'))

            with tf.variable_scope('l2'):
                w2 = self.weight_variable([4, 4, 32, 64])
                b2 = self.bias_variable([64])
                l2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l1, w2, strides=[1, 1, 2, 2], data_format='NCHW',padding='VALID'),b2,data_format='NCHW'))

            with tf.variable_scope('l3'):
                w3 = self.weight_variable([3, 3, 64, 64])
                b3 = self.bias_variable([64])
                l3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], data_format='NCHW',padding='VALID'), b3,data_format='NCHW'))
            shape = l3.get_shape().as_list()
            l3_flat = tf.reshape(l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

            with tf.variable_scope('fc1'):
                shape = l3_flat.get_shape().as_list()
                w_fc1 = self.weight_variable([shape[1], 512])
                b_fc1 = self.bias_variable([512])
                h_fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(l3_flat, w_fc1), b_fc1))

            with tf.variable_scope('fc2'):
                shape = h_fc1.get_shape().as_list()
                w_fc2 = self.weight_variable([shape[1], self.env.action_size])
                b_fc2 = self.bias_variable([self.env.action_size])
                q_value = tf.nn.bias_add(tf.matmul(h_fc1, w_fc2), b_fc2)
        if (isTargetNetwork is False):
            mergelist.append(self.variable_summaries(w1, "w1"))
            mergelist.append(self.variable_summaries(b1, "b1"))
            mergelist.append(self.variable_summaries(w2, "w2"))
            mergelist.append(self.variable_summaries(b2, "b2"))
            mergelist.append(self.variable_summaries(w3, "w3"))
            mergelist.append(self.variable_summaries(b3, "b3"))
            mergelist.append(self.variable_summaries(w_fc1, "w_fc1"))
            mergelist.append(self.variable_summaries(b_fc1, "b_fc1"))
            mergelist.append(self.variable_summaries(w_fc2, "w_fc2"))
            mergelist.append(self.variable_summaries(b_fc2, "b_fc2"))
            for varinfo in mergelist:
                for info in varinfo:
                    self.mergelist.append(info)
        return s, q_value, w1, b1, w2, b2, w3, b3, w_fc1, b_fc1, w_fc2, b_fc2

    def copyTargetQNetwork(self):
        self.sess.run(self.copyTargetQNetworkOperation)
    def CreateTrainingMethod(self):
        self.actionInput = tf.placeholder("float", [None, self.env.action_size])
        self.yInput = tf.placeholder("float", [None])

        Q_Action = tf.reduce_sum(tf.multiply(self.q_value, self.actionInput), reduction_indices=1)

        self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
        self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
                                           tf.train.exponential_decay(
                                               self.learning_rate,
                                               self.learning_rate_step,
                                               self.learning_rate_decay_step,
                                               self.learning_rate_decay,
                                               staircase=True))
        self.loss = tf.reduce_mean(self.clipped_error(self.yInput - Q_Action))

        #self.VarOflearning_rate = tf.placeholder(tf.float32)
        self.SummaryLearningRate = tf.summary.scalar('LearningRate', self.learning_rate_op)
        with tf.name_scope("Loss"):
            self.summaryLoss = tf.summary.scalar('loss', self.loss)
            self.VarOfAvgLoss = tf.placeholder(tf.float32)
            self.summaryAvgLoss = tf.summary.scalar('AvgLoss', self.VarOfAvgLoss)

        self.trainStep = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def learn(self, screen, action, reward, terminal):
        self.step += 1
        ac = self.sess.run(self.SummaryAction,feed_dict={self.VarOfAction:action})
        self.train_writer.add_summary(ac, self.step)

        if self.memory.count > self.memory_size:
            self.memory.popleft()

        if(self.env.display):
            self.env.render()

        if((self.step > self.learn_start) and self.proceed_train):
            if(self.step % self.train_frequency ==0):
                self.train()
        if self.step < self.learn_start:
            state = 'observe'
        elif self.step > self.learn_start and self.step <= self.explore:
            state = 'explore'
        else:
            state = 'train'
        print("TIMESTEP", self.step, "/ STATE", state, "/ACTION", action,"/REWARD", reward,"/ EPSILON", self.epsilon)

        if self.step % self.sv_frequency == 0:
            print("Saving Network Variables...")
            result = self.sess.run(self.merged)
            self.train_writer.add_summary(result, self.step)

        screen = self.preprocess(screen)
        newState = np.append(self.currentState[1:, :, :], screen)
        newState = newState.reshape((self.history_length, self.env.screen_height, self.env.screen_width))
        self.memory.add(self.currentState, action, reward, terminal)
        self.currentState = newState.copy()
        self.episode_reward += reward
        if(terminal):
            print("Saving Episode Reward...")
            if(self.step % self.update_reward_freq == 0):
                self.minreward = self.episode_reward
                self.maxreward = self.episode_reward
            else:
                if(self.episode_reward < self.minreward):
                    self.minreward = self.episode_reward
                if(self.episode_reward > self.maxreward):
                    self.maxreward = self.episode_reward

            self.sumOfreward += self.episode_reward

            avgreward = self.sess.run(self.summaryAvgReward,
                                    feed_dict={self.VarofAvgReward: self.sumOfreward / self.episode})
            self.train_writer.add_summary(avgreward, self.step)

            maxreward = self.sess.run(self.summaryMaxReward,
                                    feed_dict={self.VarofMaxReward: int(self.maxreward)})
            self.train_writer.add_summary(maxreward, self.step)

            minreward = self.sess.run(self.summaryMinReward,
                                    feed_dict={self.VarofMinReward: int(self.minreward)})
            self.train_writer.add_summary(minreward, self.step)

            epsilon = self.sess.run(self.SummaryEpsilon,
                                    feed_dict={self.VarOfEpsilon: self.epsilon})
            self.train_writer.add_summary(epsilon, self.step)

            learning_rate = self.sess.run(self.SummaryLearningRate,
                                    feed_dict={self.learning_rate_step: self.step})
            self.train_writer.add_summary(learning_rate, self.step)

            self.episode += 1
            self.episode_reward = 0
            self.env.reset()


    def train(self):
        minibatch = self.memory.sample(self.batch_size)
        state_batch = [minibatch[i][0] for i in range(self.batch_size)]
        action_batch = [minibatch[i][1] for i in range(self.batch_size)]
        reward_batch = [minibatch[i][2] for i in range(self.batch_size)]
        nextState_batch = [minibatch[i][3] for i in range(self.batch_size)]
        terminal_batch = [minibatch[i][4] for i in range(self.batch_size)]
        y_batch = []
        QValueT_batch = self.sess.run(self.q_valueT, feed_dict={self.sT:nextState_batch})
        QValue_batch = self.sess.run(self.q_value, feed_dict={self.s: nextState_batch})
        actions = []
        for i in range(0, self.batch_size):
            if terminal_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                action = np.argmax(QValue_batch[i])
                y_batch.append(reward_batch[i] + self.gamma * QValueT_batch[i][action])
                #y_batch.append(reward_batch[i] + self.gamma * np.max(QValueT_batch[i]))
            actions.append([1 if action_batch[i] == j else 0 for j in range(self.env.action_size)])
        actions = np.array(actions).reshape((-1,self.env.action_size))
        self.sess.run(self.trainStep, feed_dict={
            self.yInput:y_batch,
            self.actionInput: actions,
            self.s : state_batch
        })
        if(self.step % self.sl_frequency == 0):
            print("Saving Loss...")
            self.sl_update_count += 1
            loss = self.sess.run(self.loss,feed_dict={
                self.yInput:y_batch,
                self.actionInput: actions,
                self.s : state_batch
            })
            self.sumOfloss += loss
            avgloss = self.sess.run(self.summaryAvgLoss, feed_dict={
                self.VarOfAvgLoss:self.sumOfloss / self.sl_update_count
            })
            self.train_writer.add_summary(avgloss, self.step)

        if self.step % self.sn_frequency == 0:
            self.saver.save(self.sess, 'saved_networks/' + 'network' + '-dqn', global_step=self.step)

        if self.step % self.tn_update_freq == 0:
            self.copyTargetQNetwork()
    def play(self):
        self.env.reset()
        while(True):
            action = np.random.randint(0,self.env.action_size)
            observation, reward, terminal, _ = self.env.step(action)
            self.env.render()
            if(terminal):
                self.env.reset()

    def prediction(self):
        if(self.step%self.frame_per_action == 0):
            if random.random() > self.epsilon:
                q_value = self.sess.run(self.q_value, feed_dict={self.s : [self.currentState]})
                self.action = np.argmax(q_value)
            else:
                print("-------------RANDOM ACTION-------------")
                self.action = np.random.randint(0, self.env.action_size)

        if self.epsilon > self.final_epsilon:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / (self.explore + self.learn_start)


        return self.action


    def preprocess(self, observation):
        '''observation = cv2.cvtColor(cv2.resize(observation, (self.env.screen_width, 110)),
                                   cv2.COLOR_BGR2GRAY)
        ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
        observation = observation[110-self.env.screen_height:,]'''
        observation = cv2.cvtColor(cv2.resize(observation, (self.env.screen_width, self.env.screen_height)),
                                   cv2.COLOR_BGR2GRAY)
        ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
        return observation

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.02)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.0, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, 1, stride, stride], padding="VALID",data_format='NCHW')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def variable_summaries(self, var, name):
        infolist = []
        with tf.name_scope('Summaries_'+ name):
            mean = tf.reduce_mean(var)
            infolist.append(tf.summary.scalar('mean' + name, mean))
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                infolist.append(tf.summary.scalar('stddev' + name, stddev))
            infolist.append(tf.summary.scalar('max' + name, tf.reduce_max(var)))
            infolist.append(tf.summary.scalar('min' + name, tf.reduce_min(var)))
            infolist.append(tf.summary.histogram('histogram' + name, var))
        return infolist
    def setInitState(self, observation):
        observation = self.preprocess(observation)
        self.currentState = np.stack((observation, observation, observation, observation))
    def clipped_error(self, x):
        # Huber loss
        try:
            return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
        except:
            return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)