from dqn.agent import *
from dqn.environment import *

def calc_gpu_fraction(fraction_string):
  idx, num = fraction_string.split('/')
  idx, num = float(idx), float(num)

  fraction = 1 / (num - idx + 1)
  print(" [*] GPU : %.4f" % fraction)
  return fraction
gpu_options = tf.GPUOptions(
      per_process_gpu_memory_fraction=calc_gpu_fraction('1/1'))

session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def playGame():
    env = Environment()
    observation = env.reset()
    agent = Agent(env, session)
    agent.setInitState(observation)


    while(True):
        action = agent.prediction()
        observation, reward, terminal, _ = env.step(action)
        agent.learn(observation, action, reward, terminal)

if __name__ == '__main__':
    playGame()
