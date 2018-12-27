import gym
import config

class Environment:
    def __init__(self):
        self.env = gym.make(config.env_name)
        self.game_name = config.env_name
        self.dims = (config.screen_width, config.screen_height)
        self.screen_height = config.screen_height
        self.screen_width = config.screen_width
        self.observation = None
        self.screen = None
        self.reward = 0
        self.terminal = True
        self.display = config.display
        self.info = None

    def step(self, action):
        cumulated_reward = 0
        start_lives = self.lives
        for _ in range(config.action_repeat):
            self.observation, self.reward, self.terminal, self.info = self.env.step(action)

            self.reward = max(-1, min(self.reward, 1))
            cumulated_reward += self.reward

            if config.proceed_train and start_lives > self.lives:
                cumulated_reward -= 1
                self.terminal = True
                break

        self.reward = cumulated_reward
        return self.observation, self.reward, self.terminal, self.info

    def render(self):
        if(self.display):
            self.env.render()

    def reset(self):
        self.observation = self.env.reset()
        return self.observation

    @property
    def action_size(self):
        return self.env.action_space.n
    @property
    def state(self):
        return self.observation, self.reward, self.terminal

    @property
    def lives(self):
        if(self.info is None):
            return 0
        else:
            return self.info['ale.lives']