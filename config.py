env_name = 'Breakout-v0'
display = False
proceed_train = True
screen_width = 84
screen_height = 84
learning_rate = 0.00025
learning_rate_minimum = 0.00025
learning_rate_decay = 0.96
learning_rate_decay_step = 100000
discount = 0.99
memory_size = 200000
batch_size = 32
initial_epsilon = 1
train_frequency = 4
final_epsilon = 0.1
action_repeat = 4
history_length = 4
tn_update_freq = 10000
gradient_momentum = 0.95
squared_gradient_momentum = 0.95
min_squared_gradient = 0.01
final_exploration_frame = 1000000
replay_start_size = 50000
noop_max = 30
learn_start = 50000
explore = 1000000
frame_per_action = 1
save_network_frequency = 100000
sv_frequency = 5000
sl_frequency = 1000
update_reward_freq = 500