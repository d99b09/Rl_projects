

import time


'''
'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model'
modelsname = 'name'
max_reward = ''
average_reward = ''
min_reward = ''
time = str(int(time.time()))
'''

MODEL_NAME = 'RL'
max_reward = 0.123548655454486341545
average_reward = 0.7865445686565462454
min_reward = 0.9099999
output = 'models/%(MODEL_NAME)s__%(max_reward).2fmax_%(average_reward).2favg_%(min_reward).2fmin__%(time)d.model' % \
         {"MODEL_NAME": MODEL_NAME, "max_reward": max_reward, "average_reward":
             average_reward, "min_reward": min_reward, "time": int(time.time())}

print output