import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt

#List of processes where we adjust lower level learning rate on the toy dataset

learning_rates = [0.0005, 0.001, 0.002]

cmd = "python run.py --train --training_stage=0 --toy_data"

"""
for hlr in learning_rates:
    for llr in learning_rates:
        cmd_with_lrs = cmd + " --high_level_lr " + str(hlr) + " --low_level_lr " + str(llr)
        os.system(cmd_with_lrs)
        """
#Read from a text file with results (overall accuracy for now)
#TODO: modify the code where they log values in the console to also put these values in a text file?
#Also check what's in model checkpoings/if they themselves log model performance

#Make a plot
ckpt_dir = 'Log/hRL/REINFORCE/checkpoint'
dirs = os.listdir(ckpt_dir)
# for d in dirs:
#     matches = re.search('{(.*)_0', d)
#     llr = float(matches.group(1))
#     matches = re.search('0_(.*)}', d)
#     hlr = float(matches.group(1))
#     stats_file = os.path.join(ckpt_dir, d, 'stats.pkl')
#     data = pickle.load(open(stats_file))
#     print(data)

for d in dirs:
    result_dev_performance = ckpt_dir + "/" + d + "/dev_iter2performance.pkl"
    data = pickle.load(open(result_dev_performance))
    ###Read data from pickle file
    #accuracy = []
    iters = np.arange(len(accuracy)) * 250
    matches = re.search('{(.*)_0', d)
    llr = float(matches.group(1))
    matches = re.search('0_(.*)}', d)
    hlr = float(matches.group(1))
    plt.plot(iters, accuracy, label = "llr = " + str(llr) + ", hlr = " + str(hlr))
plt.legend()
plt.show()