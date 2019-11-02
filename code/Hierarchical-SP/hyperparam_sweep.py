import os

#List of processes where we adjust lower level learning rate on the toy dataset

learning_rates = [0.0005, 0.001, 0.002]

cmd = "python run.py --train --training_stage=0 --toy_data"

for hlr in learning_rates:
    for llr in learning_rates:
        cmd_with_lrs = cmd + " --high_level_lr " + str(hlr) + " --low_level_lr " + str(llr)
        os.system(cmd_with_lrs)
#Read from a text file with results (overall accuracy for now)
#TODO: modify the code where they log values in the console to also put these values in a text file?
#Also check what's in model checkpoings/if they themselves log model performance

#Make a plot
