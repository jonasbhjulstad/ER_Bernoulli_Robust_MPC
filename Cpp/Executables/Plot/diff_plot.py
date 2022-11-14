# Plot ../data/traj_

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import sys
# BINDER_DIR = "/home/arch/Documents/Bernoulli_Network_Optimal_Control/Cpp/build/Binders/"
# sys.path.insert(0, BINDER_DIR)
# from pyFROLS import *

if __name__ == '__main__':
   # read and plot all csv in ../data
    # data_path = "C:\\Users\\jonas\\Documents\\Network_Robust_MPC\\Cpp\\data\\"
    cwd = os.path.dirname(os.path.realpath(__file__))
    diff_path = "/home/man/Documents/Bernoulli_MC/Cpp/build/Executables/Regression/"

    

    N_pop = 50

    # find all csv in data_path
    files = glob.glob(diff_path + "/y_diff*.txt")
    # files = glob.glob(DATA_DIR + "SIR_Sine_Trajectory_Discrete_*.csv")
    #sort q_files according to float in name
    diffs = [np.genfromtxt(f) for f in files]
    

    [plt.plot(d) for d in diffs]

    # plot S, I, R, p_I, p_R
    plt.show()
