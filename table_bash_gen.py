import glob
import os
from os.path import join
import numpy as np

DIR_DATA = join(".", "datasets")
DIR_SAVE = os.path.join(os.environ["HOME"], "NPBC/results")
DATASETS = glob.glob(DIR_DATA + "/*.txt")
DATASETS = [f_name for f_name in DATASETS if "_test.txt" not in f_name]
DATASETS.sort()

eps = np.hstack([np.arange(1, 10) * 1e-3,
                 np.arange(1, 10) * 1e-2,
                 np.arange(1, 10) * 1e-1])
methods = ("exp", "wasserstein", "moment")
cv = 5
repeat = 10


def file_writer(b_file, command):
    """ command writer in a file """
    for dataset in DATASETS:
        if "moment" in command:
            f_name = dataset[11:-4] + "_" + command.split("\"")[1]
            f_name = os.path.join(DIR_SAVE, f_name)
            if not os.path.exists(f_name):
                print(command + "\"{}\"".format(dataset), file=b_file)
        else:
            for eps_1 in eps:
                for eps_2 in eps:
                    f_name = dataset[11:-4] + "_" + command.split("\"")[1] + "_" \
                             + str(eps_1) + "_" + str(eps_2)
                    f_name = os.path.join(DIR_SAVE, f_name)
                    if not os.path.exists(f_name):
                        print(command + "\"{}\" --eps {:0.3f} {:0.3f}".format(
                            dataset, eps_1, eps_2), file=b_file)


with open("./tester.sh", "w") as bash_file:
    for method in methods:
        command = "python tester.py --method \"{}\" --cv {} --repeat {} --dataset ".format(
            method, cv, repeat)
        file_writer(bash_file, command)
