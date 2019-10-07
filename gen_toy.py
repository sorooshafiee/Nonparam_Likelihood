import numpy as np


eps = np.hstack([np.arange(1, 10) * 1e-5,
                 np.arange(1, 10) * 1e-4,
                 np.arange(1, 10) * 1e-3,
                 np.arange(1, 10) * 1e-2,
                 np.arange(1, 10) * 1e-1,
                 np.arange(1, 11) * 1e0])
methods = ("exp", "kl", "wasserstein", "moment")


def file_writer(b_file, command):
    """ command writer in a file """
    if "moment" in command:
        print(command, file=b_file)
    else:
        for eps_1 in eps:
            print(command + " --eps {:0.5f}".format(eps_1), file=b_file)


with open("./toy.sh", "w") as bash_file:
    for method in methods:
        command = "python toy_example.py --method \"{}\"".format(method)
        file_writer(bash_file, command)