import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc


rc("font", family="serif")
rc("text", usetex=True)
rc('xtick',labelsize=18)
rc('ytick',labelsize=18)
rc('legend',fontsize=18)
plt.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}",
                                       r"\usepackage{amssymb}"]


cwd = os.getcwd()
DIR_CSV = os.path.join(cwd, "csv")
FILES = glob.glob(DIR_CSV + "/*.csv")
FILES.sort()
df_kl = []
eps_kl = []
df_wass = []
eps_wass = []
df_exp = []
eps_exp = []
df_moment = []


def plot_eps(eps, df, xlabel, ylabel, fname, fig=None, ax=None, grid=False, shade=False):
    df.index = eps
    df.sort_index(inplace=True)
    eps.sort()
    df = np.array(df.values.tolist())
    ind = int(df.shape[2] / 6)
    df = df[:, :, :ind]

    x = np.array(eps[18:])
    y_mean = np.mean(df, axis=1)
    y_max = np.max(df, axis=1)
    y_min = np.min(df, axis=1)
    if fig is None or ax is None:
        fig, ax = plt.subplots(1)
    for i in range(y_mean.shape[1]-9):
        if i < 10:
            N = i + 1
        else:
            N = (i-8) * 10
        if N in [1,2,4,8,10]:
            ax.semilogx(x, y_mean[18:, i], lw=2, label="$N_i={}$".format(N))
            if shade:
                ax.fill_between(x, y_max[18:, i], y_min[:, i], alpha=0.2)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=26)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=20)
    if grid:
        ax.grid(True, which="both")
    ax.legend(loc="best")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, 3)
    # ax.set_ylim(0.3, 2.1)
    if fname is not None:
        fig.savefig(fname, format="pdf", dpi=1000, bbox_inches = 'tight', pad_inches = 0.02)
    return fig, ax


def plot_N(N, df, xlabel, ylabel, fname, fig=None, ax=None, grid=False):
    df = np.array(df.values.tolist())
    ind = int(df.shape[2] / 6)
    df = df[:, :, :ind]
    x = np.array(N)
    y = np.mean(df, axis=-2)
    if y.ndim == 2:
        y = np.min(y, axis=0)
    if fig is None or ax is None:
        fig, ax = plt.subplots(1)
    ax.plot(x, y, lw=2)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=20)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=20)
    if grid:
        ax.grid(True, which="both")
    ax.set_xlim(1, 100)
    ax.set_ylim(0, 1.5)
    ax.set_xscale("log")
    if fname is not None:
        fig.savefig(fname, format="pdf", dpi=1000, bbox_inches = 'tight', pad_inches = 0.02)
    return fig, ax


for file in FILES:
    result = np.genfromtxt(file, delimiter=",")
    name = file.split("/")[-1][:-4]
    sname = name.split("_")
    row = []
    if sname[0] == "moment":
        df_moment.append(result.tolist())
    elif sname[0] == "wasserstein":
        eps_wass.append(float(sname[1]))
        df_wass.append(result.tolist())
    elif sname[0] == "kl":
        eps_kl.append(float(sname[1]))
        df_kl.append(result.tolist())
    elif sname[0] == "exp":
        eps_exp.append(float(sname[1]))
        df_exp.append(result.tolist())
    else:
        pass

all_N = np.hstack([np.arange(1, 10) * 1e0,
                   np.arange(1, 11) * 1e1])
y_label = r"$\mathrm{KL}(\widehat q \parallel q_{\text{discretize}}( \cdot | x))$"

df_exp = pd.DataFrame(data=df_exp)
# plot_eps(eps_exp, df_exp, r"$\varepsilon$", y_label, "eps_exp.pdf")
# plot_N(all_N, df_exp, r"$N_i$", y_label, "N_exp.pdf")

df_kl = pd.DataFrame(data=df_kl)
plot_eps(eps_kl, df_kl, r"$\varepsilon$", y_label, "eps_kl.pdf")
# plot_N(all_N, df_kl, r"$N_i$", y_label, "N_kl.pdf")

df_wass = pd.DataFrame(data=df_wass)
plot_eps(eps_wass, df_wass, r"$\varepsilon$", y_label, "eps_wass.pdf")
# plot_N(all_N, df_wass, r"$N_i$", y_label, "N_wass.pdf")

df_moment = pd.DataFrame(data=df_moment)
# plot_N(all_N[1:], df_moment, r"$N_i$", y_label, "N_mean.pdf")

fig, ax = plt.subplots(1)
fig, ax = plot_N(all_N, df_kl, None, None, None, fig, ax)
fig, ax = plot_N(all_N, df_wass, None, None, None, fig, ax)
fig, ax = plot_N(all_N, df_exp, None, None, None, fig, ax)
fig, ax = plot_N(all_N[1:], df_moment, r"$N_i$", y_label, None, fig, ax)
ax.legend([r"KL divergence",
           r"Wasserstein",
           r"Exponential kernel",
           r"Moment"])
fig.savefig("all.pdf", format="pdf", dpi=1000, bbox_inches = 'tight', pad_inches = 0.02)






