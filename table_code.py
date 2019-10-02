import os
import glob
import numpy as np
import pandas as pd

DIR_SAVE = os.path.join(os.environ["HOME"], "NPBC/results")
cwd = os.getcwd()
DIR_CSV = os.path.join(cwd, "csv")
DIR_TABLE = os.path.join(cwd, "table")

FINAL_CSV = os.path.join(DIR_CSV, "main.csv")
FILESET = glob.glob(DIR_SAVE + "/*.csv")
FILESET.sort()

columns = ['dataset', 'method', 'rho_1', 'rho_2'] + np.arange(20).tolist()
data = []


def to_latex(df, latex_file=None):
    result = 0
    for i in range(10):
        tmp = df.loc[df.groupby(['dataset', 'method'])[i].idxmax()][2*i+1].reset_index()
        tmp = pd.pivot_table(tmp[['dataset', 'method', 2*i+1]],
                             values=2*i+1,
                             index=['dataset'],
                             columns='method')
        result += tmp
    result /= 10
    if latex_file is not None:
        with open(latex_file, 'w') as f:
            f.write("\documentclass{article} \n \\usepackage{booktabs} \n \\begin{document} \n")
            f.write(result.round(2).to_latex())
            f.write("\n \\end{document}")
        os.chdir(DIR_TABLE)
        os.system('pdflatex {}'.format(latex_file))
        os.chdir(cwd)
    return result


if os.path.isfile(FINAL_CSV):
    df_main = pd.read_csv(FINAL_CSV)
    df_main.columns = columns
else:
    for ind, fname in enumerate(FILESET):
        df = pd.read_csv(fname, header=None)
        result = df.loc[:, [1, 3]].get_values()
        name = fname.split('/')[-1][:-4]
        sname = name.split('_')
        dataset = '_'.join(sname[:-3])
        method = sname[-3]
        rho_1 = sname[-2]
        rho_2 = sname[-1]
        row = np.append(np.array([dataset, method, rho_1, rho_2]), result.ravel())
        data.append(row)
    df_main = pd.DataFrame(data=data, columns=columns)
    df_main.to_csv(FINAL_CSV, index=False)

convert_dict = dict(list(zip(columns[2:], [float]*len(columns[2:]))))
df_main = df_main.astype(convert_dict)

df_ = df_main.set_index(['dataset', 'method', 'rho_1', 'rho_2'])
result_auc = to_latex(df_, os.path.join(DIR_TABLE, "auc.tex"))

cleanupFiles = glob.glob(os.path.join(DIR_TABLE, "*.*"))
for cleanupFile in cleanupFiles:
    if '.tex' in cleanupFile or '.pdf' in cleanupFile:
        pass
    else:
        os.remove(cleanupFile)