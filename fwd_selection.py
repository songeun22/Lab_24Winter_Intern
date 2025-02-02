import numpy as np
import pandas as pd
import adafdr.method as md
import matplotlib.pyplot as plt
import seaborn as sns

def adafdr_rej_num(p, x, alpha):

  res =  md.adafdr_test(p, x, alpha = alpha, output_folder = None)
  ths = res[1]

  return np.sum(p <= ths)



def adafdr_fwd(p, cov, alpha, max_dim = float('inf')):
  # p: p-value (n, ) numpy, cov: covariate (n, d) dataframe, alpha: nominal fdr
  d = cov.shape[1]
  iter = 0
  max_idx = np.nan

  # !!! raise error

  if max_dim > d:
    max_dim = d

  res_df = pd.DataFrame(columns = cov.columns) # all attained results
  max_idx_list = []  # selected indexes & rejection counts
  max_rej_list = []

  while iter <= max_dim - 1 :

    rej_list = []

    for i in range(d):
      if i == max_idx:
        rej_list.append(np.nan)
      else:
        x = cov.iloc[:, max_idx_list + [i]].values
        rej_num = adafdr_rej_num(p, x, alpha)
        rej_list.append(rej_num)

    res_df.loc[iter] = rej_list

    max_rej = max(rej_list)
    max_idx = rej_list.index(max_rej)  #!!!! 같은 max 존재할 때는 그냥 첫번째 cov 선택하는 알고리즘??

    max_idx_list.append(max_idx)
    max_rej_list.append(max_rej)

    if (iter != 0) & (max_idx_list[iter] <= max_idx_list[iter - 1]):
      max_idx_list.pop()
      max_rej_list.pop()
      break

    iter += 1

  res = {'selected_covariates': list(res_df.columns[max_idx_list]), 'selected_dim': len(max_idx_list),
         'total_rej': max_rej_list[-1], 'rej_df': res_df}

  return res


def summary_fwd(res):
  # res: result from adafdr_fwd()

  keys = list(res.keys())

  print('='*30)
  print(f'Selected Covariates: {res[keys[1]]}, {res[keys[0]]}')
  print(f'Total Rejections: {res[keys[2]]}')
  print('='*30 + '\n')

  plt.figure(figsize=(20, res[keys[1]]))
  sns.heatmap(res[keys[3]], cmap = 'Greys', square = False)
  plt.show()