import argparse
import os
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd
from typing import List, Tuple, Dict
import numpy as np
from math import log10
from matplotlib.ticker import Locator
from scipy.stats import gmean

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

BENCHMARK_DIR = 'benchmarks'

DEFAULT_BENCHMARKS = [
  'noise',
  # 'radar',
  'envnoise',
  'outlier',
  'outlierheavy',
  'gtree',
  'slds',
  'runner',
  'wheels',
  'slam',
]

DEFAULT_METHODS = [
  'ssi',
  'ds',
  'bp',
]

N_INTERVALS = 30

BENCHMARK_LATEX_NAMES = {
  # 'radar': r'\bRadar{}',
  'envnoise': r"\bEnvnoise{}",
  'noise': r"\bNoise{}",
  'outlier': r"\bOutlier{}",
  'gtree': r"\bGtree{}",
  'outlierheavy': r"\bOutlierheavy{}",
  'slds': r"\bSlds{}",
  'runner': r"\bRunner{}",
  'example': r"\bExample{}",
  'wheels': r"\bWheels{}",
  'slam': r"\bSlam{}",
}

N_INTERVALS = 30

COLORS = [
  "#f4827b", # red
  "#feb140", # orange
  "#9baa20", # green
  "#7cc7d2", # blue
  "#72b0e8", # indigo
  "#c6abd1", # purple
  "#f288de", # magenta
  "#c78f77", # brown
  "#fac52f", # yellow
  "#969595", # gray
]
EDGECOLORS = [
  "#b7575c", # red
  "#c78200", # orange
  "#708200", # green
  "#4d9aa4", # blue
  "#204f87", # indigo
  "#7c5b83", # purple
  "#9c0085", # magenta
  "#56281b", # brown
  "#c0ab5f", # yellow
  "#171616", # gray
]

MARKERS = ['s', 'v', 'd', 'o', 'X', 'p', 'h', 'P', '*', '<']

MARKERSSIZE = 55

BARWIDTH = 0.25
BARSPACING = 0.03
MAGICSPACING = 0.8

YLABELSIZE = 9
XLABELSIZE = 12
BARLABELSIZE = 8

N = 100

GRIDPARAMS = {'which': 'major', 'color': 'gray', 'linestyle': '--', 'alpha': 0.5}

PLT_SETTINGS = {
  'font.size': 14, 
  # 'font.family': 'Linux Libertine', 
  # "text.usetex": True,                # use LaTeX to write all text
  # "text.latex.preamble": "\n".join([         # plots will use this preamble
  #   '\\usepackage{libertine}'
  # ])
}

# https://stackoverflow.com/questions/20470892/how-to-place-minor-ticks-on-symlog-scale
class MinorSymLogLocator(Locator):
  """
  Dynamically find minor tick positions based on the positions of
  major ticks for a symlog scaling.
  """
  def __init__(self, linthresh, nints=10):
    """
    Ticks will be placed between the major ticks.
    The placement is linear for x between -linthresh and linthresh,
    otherwise its logarithmically. nints gives the number of
    intervals that will be bounded by the minor ticks.
    """
    self.linthresh = linthresh
    self.nintervals = nints

  def __call__(self):
    # Return the locations of the ticks
    majorlocs = self.axis.get_majorticklocs()

    if len(majorlocs) == 1:
      return self.raise_if_exceeds(np.array([]))

    # add temporary major tick locs at either end of the current range
    # to fill in minor tick gaps
    dmlower = majorlocs[1] - majorlocs[0]    # major tick difference at lower end
    dmupper = majorlocs[-1] - majorlocs[-2]  # major tick difference at upper end

    # add temporary major tick location at the lower end
    if majorlocs[0] != 0. and ((majorlocs[0] != self.linthresh and dmlower > self.linthresh) or (dmlower == self.linthresh and majorlocs[0] < 0)):
      majorlocs = np.insert(majorlocs, 0, majorlocs[0]*10.)
    else:
      majorlocs = np.insert(majorlocs, 0, majorlocs[0]-self.linthresh)

    # add temporary major tick location at the upper end
    if majorlocs[-1] != 0. and ((np.abs(majorlocs[-1]) != self.linthresh and dmupper > self.linthresh) or (dmupper == self.linthresh and majorlocs[-1] > 0)):
      majorlocs = np.append(majorlocs, majorlocs[-1]*10.)
    else:
      majorlocs = np.append(majorlocs, majorlocs[-1]+self.linthresh)

    # iterate through minor locs
    minorlocs = []

    # handle the lowest part
    for i in range(1, len(majorlocs)):
      majorstep = majorlocs[i] - majorlocs[i-1]
      if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
          ndivs = self.nintervals
      else:
          ndivs = self.nintervals - 1.

      minorstep = majorstep / ndivs
      locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
      minorlocs.extend(locs)

    return self.raise_if_exceeds(np.array(minorlocs))

  def tick_values(self, vmin, vmax):
    raise NotImplementedError('Cannot get tick locations for a '
                      '%s type.' % type(self))

def table(statistics):
  content = []

  for method in DEFAULT_METHODS:

    method_str = '\\' + method

    method_content = []
    for benchmark in DEFAULT_BENCHMARKS:
      
      n_true_satisfied = statistics[benchmark][method]['n_true_satisfied']
      n_inferred_satisfied = statistics[benchmark][method]['n_inferred_satisfied']

      method_content.append(f"{n_inferred_satisfied}/{n_true_satisfied}")

    content.append(f"{method_str} & {' & '.join(method_content)} \\\\")

  total_content = []
  for benchmark in DEFAULT_BENCHMARKS:
    n_plans = statistics[benchmark]['n_plans']
    total_content.append(f"{n_plans}")

  content.append(f"\\midrule")
  content.append(f"Total Possible Plans & {' & '.join(total_content)} \\\\")

  print(f"Algorithm & {' & '.join([BENCHMARK_LATEX_NAMES[benchmark] for benchmark in DEFAULT_BENCHMARKS])} \\\\")
  print('\\midrule')

  for line in content:
    print(line)

def close_to_target_error(target, value):
  if value == 0:
    return True
  if log10(value) - log10(target) <= 0.5:
    return True
  return False

def close_to_target_runtime(target_runtime, runtime):
  if abs(log10(runtime) - log10(target_runtime)) <= 0.15:
    return True

  return False
  
def plot_particles(data, output, methods, plan_ids, all_plans, 
                   particles, n_y, n_x, base_x, base_y, legend_width, is_example, error_bar):
  plt.rcParams.update(PLT_SETTINGS)

  # first 4 columns are metadata
  variables = data.columns[4:]

  # drop rows with particles not in particles
  if particles is not None:
    data = data.loc[data['particles'].isin(particles)]

  particles = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
  original_plan_ids = plan_ids

  if is_example:
    methods = ['ssi']

  for method_i, method in enumerate(methods):

    figsize = (base_x * n_x, base_y * n_y) if benchmark != 'slds' else (base_x * n_x, base_y * n_y + 1)

    fig, axes = plt.subplots(
                  n_y, n_x,
                  figsize=figsize,  
                  sharex=True, 
                  # sharey=True
    )

    for ax in axes.flatten():
      ax.set_visible(False)

    if original_plan_ids is None:
      plans = sorted(data[data['method'] == method]['plan_id'].unique())
    else:
      plans = original_plan_ids
      
    if len(plans) <= 1:
      continue

    use_label = True
    for var_i, var in enumerate(variables):
      plot_i = var_i % n_x
      plot_j = var_i // n_x

      if n_y == 1:
        ax = axes[plot_i]
      else:
        ax = axes[plot_j, plot_i]

      ax.set_visible(True)

      use_log = True

      plan_i = 0
      for plan_id in plans:
        plan_id = int(plan_id)

        if use_label:
          label = f'Plan {plan_id}'
          if is_example:
            labels = {
              1: 'No Annotations',
              2: 'Symbolic r Plan',
              3: 'Dynamic r Plan',
              4: 'Symbolic x Plan',
              5: 'Sample All Plan',
            }
            label = labels[plan_id]
        else:
          label = None

        # 90th percentile error over n runs
        upper = data.loc[data['method'] == method]\
                     .loc[data['plan_id'] == plan_id]\
                     .groupby(['particles'])[var]\
                     .quantile(0.9)
        
        # median error over n runs
        median = data.loc[data['method'] == method]\
                     .loc[data['plan_id'] == plan_id]\
                     .groupby(['particles'])[var]\
                      .median()
        
        # 10th percentile error over n runs
        lower = data.loc[data['method'] == method]\
                     .loc[data['plan_id'] == plan_id]\
                     .groupby(['particles'])[var]\
                     .quantile(0.1)
        
        # min non-zero value
        
        nonzero = [x for x in lower if x > 0]
        thresh = min(nonzero) if len(nonzero) > 0 else 1e-10
        
        # median time over n runs
        runtimes = data.loc[data['method'] == method]\
                        .loc[data['plan_id'] == plan_id]\
                        .groupby(['particles'])['time']\
                        .median()
        
        yerr = np.array([median - lower, upper - median])

        if runtimes.shape[0] == 0:
          continue

        if is_example:
          mfc = {
            1: '#f4827b',
            2: '#7cc7d2',
            3: '#feb140',
            4: '#9baa20',
            5: '#fac52f',
          }[plan_id]
          mec = {
            1: '#b7575c',
            2: '#4d9aa4',
            3: '#c78200',
            4: '#708200',
            5: '#c0ab5f',
          }[plan_id]
          marker = {
            1: 's',
            2: 'v',
            3: 'o',
            4: 'd',
            5: 'X',
          }[plan_id]
        else:
          mfc = COLORS[plan_i]
          mec = EDGECOLORS[plan_i]
          marker = MARKERS[plan_i]

        if error_bar:
          ax.errorbar(runtimes, median, yerr=yerr, marker=MARKERS[plan_i], mfc=mfc, mec=mec, ecolor=mec, label=label,
                    capsize=5, markersize=5, xerr=None, ls='none')
          
        else:
          ax.scatter(runtimes, upper, marker=marker, color=mfc, label=label, 
                                    edgecolor=mec, s=MARKERSSIZE)
        
        # if y range doesn't cross any power of 10, use log scale
        ax.set_yscale('symlog', linthresh=thresh)
        max_y = max(upper)
        min_y = min(upper)
        powers = [10**i for i in range(-10, 10)]
        for power in powers:
          if min_y < power and max_y > power:
            use_log = False
            break
        
        plan_i += 1

      use_label = False
      ax.set_title(f'{var}')
      ax.grid(**GRIDPARAMS)
      ax.set_axisbelow(True)

      ax.set_xscale('log')
      ax.tick_params(
        axis='x',           # changes apply to the x-axis
        which='major',       # both major and minor ticks are affected
        bottom=True,
        top=False,
        labelbottom=True)

      ax.yaxis.set_minor_locator(MinorSymLogLocator(thresh))
      # check if there are any major ticks
      if use_log and not error_bar:
        # label minor ticks
        ax.yaxis.set_minor_formatter(ScalarFormatter())

    print('Saving particles plots')

    if benchmark == 'slds':
      y = 1
      bbox = (0.5, 0)
    else:
      y = 1.08
      bbox = (0.5, -0.1)

    # fig.suptitle(f'Variable Accuracy to Execution Time', y=y)
    lgd = fig.legend(loc='upper center', ncols=legend_width, bbox_to_anchor=bbox)

    if n_y == 1:
      for i in range(n_x):
        plt.setp(axes[i], xlabel='Execution Time in s (log scale)')
      plt.setp(axes[0], ylabel='Error (log scale)')
    else:
      for i in range(n_x):
        plt.setp(axes[-1, i], xlabel='Execution Time in s (log scale)')
      plt.setp(axes[:, 0], ylabel='Error (log scale)')
      
    if benchmark == 'slds':
      plt.subplots_adjust(hspace=0.3)

    if is_example:
      plan_id_str = ''.join(plan_ids)
      filename = f'{method}_example_{plan_id_str}'
    else:
      filename = f'{method}_particles'
    if error_bar:
      filename += '_errorbar'
    # fig.savefig(os.path.join(output, f'{filename}.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(output, f'{filename}.png'), bbox_inches='tight')

    plt.close(fig)

def compare_to_default(data, methods, plan_ids, all_plans, default_plans):
  def get_error_runtime(data, method, plan_id):
    # default 90th percentile error over n runs
    upper = data.loc[data['method'] == method]\
                  .loc[data['plan_id'] == plan_id]\
                  .drop(columns=['plan_id', 'method', 'time'])\
                  .groupby(['particles'])\
                  .quantile(0.9)

    runtimes = data.loc[data['method'] == method]\
                  .loc[data['plan_id'] == plan_id]\
                  .groupby(['particles'])['time']\
                  .median()

    # combine
    errors = pd.concat([runtimes, upper], axis=1)
    errors['particles'] = errors.index
    errors = errors.reset_index(drop=True)
    
    return errors
  
  original_plan_ids = plan_ids
  
  variables = data.columns[4:]

  for method in methods:
    if method not in default_plans:
      continue
    print('==============')
    print(f'Method: {method}')
    default_plan = int(default_plans[method])

    print('Default Plan:', default_plan)

    if original_plan_ids is None:
      plan_ids = [plan_id for plan_id, data in all_plans.items() if data['satisfiable'][method]]

    plan_ids = map(int, plan_ids)

    default_errors = get_error_runtime(data, method, default_plan)

    ratios = pd.DataFrame(columns=list(variables)+['plan', 'runtime', 'particles'])
    outlier_ratios = pd.DataFrame(columns=list(variables)+['plan', 'runtime'])
    for j, plan_id in enumerate(plan_ids):
      if plan_id == default_plan:
        continue
      # 90th percentile error over n runs
      errors = get_error_runtime(data, method, plan_id)

      # compute ratio of error to default error based on runtime
      for row in errors.iterrows():
        runtime = row[1]['time']
        error = row[1].drop(['time', 'particles'])
        default_error = default_errors.loc[default_errors['time'].apply(lambda x: close_to_target_runtime(runtime, x))]

        if default_error.shape[0] == 0:
          continue

        default_error = default_error.iloc[-1].drop(['time', 'particles'])

        ratio = default_error / error

        if benchmark == 'outlier':
          # get max ratio of error to default error when runtime > 10
          if runtime > 10:
            outlier_ratios.loc[len(outlier_ratios.index)] = list(ratio) + [plan_id, runtime]

        if runtime > 0.1:        
          ratios.loc[len(ratios.index)] = list(ratio) + [plan_id, runtime, row[1]['particles']]

    print(ratios)
    max_ratio = ratios.drop(columns=['plan', 'runtime']).max().max()
    print('Max Ratio:', max_ratio)
    # find the plan that had the max ratio
    for i, row in ratios.iterrows():
      if row.drop(['plan','runtime']).max() == max_ratio:
        print('Plan:', row['plan'])
        print('Runtime:', row['runtime'])
        break

    if benchmark == 'outlier' and method == 'ssi':
      print('======')
      print('Max Ratio when runtime > 10:')
      max_outlier_ratio = outlier_ratios['xt'].max()
      print('Max Ratio:', max_outlier_ratio)
      print('Plan:', outlier_ratios.loc[outlier_ratios['xt'].idxmax()]['plan'])
      print('Runtime:', outlier_ratios.loc[outlier_ratios['xt'].idxmax()]['runtime'])
    elif benchmark == 'noise' and method == 'ssi':
      print('======')
      print('Max Ratio for x:')
      max_x_ratio = ratios['x'].max()
      print('Max Ratio:', max_x_ratio)
      print('Plan:', ratios.loc[ratios['x'].idxmax()]['plan'])
      print('Runtime:', ratios.loc[ratios['x'].idxmax()]['runtime'])
      print('======')
      print('Max Ratio for r with Plan 3:')
      max_r_ratio = ratios.loc[ratios['plan'] == 3]['r'].max()
      print('Max Ratio:', max_r_ratio)
      print('Plan:', 3)
      print('Runtime:', ratios.loc[ratios['plan'] == 3].iloc[0]['runtime'])
    elif benchmark == 'runner' and method == 'bp':
      print('======')
      print('Max Ratio for Plan 3:')
      max_r_ratio = ratios.loc[ratios['plan'] == 3].max().max()
      print('Max Ratio:', max_r_ratio)
      print('Plan:', 3)
      for i, row in ratios.loc[ratios['plan'] == 3].iterrows():
        if row.drop(['plan','runtime']).max() == max_r_ratio:
          print('Runtime:', row['runtime'])
          break
    elif benchmark == 'example':
      print('======')
      print('Max Ratio for x:')
      max_x_ratio = ratios['x'].max()
      print('Max Ratio:', max_x_ratio)
      print('Plan:', ratios.loc[ratios['x'].idxmax()]['plan'])
      print('Runtime:', ratios.loc[ratios['x'].idxmax()]['runtime'])

    plan_ids = None
  print()

def compare_to_default_example(data, methods, plan_ids, all_plans, default_plans):
  def get_error_runtime(data, method, plan_id):
    # default 90th percentile error over n runs
    upper = data.loc[data['method'] == method]\
                  .loc[data['plan_id'] == plan_id]\
                  .drop(columns=['plan_id', 'method', 'time'])\
                  .groupby(['particles'])\
                  .quantile(0.9)

    runtimes = data.loc[data['method'] == method]\
                  .loc[data['plan_id'] == plan_id]\
                  .groupby(['particles'])['time']\
                  .median()

    # combine
    errors = pd.concat([runtimes, upper], axis=1)
    errors['particles'] = errors.index
    errors = errors.reset_index(drop=True)
    
    return errors
  
  original_plan_ids = plan_ids
  
  variables = data.columns[4:]

  plan5 = get_error_runtime(data, 'ssi', 5)
  plan2 = get_error_runtime(data, 'ssi', 2)

  # ratio of errors matching by particles
  ratios = pd.DataFrame(columns=list(variables)+['plan', 'particles'])
  for j, particles in enumerate(plan5['particles']):
    # get error for particles
    error5 = plan5.loc[plan5['particles'] == particles].drop(columns=['particles', 'time'])
    error2 = plan2.loc[plan2['particles'] == particles].drop(columns=['particles', 'time'])

    ratio = error5 / error2

    ratios.loc[len(ratios.index)] = list(ratio.iloc[0]) + [5, particles]

  print(ratios)
  

def compare_to_default_accuracy(benchmark, data, methods, plan_ids, all_plans, default_plans):
  def get_error_runtime(data, method, plan_id, median=False):
    # default 90th percentile error over n runs
    if median:
      data_slice = data.loc[data['method'] == method]\
                  .loc[data['plan_id'] == plan_id]\
                  .drop(columns=['plan_id', 'method', 'time'])\
                  .groupby(['particles'])\
                  .median()
    else:
      data_slice = data.loc[data['method'] == method]\
                    .loc[data['plan_id'] == plan_id]\
                    .drop(columns=['plan_id', 'method', 'time'])\
                    .groupby(['particles'])\
                    .quantile(0.9)

    median_runtimes = data.loc[data['method'] == method]\
                  .loc[data['plan_id'] == plan_id]\
                  .groupby(['particles'])['time']\
                  .median()
    
    upper_runtimes = data.loc[data['method'] == method]\
                  .loc[data['plan_id'] == plan_id]\
                  .groupby(['particles'])['time']\
                  .quantile(0.9)
    
    lower_runtimes = data.loc[data['method'] == method]\
                  .loc[data['plan_id'] == plan_id]\
                  .groupby(['particles'])['time']\
                  .quantile(0.1)

    # combine
    errors = pd.concat([median_runtimes, lower_runtimes, upper_runtimes, data_slice], axis=1)
    errors.columns = ['median_time', 'lower_time', 'upper_time'] + list(data_slice.columns)
    errors['particles'] = errors.index
    errors = errors.reset_index(drop=True)
    
    return errors
  
  variables = data.columns[4:]

  original_plan_ids = plan_ids

  row_size = {
    'noise': 5,
    'envnoise': 5,
    'outlier': 3,
    'outlierheavy': 3,
    'gtree': 3,
    'slds': 10,
    'runner': 7,
    'example': 5,
  }

  print("\\midrule")
  print(f"\\multirow{{{row_size[benchmark]}}}{{*}}{{{BENCHMARK_LATEX_NAMES[benchmark]}}}", end='')

  all_times = pd.DataFrame(columns=['method', 'variable', 'plan', 'particles', 'median', 'default_particles', 'default', 'speedup'])

  for method in methods:
    if method not in default_plans:
      continue

    if original_plan_ids is None:
      plan_ids = [plan_id for plan_id, data in all_plans.items() if data['satisfiable'][method]]

    if len(list(plan_ids)) <= 1:
      continue

    default_plan = int(default_plans[method])
    plan_ids = map(int, plan_ids)

    target_errors = get_error_runtime(data, method, default_plan)
    target = target_errors[target_errors['particles'] == 1024]

    default = get_error_runtime(data, method, default_plan)

    min_time = pd.DataFrame(columns=['variable', 'plan', 'particles', 'median', 'default_particles', 'default', 'speedup'])
    for j, plan_id in enumerate(plan_ids):

      # 90th percentile error over n runs
      errors = get_error_runtime(data, method, plan_id)

      for k, var in enumerate(variables):
        # compute particles where error is close to default error for var
        is_close = errors[var].apply(lambda x: close_to_target_error(target[var].values[0], x))

        # get smallest runtime where error is close to default error
        close = errors.loc[is_close]
        if close.shape[0] == 0:
          particles = np.nan
          median_runtime = np.nan
          error = np.nan
        else:
          min_idx = close['median_time'].idxmin()
          particles = errors.loc[min_idx]['particles']
          median_runtime = round(errors.loc[min_idx]['median_time'], 2)
          error = round(errors.loc[min_idx][var], 2)

        # compute particles where error is close to default error for var
        default_is_close = default[var].apply(lambda x: close_to_target_error(target[var].values[0], x))

        # get smallest runtime where error is close to default error
        default_close = default.loc[default_is_close]
        if default_close.shape[0] == 0:
          default_particles = np.nan
          default_median_runtime = np.nan
          default_error = np.nan
        else:
          default_min_idx = default_close['median_time'].idxmin()
          default_particles = default_close.loc[default_min_idx]['particles']
          default_median_runtime = round(default_close.loc[default_min_idx]['median_time'], 2)
          default_error = round(default_close.loc[default_min_idx][var], 2)

        speedup = round(default_median_runtime / median_runtime, 2)

        min_time.loc[len(min_time.index)] = [
          var, plan_id, particles, median_runtime, default_particles, default_median_runtime, speedup
        ]

    # print(min_time)

    # keep the plan with the smallest runtime, make nan infinity
    min_time = min_time.replace(np.nan, np.inf)
    min_time = min_time.loc[min_time.groupby('variable')['median'].idxmin()]
    min_time = min_time.replace(np.inf, np.nan)

    min_time['method'] = method
    if all_times.empty:
      all_times = min_time
    else:
      all_times = pd.concat([all_times, min_time])
    
    # print(min_time)

  all_times = all_times.reset_index(drop=True)

  table_values = {
    var: [] for var in variables
  }
  table_plans = {
    var: [] for var in variables
  }
  for var in variables:
    for method in methods:
      row = all_times.loc[(all_times['variable'] == var) & (all_times['method'] == method)]
      if len(row) == 0:
        table_values[var].append(["--", "--"])
        table_plans[var].append("")
      elif row['median'].isnull().all():
        table_values[var].append(["\\xmark", round(row['default'].values[0], 2)])
        table_plans[var].append("")
      else:
        table_values[var].append([round(row['median'].values[0], 2), round(row['default'].values[0], 2)])
        table_plans[var].append(int(row['plan'].values[0]))

  for var in variables:
    var_str = var.replace("_", "\\_")
    print(f"& \mkw{{{var_str}}} ", end='')
    for i, method in enumerate(methods):
      if isinstance(table_values[var][i][0], str) or table_values[var][i][0] >= table_values[var][i][1]:
        print(f"& {table_values[var][i][0]} & {table_values[var][i][1]} ", end='')
      else:
        print(f"& \\textbf{{{table_values[var][i][0]}}} & {table_values[var][i][1]} ", end='')
      
      if i == len(methods) - 1:
        print("\\\\[-0.4em]")

    for i, method in enumerate(methods):

      if table_plans[var][i] == "":
        print("& & ", end='')
      else:
        default_plan = int(default_plans[method])
        if table_plans[var][i] == default_plan:
          print("& & \defaultplan{} ", end='')
        else:
          print("& & \\plan{" + str(table_plans[var][i]) + "}", end='')
      if i == len(methods) - 1:
        print("\\\\")

  non_default = all_times.apply(lambda x: x['plan'] != int(default_plans[x['method']]), axis=1)
  non_default_times = all_times.loc[non_default].copy()
  return all_times, non_default_times

def compare_to_default_accuracy_example(data, all_plans):
  def get_error_runtime(data, method, plan_id, median=False):
    # default 90th percentile error over n runs
    if median:
      data_slice = data.loc[data['method'] == method]\
                  .loc[data['plan_id'] == plan_id]\
                  .drop(columns=['plan_id', 'method', 'time'])\
                  .groupby(['particles'])\
                  .median()
    else:
      data_slice = data.loc[data['method'] == method]\
                    .loc[data['plan_id'] == plan_id]\
                    .drop(columns=['plan_id', 'method', 'time'])\
                    .groupby(['particles'])\
                    .quantile(0.9)

    median_runtimes = data.loc[data['method'] == method]\
                  .loc[data['plan_id'] == plan_id]\
                  .groupby(['particles'])['time']\
                  .median()
    
    upper_runtimes = data.loc[data['method'] == method]\
                  .loc[data['plan_id'] == plan_id]\
                  .groupby(['particles'])['time']\
                  .quantile(0.9)
    
    lower_runtimes = data.loc[data['method'] == method]\
                  .loc[data['plan_id'] == plan_id]\
                  .groupby(['particles'])['time']\
                  .quantile(0.1)

    # combine
    errors = pd.concat([median_runtimes, lower_runtimes, upper_runtimes, data_slice], axis=1)
    errors.columns = ['median_time', 'lower_time', 'upper_time'] + list(data_slice.columns)
    errors['particles'] = errors.index
    errors = errors.reset_index(drop=True)
    
    return errors
  
  variables = data.columns[4:]
  methods = ['ssi']
  plan_ids = [2, 5]

  original_plan_ids = plan_ids

  all_times = pd.DataFrame(columns=['method', 'variable', 'plan', 'particles', 'median', 'default_particles', 'default', 'speedup'])

  for method in methods:
    if original_plan_ids is None:
      plan_ids = [plan_id for plan_id, data in all_plans.items() if data['satisfiable'][method]]

    if len(list(plan_ids)) <= 1:
      continue

    plan_ids = map(int, plan_ids)

    target_errors = get_error_runtime(data, method, 5)
    target = target_errors[target_errors['particles'] == 1024]

    sample_r_errors = get_error_runtime(data, method, 2)
    sample_all_errors = get_error_runtime(data, method, 5)

    is_close = sample_r_errors['r'].apply(lambda x: close_to_target_error(target['r'].values[0], x))
    close = sample_r_errors.loc[is_close]
    sample_r_runtime = close['median_time'].min()
    print('Sample r')
    print(close)

    is_close = sample_all_errors['r'].apply(lambda x: close_to_target_error(target['r'].values[0], x))
    close = sample_all_errors.loc[is_close]
    sample_all_runtime = close['median_time'].min()
    print('Sample all')
    print(close)

    print("Ratio of fastest runtime Sample r to Sample all")
    print(sample_r_runtime / sample_all_runtime)

if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--task', '-t', type=str, required=False, default='plot', help='plot, compare, table, accuracy')
  p.add_argument('--benchmark', '-b', type=str, required=False, nargs="+", default=DEFAULT_BENCHMARKS)
  p.add_argument('--output', '-o', type=str, required=False, default='output')
  p.add_argument('--plan-ids', '-pi', type=int, required=False, nargs="+")
  p.add_argument('--methods', '-m', type=str, required=False, nargs="+", default=DEFAULT_METHODS)
  p.add_argument('--particles', '-p', type=int, required=False, nargs='+')
  p.add_argument('--example', action='store_true')
  p.add_argument('--error-bar', action='store_true')

  args = p.parse_args()

  methods = [method for method in args.methods if method in DEFAULT_METHODS]
  particles = [int(particle) for particle in args.particles] if args.particles is not None else None

  if args.task == 'table':
    # Load statistics
    all_statistics = {}

    for benchmark in args.benchmark:
      output = os.path.join(benchmark, args.output)
      with open(os.path.join(output, 'statistics.json')) as f:
        statistics = json.load(f)
      all_statistics[benchmark] = statistics
    table(all_statistics)
  else:
    if args.example:
      args.benchmark = ['example']

    print(args.benchmark)

    all_times = pd.DataFrame(columns=['benchmark', 'method', 'variable', 'plan', 'particles', 'median', 'default_particles', 'default', 'speedup'])
    all_non_default_times = pd.DataFrame(columns=['benchmark', 'method', 'variable', 'plan', 'particles', 'median', 'default_particles', 'default', 'speedup'])

    for benchmark in args.benchmark:
      if not args.task == 'accuracy':
        print('=============================')
        print('Benchmark: {}'.format(benchmark))

      with open(os.path.join(benchmark, 'config.json')) as f:
        config = json.load(f)

      output = os.path.join(benchmark, args.output)

      n_vars = len(config['true_vars'])
      # default to number of variables as number of columns
      n_y = int(config['n_y']) if 'n_y' in config else 1
      n_x = int(config['n_x']) if 'n_x' in config else n_vars
      base_x = int(config['base_x']) if 'base_x' in config else 5
      base_y = int(config['base_y']) if 'base_y' in config else 6
      legend_width = int(config['legend_width']) if 'legend_width' in config else n_vars

      if os.path.exists(os.path.join(output, 'results.csv')):
        data = pd.read_csv(os.path.join(output, 'results.csv'), delimiter=',')

        if args.task == 'plot':
          if args.example:
            plan_ids_sets = [['1', '2'], ['1', '2', '5'], ['1', '2', '4'], ['1', '2', '4', '5']]
          else:
            plan_ids_sets = [args.plan_ids]
          for plan_ids in plan_ids_sets:
            plot_particles(data, output, methods, plan_ids, config['plans'], particles, n_y, n_x, base_x, base_y, legend_width, args.example, args.error_bar)

        if args.task == 'compare':
          if args.example:
            compare_to_default_example(data, methods, args.plan_ids, config['plans'], config['default'])
          else:
            compare_to_default(data, methods, args.plan_ids, config['plans'], config['default'])
        
        if args.task == "accuracy":

          if args.example:
            compare_to_default_accuracy_example(data, config['plans'])
          else:
            benchmark_times, non_default_times = compare_to_default_accuracy(benchmark, data, methods, args.plan_ids, config['plans'], config['default'])

            benchmark_times['benchmark'] = benchmark
            non_default_times['benchmark'] = benchmark
            if all_times.empty:
              all_times = benchmark_times
            else:
              all_times = pd.concat([all_times, benchmark_times])

            if all_non_default_times.empty:
              all_non_default_times = non_default_times
            else:
              all_non_default_times = pd.concat([all_non_default_times, non_default_times])

    if args.task == 'accuracy' and not args.example:
      # print(all_times)
      # print(all_non_default_times)

      # compute geometric mean of speedups
      gm = gmean(list(all_times['speedup'].values) + [1] * 9)
      min_speedup = all_times['speedup'].min()
      max_speedup = all_times['speedup'].max()

      print('Across All')
      print(f"Geometric Mean Speedup: {round(gm, 2)}")
      print(f"Min Speedup: {round(min_speedup, 2)}")
      print(f"Max Speedup: {round(max_speedup, 2)}")

      print('Non Default')
      gm = gmean(all_non_default_times['speedup'].values)
      min_speedup = all_non_default_times['speedup'].min()
      max_speedup = all_non_default_times['speedup'].max()

      print(f"Geometric Mean Speedup: {round(gm, 2)}")
      print(f"Min Speedup: {round(min_speedup, 2)}")
      print(f"Max Speedup: {round(max_speedup, 2)}")
