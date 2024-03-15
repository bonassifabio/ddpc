'''
Copyright (C) 2024 Fabio Bonassi and co-authors

This file is part of gddpc.

gddpc is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ssnet is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU General Public License
along with gddpc.  If not, see <http://www.gnu.org/licenses/>.
'''

import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from gddpc.controller import DDPC, AbstractDDPC, Oracle
from gddpc.system import (BenchmarkSystem, ddpc_system, overshooting_system,
                          simple_system)
from gddpc.testers import (DistanceFromOracle, SplitQuadraticCriterion,
                           batch_compare_controllers_openloop)

warnings.simplefilter(action='ignore', category=FutureWarning)


def build_benchmark_system(name: str, params: dict = {}) -> BenchmarkSystem:
    """Build a benchmark system based on its name and parameters

    Args:
        name (str): The name of the system
        params (dict, optional): The parameters of the system. Defaults to {}.

    Returns:
        BenchmarkSystem: The benchmark system
    """
    if name == 'overshooting_system' or name == 'overshooting':
        return BenchmarkSystem(overshooting_system(), **params)
    elif name == 'simple_system' or name == 'simple':
        return BenchmarkSystem(simple_system(), **params)
    elif name == 'ddpc_system' or name == 'ddpc':
        return BenchmarkSystem(ddpc_system(), **params)
    else:
        raise ValueError('The system name is not valid')
    

def print_or_save(J: np.ndarray, 
                  controllers: list[AbstractDDPC], 
                  title: str,
                  N_bar: int,
                  out_folder: Path = None,
                  plot_opts: dict = {}) -> None:
    """Display J in a figure or save it to a .png file

    Args:
        J (np.ndarray): The thing to display. It should have shape (..., n_controllers)
        controllers (list[AbstractDDPC]): List of the controllers.
        title (str): Title of the plot.
        N_bar (int): Length of the training dataset.
        out_folder (Path, optional): Output folder. If None, the figure is displayed and not saved.
        plot_opts (dict, optional): Options for the plot. Defaults to {}.
    """
    
    def plot_mean_std(x, mean, std, label, ax):
        # Auxiliary function to plot mean and standard deviation
        ax.plot(x, mean, label=label)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)

    # Separate SPC, C-SPC and DeePC controllers
    controller_names = [ctr.name for ctr in controllers]
    acausal_idx = controller_names.index('SPC')
    causal_idx = controller_names.index('C-SPC')

    J_acausal = J[..., acausal_idx]
    J_causal = J[..., causal_idx]
    mask = np.ones(J.shape[1], dtype=bool)
    mask[acausal_idx] = False
    mask[causal_idx] = False
    J_deepc = J[..., mask]
    slack_deepc = np.array([controllers[i].slack_regularization for i in range(len(controllers)) if mask[i]])

    f, ax = plt.subplots(figsize=plot_opts.get('figsize', (10, 6)))
    plot_mean_std(np.array((0, slack_deepc.max())), 
                  J_acausal.mean(axis=0) * np.ones((2,)), 
                  J_acausal.std(axis=0) * np.ones((2,)), 'SPC', ax)
    plot_mean_std(np.array((0, slack_deepc.max())),
                    J_causal.mean(axis=0) * np.ones((2,)), 
                    J_causal.std(axis=0) * np.ones((2,)), 'C-SPC', ax)
    
    # Sort slack_deepc in ascending order along with corresponding means and stds
    sorted_indices = np.argsort(slack_deepc)
    slack_deepc = slack_deepc[sorted_indices]
    J_deepc_mean = J_deepc.mean(axis=0)[sorted_indices]
    J_deepc_std = J_deepc.std(axis=0)[sorted_indices]

    plot_mean_std(slack_deepc,
                  J_deepc_mean,
                  J_deepc_std,
                  'DeePC', ax)

    ax.set_xscale('log')
    ax.set_yscale(plot_opts.get('yscale', 'linear'))
    ax.set_xlabel(plot_opts.get('slack_label', 'Slack regularization'))
    ax.set_title(title)
    ax.legend()

    if out_folder is not None:
        f.savefig(out_folder / f'{title} - N_bar={N_bar}.png')
        plt.close(f)
    else:
        plt.show(block=False)
        
    
def run_openloop_tests(N_bar: int,
                       random_seed: int,
                       Niter: int,
                       ddpc_params: dict,
                       slack_regs: list[float],
                       system: BenchmarkSystem,
                       omit_plots: bool = False,
                       plot_opts: dict = {},
                       out_folder: Path = None) -> pd.DataFrame:
    """Run an openloop test. Here we instantiate the DDPCs we are interested in, run `batch_compare_controllers_openloop` on them, and export the results.

    Args:
        N_bar (int): Length of the training dataset we wish to generate.
        random_seed (int): Random seed for reproducibility.
        Niter (int): Number of realizations of the context window to consider.
        slack_regs (list[float]): List of slack regularizations to test. 
        T (int): Prediction horizon.
        rho (int): Context window size.
        Q (float | np.ndarray): State cost
        R (float | np.ndarray): Control cost
        S (float | np.ndarray): Slack cost
        ref (float | np.ndarray): Reference
        system (BenchmarkSystem): The Benchmark System to consider
        omit_plots (bool, optional): If True, the plots are not displayed. Defaults to False.
        plot_opts (dict, optional): Options for the plots. Defaults to {}.
        out_folder (Path, optional): Output folder. If None, the results are not saved to disk. Defaults to None.

    Returns:
        pd.DataFrame: The results of the campaign 
    """

    u_tr, y_tr = system.get_training_data(N_bar=N_bar, random_seed=random_seed)

    # Set the seed to something else.
    seed_ = random_seed + np.int64(100 * np.log10(N_bar))

    # If ddpc_params['y_bound'] is None, set y_lb and y_ub to -np.inf and np.inf
    y_bound = ddpc_params.pop('y_bound', None)
    if y_bound is None:
        ddpc_params['y_lb'] = -np.inf
        ddpc_params['y_ub'] = np.inf
    else:
        ddpc_params['y_lb'] = y_bound[0]
        ddpc_params['y_ub'] = y_bound[1]

    # SPC: DDPC without slack variable and non-causal predictor
    d_spc = DDPC(u_tr, y_tr, T=ddpc_params['T'], rho=ddpc_params['rho'], causal=False, solver=ddpc_params['solver'], name='SPC')
    d_spc.mpc_setup(u_lb=system.u_bound[0], 
                    u_ub=system.u_bound[1],
                    y_lb=ddpc_params['y_lb'],
                    y_ub=ddpc_params['y_ub'],
                    Q= ddpc_params['Q'],
                    R= ddpc_params['R'],
                    S= ddpc_params['S'],
                    slack_in_stage_cost=False,
                    slack_in_constraint=False)

    # C-SPC: DDPC without slack variable and causal predictor
    d_cspc = DDPC(u_tr, y_tr, T=ddpc_params['T'], rho=ddpc_params['rho'], causal=True, solver=ddpc_params['solver'], name='C-SPC')
    d_cspc.mpc_setup(u_lb=system.u_bound[0], 
                     u_ub=system.u_bound[1],
                     Q= ddpc_params['Q'],
                     R= ddpc_params['R'],
                     S= ddpc_params['S'],
                     slack_in_stage_cost=False, 
                     slack_in_constraint=False)

    controllers = [d_spc, d_cspc]

    # DeePC: (23) where slack variable is both in the stage cost and in the constraint (since it affects prediction)
    # This is equivalent to using the class DeePC, but more efficient from a computational point of view.
    for lambda2 in slack_regs:
        d_deepc = DDPC(u_tr, y_tr, T=ddpc_params['T'], rho=ddpc_params['rho'], solver=ddpc_params['solver'], name=f'DeePC-λ={lambda2}')
        d_deepc.mpc_setup(u_lb=system.u_bound[0], 
                          u_ub=system.u_bound[1],
                          Q= ddpc_params['Q'],
                          R= ddpc_params['R'],
                          S= ddpc_params['S'],
                          slack_in_stage_cost=True,
                          slack_in_constraint=True,
                          lambda2=lambda2)
        controllers.append(d_deepc)


    # The oracle has access to the actual system dynamics
    d_oracle = Oracle(system.sys, T=ddpc_params['T'], solver=ddpc_params['solver'], name='Oracle')
    d_oracle.mpc_setup(u_lb=system.u_bound[0], 
                       u_ub=system.u_bound[1],
                       y_lb=ddpc_params['y_lb'],
                       y_ub=ddpc_params['y_ub'],
                       Q = ddpc_params['Q'],
                       R = ddpc_params['R'],
                       S = ddpc_params['S'],)

    # Make sure that Q and R are as expected
    Q = ddpc_params['Q'] if isinstance(ddpc_params['Q'], np.ndarray) else ddpc_params['Q'] * np.eye(system.n_out)
    R = ddpc_params['R'] if isinstance(ddpc_params['R'], np.ndarray) else ddpc_params['R'] * np.eye(system.n_in)

    # Criteria
    criterion = SplitQuadraticCriterion(Q, R)
    distance_criterion = DistanceFromOracle(Q, R)

    J_tr, J_co, Delta_tr, Delta_co, violations, slack_mse, slack_max \
        = batch_compare_controllers_openloop(controllers=controllers,
                                             oracle=d_oracle,
                                             system=system,
                                             reference=ddpc_params['ref'],
                                             rho=ddpc_params['rho'],
                                             Niter=Niter,
                                             criterion=criterion,
                                             distance_oracle=distance_criterion,
                                             random_seed=seed_) 

    # Save all the results in a dataframe 
    df = pd.DataFrame(columns=['Controller', 'E[J_tr]', 'E[J_co]', 'E[Delta_tr]', 'E[Delta_co]', 'E[violations]', 'E[slack_mse]', 'E[slack_max]',
                                            'std[J_tr]', 'std[J_co]', 'std[Delta_tr]', 'std[Delta_co]', 'std[violations]', 'std[slack_mse]', 'std[slack_max]'])
    
    for i, ctr in enumerate(controllers):
        # Create a new row into the dataframe
        # Note that since we do not have a slack variable in SPC and C-SPC, we set the regularization to np.inf for convenience.
        reg = ctr.slack_regularization if ctr.name.startswith('DeePC') else np.inf
        name = 'DeePC' if ctr.name.startswith('DeePC') else ctr.name

        df = df._append({'Controller': name,
                        'lambda': reg,
                        'E[J_tr]': J_tr[:, i].mean(),
                        'E[J_co]': J_co[:, i].mean(),
                        'E[Delta_tr]': Delta_tr[:, i].mean(),
                        'E[Delta_co]': Delta_co[:, i].mean(),
                        'E[violations]': violations[:, i].mean(),
                        'E[slack_mse]': slack_mse[:, i].mean(),
                        'E[slack_max]': slack_max[:, i].mean(),
                        'std[J_tr]': J_tr[:, i].std(),
                        'std[J_co]': J_co[:, i].std(),
                        'std[Delta_tr]': Delta_tr[:, i].std(),
                        'std[Delta_co]': Delta_co[:, i].std(),
                        'std[violations]': violations[:, i].std(),
                        'std[slack_mse]': slack_mse[:, i].std(),
                        'std[slack_max]': slack_max[:, i].std()}, 
                        ignore_index=True)
        
        # Sort DeePCs by value of lambda
        df.loc[df['Controller'].str.startswith('DeePC'), :] = df.loc[df['Controller'].str.startswith('DeePC'), :].sort_values(by='lambda', ascending=True)

    if out_folder is not None:
        # Pause by a random amount of time to avoid overwriting files (0.001s to 1s)
        time.sleep(np.random.rand())
        if not out_folder.exists():
            out_folder.mkdir()
        
        i = 0
        while True:
            new_folder = out_folder / f'N_bar{N_bar}_{i}'
            if (out_folder / f'N_bar{N_bar}_{i}').exists():
                i += 1
            else:
                new_folder.mkdir()
                break

        df.to_csv(new_folder / 'results.csv')
    else:
        new_folder = None
        print(df)

    # Plot the results
    if not omit_plots:
        plot_opts_v = plot_opts
        plot_opts_v['yscale'] = 'linear'

        print_or_save(J_tr, controllers, plot_opts.get('J_tr_title', 'Tracking criterion'), N_bar, new_folder, plot_opts)
        print_or_save(J_co, controllers, plot_opts.get('J_co_title', 'Tracking criterion'), N_bar, new_folder)
        print_or_save(Delta_tr, controllers, plot_opts.get('Delta_tr_title', 'Distance from Oracle (tracking)'), N_bar, new_folder, plot_opts)
        print_or_save(Delta_co, controllers, plot_opts.get('Delta_co_title', 'Distance from Oracle (control)'), N_bar, new_folder, plot_opts)
        print_or_save(violations, controllers, plot_opts.get('Violation_title', 'Violation rate'), N_bar, new_folder, plot_opts_v)
        print_or_save(slack_mse, controllers, plot_opts.get('slack_ms_title', 'Mean Squared slack'), N_bar, new_folder, plot_opts)
        print_or_save(slack_max, controllers, plot_opts.get('slack_max_title', 'Maximum slack'), N_bar, new_folder, plot_opts)

        if out_folder is None:
            plt.show()
    
    return df


if __name__ == "__main__":
    # Read the default parameters from default_campaign.yaml
    with open('default_campaign.yaml', 'r') as file:
        params = yaml.safe_load(file)
        benchmark_system = params.pop('benchmark_system')
        benchmark_params = params.pop('benchmark_params')
        ddpc_params = params.pop('ddpc_params')

        system = build_benchmark_system(benchmark_system, benchmark_params)

    df = run_openloop_tests(**params, ddpc_params=ddpc_params, system=system, out_folder=Path('out'))
