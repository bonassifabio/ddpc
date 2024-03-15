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

#%% 
import multiprocessing
from functools import partial
from pathlib import Path

import yaml

from gddpc.system import BenchmarkSystem
from openloop_test import build_benchmark_system, run_openloop_tests


def run_openloop_test_campaign(N_bars: list[int],
                               random_seed: list[int],
                               Niter: int,
                               slack_regs: list[float],
                               ddpc_params: dict,
                               system: BenchmarkSystem,
                               out_folder: Path = None):
    
    num_processes = multiprocessing.cpu_count() - 2

    experiment = partial(run_openloop_tests, 
                         Niter=Niter,
                         slack_regs=slack_regs,
                         ddpc_params=ddpc_params,
                         system=system,
                         omit_plots=True,
                         out_folder=out_folder)

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Map the task function to a range of input values
        for nb, rs in zip(N_bars, random_seed):
            pool.apply_async(experiment, args=(nb, rs))

        pool.close()
        pool.join()


if __name__ == "__main__":
    # Load params from default_campaign.yaml
    with open('default_campaign.yaml', 'r') as file:
        params = yaml.safe_load(file)
        params.pop('N_bar')
        base_seed = params.pop('random_seed')
        benchmark_system  = params.pop('benchmark_system')
        benchmark_params = params.pop('benchmark_params')
        ddpc_params = params.pop('ddpc_params')

    # Number of times each value of N_bar is tested (with different random seeds)
    # N_cycles = 200
    N_cycles = 1

    # Values of N_bar that we want to test
    target_N_bar = [119] + list(range(120, 240, 5)) + list(range(240, 300, 10)) + list(range(300, 750, 50)) + list(range(600, 1200, 100)) + list(range(1200, 10000, 1000)) + list(range(2000, 11000, 1000))
    target_N_bar = [ 150, 200 ]

    # Create list of rhos and seeds
    N_bars = []
    seeds = []

    # Build the benchmark system
    benchmark_system = build_benchmark_system(benchmark_system, benchmark_params)

    # Use a different seed for each experiment!
    for i in range(N_cycles):
        N_bars += target_N_bar
        seeds += [base_seed + i * params['Niter']**2] * len(target_N_bar) 

    run_openloop_test_campaign(N_bars, random_seed=seeds, **params, ddpc_params=ddpc_params, system=benchmark_system, out_folder=Path('out'))
