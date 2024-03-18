'''
Copyright (C) 2024 Fabio Bonassi and co-authors

This file is part of gddpc.

gddpc is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

gddpc is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU General Public License
along with gddpc.  If not, see <http://www.gnu.org/licenses/>.
'''

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TABLEAU_COLORS

from gddpc.controller import AbstractDDPC, Oracle
from gddpc.system import BenchmarkSystem
from gddpc.utils import numpy_seed


class QuadraticCriterion:
    def __init__(self, Q: np.ndarray, R: np.ndarray) -> None:
        """Quadratic criterion to be used for computing the performances of the controllers.See (25) in the paper.
            J = (y - r)' Q (y - r) + u' R u
            
        Args:
            Q (np.ndarray): The tracking cost matrix
            R (np.ndarray): The control cost matrix
        """
        self.Q = Q
        self.R = R

    def __call__(self, input: np.ndarray, output: np.ndarray, reference: np.ndarray) -> float:
        J = 0

        for t in range(output.shape[0]):
            et = output[t, :] - reference
            ut = input[t, :]
            stage = et.T @ self.Q @ et + ut.T @ self.R @ ut
            J += stage.item()

        return J
    
    
class SplitQuadraticCriterion:
    def __init__(self, Q: np.ndarray, R: np.ndarray) -> None:
        """Quadratic criterion to be used for computing the performances of the controllers.See (25) in the paper.
            J = (y - r)' Q (y - r) + u' R u
            
            The difference compared to QuadraticCriterion is that here we return the tracking cost and the control cost separately.

        Args:
            Q (np.ndarray): The tracking cost matrix
            R (np.ndarray): The control cost matrix
        """
        self.Q = Q
        self.R = R

    def __call__(self, input: np.ndarray, output: np.ndarray, reference: np.ndarray) -> tuple[float, tuple]:
        J_tr = 0
        J_co = 0

        for t in range(output.shape[0]):
            et = output[t, :] - reference
            ut = input[t, :]
            J_tr += (et.T @ self.Q @ et).item()
            J_co += (ut.T @ self.R @ ut).item()
        return J_tr, J_co


class DistanceFromOracle:
    def __init__(self, Q: np.ndarray, R: np.ndarray) -> None:
        """Dissimilarity from the Oracle MPC. See (26) in the paper.

        Args:
            Q (np.ndarray): The tracking cost matrix
            R (np.ndarray): The control cost matrix
        """
        self.Q = Q
        self.R = R

    def __call__(self, input: np.ndarray, output: np.ndarray, input_oracle: np.ndarray, output_oracle: np.ndarray) -> tuple[float, tuple]:
        J_tr = 0
        J_co = 0

        for t in range(output.shape[0]):
            et = output[t, :] - output_oracle[t, :]
            ut = input[t, :] - input_oracle[t, :]
            J_tr += (et.T @ self.Q @ et).item()
            J_co += (ut.T @ self.R @ ut).item()
        return J_tr, J_co


def compare_controllers_openloop(controllers: list[AbstractDDPC],
                                 oracle: Oracle,
                                 system: BenchmarkSystem,
                                 x0: np.ndarray,
                                 reference: np.ndarray,
                                 rho: int,
                                 criterion: callable,
                                 distance_oracle: callable,
                                 random_seed: int = None,
                                 ax: plt.Axes = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compare the DDPC controllers using the same context data.

    Args:
        controllers (list[AbstractDDPC]): List containing all the DDPCs to be tested.
        oracle (Oracle): Oracle MPC controller.
        system (BenchmarkSystem): Benchmark system.
        x0 (np.ndarray): Initial condition.
        reference (np.ndarray): Setpoint to be tracked.
        rho (int): Length of the context window.
        criterion (callable): Criterion to be used for evaluating the controllers.
        distance_oracle (callable): Criterion for evaluating the dissimilarity from the Oracle.
        random_seed (int, optional): Random seed for reproducibility. Defaults to None.
        ax (plt.Axes, optional): Matplotlib axes on which the open-loop trajectories should be plotted. Defaults to None.

    Returns:
        np.ndarray: Tracking criterion for each controller (tracking component).
        np.ndarray: Control criterion for each controller (control component).
        np.ndarray: Dissimilarity from the Oracle for each controlle (tracking component).
        np.ndarray: Dissimilarity from the Oracle for each controller (control component).
        np.ndarray: Constraint violation rate.
        np.ndarray: Mean Squared value of the slack variable Δy.
        np.ndarray: Maximum value of the slack variable Δy.
    """
    
    criteria_tr = np.zeros((len(controllers),))    
    criteria_co = np.zeros((len(controllers),))   
    distance_oracle_tr = np.zeros((len(controllers),))   
    distance_oracle_co = np.zeros((len(controllers),))   
    violation_rate = np.zeros((len(controllers),))     
    slack_mse = np.zeros((len(controllers),))       
    slack_max = np.zeros((len(controllers),))   

    # NOTE: In this test we adopt zero initial state, and null input in the context window.
    # This way, the context data only depends on x0 and on the measurement noise of the output measurement in y_ctx.
    u_ctx, y_ctx, xf = system.get_online_context(rho, x0, random_seed=random_seed, exc_u_std=0.0 * system.exc_u_std)

    # Make sure that ref_ is a vector of shape (n_out,)
    if type(reference) is np.array: 
        ref_ = reference.reshape((-1,))
    else:
        ref_ = np.array(reference)

    # Solve the Oracle's optimization problem, and get the open-loop output trajectory
    with numpy_seed(random_seed):
        _ = oracle.mpc_step(reference=ref_, x0=xf)
        y_oracle = system.forced_motion(oracle._u.value, x0=xf, noise_y_std=0.0)[0]
    
    # Solve the DDPC's optimization problem for every controller
    for d in controllers:
        _ = d.mpc_step(reference=ref_, Up=u_ctx, Yp=y_ctx)

    # Get the performance indices for all the DDPCs
    for i, d in enumerate(controllers):       
        with numpy_seed(random_seed):
            if d._u.value is None:
                # Unfeasible problem....
                y_gt = np.inf * np.ones((d.T, system.n_out))
                criteria_tr[i], criteria_co[i] =  np.nan, np.nan
                distance_oracle_tr[i], distance_oracle_co[i] =  np.nan, np.nan
                violation_rate[i] = np.nan
                slack_mse[i] = np.nan
            else:
                y_gt = system.forced_motion(d._u.value, x0=xf, noise_y_std=0.0)[0]
                criteria_tr[i], criteria_co[i] = criterion(d._u.value, y_gt, ref_)
                distance_oracle_tr[i], distance_oracle_co[i] = distance_oracle(d._u.value, y_gt, oracle._u.value, y_oracle)
                violation_rate[i] = np.mean(np.logical_or(y_gt < d._y_lb.value, y_gt > d._y_ub.value))
                slack_mse[i] = np.mean(d._delta.value**2)
                slack_max[i] = np.max(np.abs(d._delta.value))
        
        # Plot if ax is not None
        if ax is not None and d._u.value is not None:
            color = list(TABLEAU_COLORS.keys())[i]
            ax.plot(y_gt, label=d.name, color=color)

    return criteria_tr, criteria_co, distance_oracle_tr, distance_oracle_co, violation_rate, slack_mse, slack_max

    
def batch_compare_controllers_openloop(controllers: list[AbstractDDPC],
                                       oracle: Oracle,
                                       system: BenchmarkSystem,
                                       reference: np.ndarray,
                                       rho: int,
                                       criterion: callable,
                                       distance_oracle: callable,
                                       Niter: int,
                                       random_seed: int = None,
                                       enable_plot: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batch comparison of the DDPC controllers using different realizations of the context data.

    Args:
        controllers (list[AbstractDDPC]): List containing all the DDPCs to be tested.
        oracle (Oracle): Oracle MPC controller.
        system (BenchmarkSystem): Benchmark system.
        x0 (np.ndarray): Initial condition.
        reference (np.ndarray): Setpoint to be tracked.
        rho (int): Length of the context window.
        criterion (callable): Criterion to be used for evaluating the controllers.
        distance_oracle (callable): Criterion for evaluating the dissimilarity from the Oracle.
        Niter (int): Number of iterations, i.e. realizations of the context data.
        random_seed (int, optional): Random seed for reproducibility. Defaults to None.
        enable_plot (bool, optional): Whether to plot the open-loop trajectories. Defaults to False.

    Returns:
        np.ndarray: Tracking criterion for each controller (tracking component) over the different realizations (shape: (len(controllers), Niter)).
        np.ndarray: Control criterion for each controller (control component) over the different realizations (shape: (len(controllers), Niter)).
        np.ndarray: Dissimilarity from the Oracle for each controlle (tracking component) over the different realizations (shape: (len(controllers), Niter)).
        np.ndarray: Dissimilarity from the Oracle for each controller (control component) over the different realizations (shape: (len(controllers), Niter)).
        np.ndarray: Constraint violation rate over the different realizations (shape: (len(controllers), Niter)).
        np.ndarray: Mean Squared value of the slack variable Δy over the different realizations (shape: (len(controllers), Niter)).
        np.ndarray: Maximum value of the slack variable Δy over the different realizations (shape: (len(controllers), Niter)).
    """
    if enable_plot:
        _, ax = plt.subplots()
        _y_lb = controllers[0]._y_lb.value.item()
        _y_ub = controllers[0]._y_ub.value.item()
        ax.plot([0, controllers[0].T], [reference, reference], 'r:')
        ax.plot([0, controllers[0].T], [_y_lb, _y_lb], 'k--')
        ax.plot([0, controllers[0].T], [_y_ub, _y_ub], 'k--')
    else:
        ax = None
        
    criteria_tr = np.zeros((Niter, len(controllers)))    
    criteria_co = np.zeros((Niter, len(controllers)))     
    distance_oracle_tr = np.zeros((Niter, len(controllers)))  
    distance_oracle_co = np.zeros((Niter, len(controllers)))    
    violation_rate = np.zeros((Niter, len(controllers)))  
    slack_mse = np.zeros((Niter, len(controllers)))
    slack_max = np.zeros((Niter, len(controllers)))

    for i in range(Niter):
        mseed = random_seed + i
        with numpy_seed(random_seed - i):
            # We are just interested in different realizations of the measurement noise
            x0 = 0.0 * np.random.randn(system.sys.A.shape[0], 1)

        criteria_tr[i, :], criteria_co[i, :], distance_oracle_tr[i, :], distance_oracle_co[i, :], \
        violation_rate[i, :], slack_mse[i, :], slack_max[i, :] = \
            compare_controllers_openloop(controllers=controllers,
                                        oracle=oracle,
                                        system=system,
                                        x0=x0,
                                        reference=reference,
                                        rho=rho,
                                        criterion=criterion,
                                        distance_oracle=distance_oracle,
                                        random_seed=mseed,
                                        ax=ax)
        
    return criteria_tr, criteria_co, distance_oracle_tr, distance_oracle_co, violation_rate, slack_mse, slack_max
