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

import control
import control.matlab
import numpy as np

from gddpc.utils import numpy_seed


class BenchmarkSystem:
    def __init__(self, 
                 sys: control.ss,
                 noise_y_std: float = 0.1, 
                 exc_u_std: float = 0.0,
                 u_bound: tuple[float, float] = (-1, 1)) -> None:
        """Benchmark System considered in the numerical example. This wrapper allows to collect training data, 
           test data and context data for the DDPC.

        Args:
            sys (control.ss): The LTI system to be used. 
            noise_y_std (float, optional): Standard Deviation of the measurement noise. Defaults to 0.1.
            exc_u_std (float, optional): Standard Deviation of the Gaussian excitation signal. Defaults to 0.0.
            u_bound (tuple[float, float], optional): Constraint on the input variable. Defaults to (-1, 1).
        """
        self.sys =  sys
        self.n_in = sys.ninputs
        self.n_out = sys.noutputs
        self.noise_y_std = noise_y_std
        self.exc_u_std = exc_u_std
        self.u_bound = u_bound

    def forced_motion(self, 
                      u: np.ndarray, 
                      x0: np.ndarray = None, 
                      noise_y_std: float = None,
                      random_seed: int = None) -> tuple[np.ndarray, np.ndarray]:
        """Simulate the system under a given input signal.

        Args:
            u (np.ndarray): The input signal. Shape (T, n_in).
            x0 (np.ndarray, optional): Initial condition. Shape (n_states,). Defaults to None. 
            noise_y_std (float, optional): Standard Deviation of the measurement noise. Defaults to None.
            random_seed (int, optional): Random seed for reproducibility. Defaults to None.

        Returns:
            np.ndarray: The output signal. Shape (T, n_out).
            np.ndarray: The final state of the system. Shape (n_states,).
        """

        # The control toolbox wants the input to be in the form (n_in, T)
        u_ = u
        if u_.ndim == 1:
            u_ = u_.reshape(1, -1)
        if u_.shape[1] == self.n_in:
            u_ = u_.T
        
        with numpy_seed(random_seed):
            # Default initial condition
            x0 = np.zeros((self.sys.A.shape[0], 1)) if x0 is None else x0

            # Forced response of the system
            resp = control.forced_response(self.sys, U=u_, X0=x0) 

            # Make sure the output ha shape (T, n_out)
            y = resp[1]
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            elif y.shape[0] == self.n_out:
                y = y.T
            
            # If noise_y_std is not None, we add noise to the output
            if noise_y_std is not None and noise_y_std > 1e-12:
                y += noise_y_std * np.random.randn(*y.shape)
            elif noise_y_std is None:
                y += self.noise_y_std * np.random.randn(*y.shape)

        # Return also the final state
        xf = resp[2][:, -1]
        return y, xf
    

    def get_training_data(self, 
                          N_bar: int, 
                          x0: np.ndarray = None,
                          exc_u_std: float = None,
                          random_seed: int = None) -> tuple[np.ndarray, np.ndarray]:
        """Generate the training data

        Args:
            N_bar (int): The number of samples to be generated. 
            x0 (np.ndarray, optional): The initial state. Defaults to None.
            exc_u_std (float, optional): Override the standard deviation of the excitation signal. Defaults to None.
            random_seed (int, optional): Random seed for reproducibility. Defaults to None.

        Returns:
            np.ndarray: Trainining input. Shape (N_bar, n_in).
            np.ndarray: Training output. Shape (N_bar, n_out).
        """
        # If we specify the excitation standard deviation, we use it, otherwise we use the default
        u_std = self.exc_u_std if exc_u_std is None else exc_u_std
        
        with numpy_seed(random_seed - 1 if random_seed is not None else None):
            u = u_std * np.random.randn(N_bar, self.n_in)
            u = np.clip(u, self.u_bound[0], self.u_bound[1]) #(shape: (N_bar, n_in))

        # Get the output
        y, _ = self.forced_motion(u, x0=x0) # (y shape: (N_bar, n_out))    

        # Print the SNR ratio
        print(f'Training set generated [N_bar = {N_bar}]. SNR = {np.mean(y ** 2, axis=0) / self.noise_y_std ** 2}')
        return u, y


    def get_test_data_with_context(self,
                                   T: int,
                                   rho: np.ndarray,
                                   exc_u_std: float = None, 
                                   u_bias: np.ndarray = 0.0,
                                   random_seed: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate the test data with context

        Args:
            Tval (int): The length of the test data.
            rho (int): The length of the context window.
            exc_u_std (float, optional): Override the standard deviation of the excitation signal. Defaults to None.
            u_bias (np.ndarray, optional): Bias of the excitation signal. Shape: (n_in,). Defaults to 0.0.
            random_seed (int, optional): Random seed for reproducibility. Defaults to None.

        Returns:
            np.ndarray: Context input. Shape (rho, n_in).
            np.ndarray: Context output. Shape (rho, n_out).
            np.ndarray: Test input. Shape (Tval, n_in).
            np.ndarray: Test output. Shape (Tval, n_out).
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        u_std_context = self.exc_u_std
        u_std_exp = self.exc_u_std if exc_u_std is None else exc_u_std

        
        with numpy_seed(random_seed):
            u = np.zeros((rho + T, self.n_in))
            # The input is identically null in the "context window"
            u[:rho, :] = u_std_context * np.random.randn(rho, self.n_in)
            # The input is u = u_bias + exc_u_std * N(0, 1) after this context window
            u[rho:, :] = u_bias + u_std_exp * np.random.randn(T, self.n_in)
            u = np.clip(u, self.u_bound[0], self.u_bound[1])

            # Random initial conditions
            x0 = 0.5*np.random.randn(self.sys.A.shape[0], 1)

            # In the test/validation dataset don't want noise after the context window! 
            y, _ = self.forced_motion(u, x0=x0, noise_y_std=0.0)

            # ... but of course the noise is still there in the context window!
            y[:rho, :] += self.noise_y_std * np.random.randn(rho, self.n_out)
        
        return u[:rho, :], y[:rho, :], u[rho:, :], y[rho:, :]
    
    
    def get_online_context(self, 
                           rho: int, 
                           x0: np.ndarray,
                           burn: int = 0,
                           exc_u_std: float = None,
                           bias_u: np.ndarray = 0.0,
                           random_seed: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate the online context data for testing the DDPC

        Args:
            rho (int): The context window length
            x0 (np.ndarray): The initial state
            burn (int, optional): How many samples we throw away before getting the context window. Might be done for reproducibility, allowing for the transients due to initial conditions. Defaults to 0.
            exc_u_std (float, optional): Override the default Standard Deviation of the input signal. Defaults to None.
            bias_u (float, optional): Bias of the input signal throughout the context window. Might be used if we want to start from nonnull initial conditions. Shape: (n_in,). Defaults to 0.0.
            random_seed (int, optional): Random seed for reproducibility. Defaults to None.

        Returns:
            np.ndarray: Context input. Shape (rho, n_in).
            np.ndarray: Context output. Shape (rho, n_out).
            np.ndarray: Final state of the system. Shape (n_states,).
        """

        with numpy_seed(random_seed):
            u_std = self.exc_u_std if exc_u_std is None else exc_u_std
            u = bias_u + u_std * np.random.randn(rho + burn, self.n_in)
            u = np.clip(u, self.u_bound[0], self.u_bound[1])

            y_seed = np.random.randint(0, 1000)
            y, xf = self.forced_motion(u, x0=x0, random_seed=y_seed)

        return u[-rho:, :], y[-rho:, :], xf
    

    def get_equilibrium(self, u_bar: np.ndarray | tuple) -> tuple[np.ndarray, np.ndarray]:
        """Get the equilibrium of the system, i.e. x_bar and u_bar such that
                x_bar = A @ x_bar + B @ u_bar
                y_bar = C @ x_bar + D @ u_bar

        Args:
            u_bar (np.ndarray | tuple): The constant input

        Returns:
            np.ndarray: The output at the equilibrium
            np.ndarray: The state at the equilibrium
        """
        if type(u_bar) == tuple:
            u_bar_ = np.array(u_bar).reshape((self.n_in, 1))
        else:                               
            u_bar_ = u_bar.reshape((self.n_in, 1))

        x_bar = np.linalg.inv(np.eye(self.sys.A.shape[0]) - self.sys.A) @ self.sys.B @ u_bar_.reshape((self.n_in, 1))
        y_bar = self.sys.C @ x_bar.reshape((-1, 1)) + self.sys.D @ u_bar_.reshape((self.n_in, 1))
        return y_bar.A1, x_bar.A1


def overshooting_system() -> control.ss:
    """Overshooting system characterized by a large overshoot in the step response and unit gain.
    """
    s = control.TransferFunction.s
    G = 0.5525 * (1+12.5*s) / (s*s + 0.5*s + 0.5525) * 1/ (1+10*s)

    Gd = control.matlab.c2d(G, 0.25, 'zoh')
    ss_form, _ = control.canonical_form(control.ss(Gd))
    return ss_form


def simple_system() -> control.ss:
    A = np.array([[0.9, 0.1], [-0.05, 0.45]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])
    D = np.array([[0]])
    return control.ss(A, B, C, D, 1.0)


def ddpc_system() -> control.ss:
    """Benchmark system used in (Mattson and Schön, 2023) and in (Mattson, Bonassi, Breschi, Schön, 2024)
    """
    A = np.array([[1.2,  -0.6,  -0.4], [0.5, 0, 0], [0, 0.5, 0]])
    B = np.array([[1], [0], [0]])
    C = np.array([[0.5, -0.8, 0.4]])
    D = np.array([[0]])
    return control.ss(A, B, C, D, 1.0)
