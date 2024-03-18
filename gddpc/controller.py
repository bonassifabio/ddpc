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
import cvxpy as cp
import numpy as np
import scipy.linalg as la


def cp_flatten(x: cp.Variable) -> np.ndarray:
    """Flatten a cvxpy variable into a numpy array

    Args:
        x (cp.Variable): The cvxpy variable to be flattened

    Returns:
        np.ndarray: The flattened variable
    """
    return cp.reshape(x, (-1, 1), order='C')


def cp_unflatten(x: cp.Variable, last_dim: int) -> cp.Variable:
    """Reshape a cvxpy variable

    Args:
        x (cp.Variable): The cvxpy variable to be reshaped
        last_dim (int): The last dimension

    Returns:
        cp.Variable: The reshaped variable
    """
    return cp.reshape(x, (-1, last_dim), order='C')


def hankel(X: np.ndarray, T: int, N: int = -1, start: int = 0) -> np.ndarray:
    """Build the Hankel matrix of the given data

    Args:
        X (np.ndarray): Data matric of shape (N_bar, n)
        T (int): Number of shifts along the axis 0 (prediction horizon)
        N (int): Number of shifts along the axis 1 (n. sequences)

    Returns:
        np.ndarray: The Hankel matrix
    """
    n = X.shape[1]
    X_ = X[start:, :].flatten()    # (shape: ((N_bar - start)*n, ))

    # Create the Hankel matrix
    H = np.lib.stride_tricks.sliding_window_view(X_, T*n, axis=0, writeable=False)[::n, :].T
    return H[:, :N] if N > 0 else H


class AbstractDDPC:
    def __init__(self, 
                U_train: np.ndarray,   # (shape: (N_bar, n_in))
                Y_train: np.ndarray,   # (shape: (N_bar, n_out))
                T: int = 50,           # Prediction horizon
                rho: int = 5,          # Length of the regression window / past horizon
                name: str = 'DDPC') -> None:
        """Abstract class for the DDPC algorithms
        
        Args:
            U_train (np.ndarray): Training input (shape: (N_bar, n_in))
            Y_train (np.ndarray): Training output (shape: (N_bar, n_out))
            T (int, optional): Prediction horizon. Defaults to 50.
            rho (int, optional): Length of the regression window. Defaults to 5.
            name (str, optional): The name of the model. Defaults to 'DDPC'.
        """

        self.U_train = U_train
        self.Y_train = Y_train
        self.T = T
        self.N = U_train.shape[0] - rho - T + 1     # Number of training data
        self.n_in = U_train.shape[1]
        self.n_out = Y_train.shape[1]
        self.rho = rho
        self.name = name

    def openloop_simulation(self, sys: control.StateSpace, x0: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Simulate the open-loop response of the system

        Args:
            sys (control.StateSpace): The discrete-time system to be simulated
            x0 (np.ndarray): Initial state of the system (shape: (n, ))
            u (np.ndarray): The input data (shape: (T, n_in))

        Returns:
            np.ndarray: The predicted output data (shape: (T, n_out))
        """
        return control.forced_response(sys, X0=x0, U=u.T)[1].T
    
    @property
    def slack_regularization(self):
        raise NotImplementedError
    
    def predict(self):
        raise NotImplementedError
    
    def mpc_setup(self, **kwargs):
        raise NotImplementedError
    
    def mpc_step(self, reference: np.ndarray, Up: np.ndarray, Yp: np.ndarray, x0: np.ndarray, slack_reg_override: float) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    

class DDPC(AbstractDDPC):
    def __init__(self, 
                 U_train: np.ndarray,   # (shape: (N_bar, n_in))
                 Y_train: np.ndarray,   # (shape: (N_bar, n_out))
                 T: int = 50,           # Prediction horizon
                 rho: int = 5,          # Length of the regression window / past horizon
                 causal: bool = False,
                 solver: str = None,
                 name: str = 'DDPC') -> None:
        """'Generalized' DDPC algorithm, see (Mattson, Bonassi, Breschi, Schön, 2024) equation (23)

        Args:
            U_train (np.ndarray): Training input (shape: (N_bar, n_in))
            Y_train (np.ndarray): Training output (shape: (N_bar, n_out))
            T (int, optional): Prediction horizon. Defaults to 50.
            rho (int, optional): Length of the regression window. Defaults to 5.
            causal (bool, optional): Enforce the causal structure of the regressor. Defaults to False.
            solver (str, optional): The solver to be used. To let CVXPY choose the best solver, set it to None. Defaults to None.
            name (str, optional): The name of the model. Defaults to 'DDPC'.
        """
        super().__init__(U_train, Y_train, T, rho, name)
        self.__causal = causal
        self.solver = solver
        
        # Check in which data-regime we are.
        # See (Mattson, Bonassi, Breschi, Schön, 2024) Section V.C
        match get_data_regime(U_train.shape[0], T, rho, self.n_in, self.n_out):
            case 0:
                print('Warning: the number of training samples is smaller than the number of samples needed for the model -> Σ only pseudo-invertible')
            case -1:
                print('The regression matrix is not full row-rank!')
        
        # Fit the multi-step predictive model to the training data
        self.fit_multistep_linear(U_train, Y_train)
    
        
    @property
    def causal(self):
        return self.__causal
    
    def _fit_theta_ls(self, Phi, Yf):
        """Solve the Least Square problem"""
        return Yf @ la.pinv(Phi)
    

    def _fit_theta_causal(self, Phi, Yf):
        """Solve the *causal* Least Square problem"""
        Theta = np.zeros((self.n_out * self.T, Phi.shape[0]))

        phi_curs = self.rho * (self.n_in + self.n_out) + self.n_in

        # At every step, we solve the LS problem for the current Φ (contains all the rows of phi up to "t")
        # Theta is built row by row
        for t in range(self.T):
            Yf_t = Yf[t*self.n_out:(t+1)*self.n_out, :]
            Phi_t = Phi[:phi_curs, :]
            self._Phi = Phi_t
            Theta_t = Yf_t @ la.pinv(Phi_t)
            Theta[t*self.n_out:(t+1)*self.n_out, :Theta_t.shape[1]] = Theta_t
            phi_curs += self.n_in

        return Theta
    
    def fit_multistep_linear(self, U_train: np.ndarray, Y_train: np.ndarray) -> np.ndarray:
        """Build the data matrices for DDPC

        Args:
            U_train (np.ndarray): Training input data matrix of shape (N_bar, n_in)
            Y_train (np.ndarray): Training output data matrix of shape (N_bar, n_out)
            rho (int): Length of the regression window
            T (int):  Prediction horizon that will be adopted
        """

        # Number of shift in the Hankel matrix (N in the paper)
        N = self.N
        U_past = hankel(U_train, self.rho, N)
        Y_past = hankel(Y_train, self.rho, N)
        U_future = hankel(U_train, self.T, N, start=self.rho)

        # Fit the model to the data by solving the least squares problem
        # The regressors will be Phi = [ U_past; Y_past; U_future ]
        # See (8) and (9)
        Phi = np.vstack((U_past, Y_past, U_future))
        Yf = hankel(Y_train, self.T, N, start=self.rho)

        if self.causal:
            self.Theta = self._fit_theta_causal(Phi, Yf)
        else:
            self.Theta = self._fit_theta_ls(Phi, Yf)
        
        # Store the data, just in case
        self._Phi = Phi

        # Compute the Σ_Δ and Σ_Φ matrices
        self.Sigma_Delta = 1.0 / self.N * (Yf - self.Theta @ Phi) @ (Yf - self.Theta @ Phi).T
        self.Sigma_Phi = 1.0 / self.N * Phi @ Phi.T
    
    
    def predict(self,
                Up: np.ndarray,
                Yp: np.ndarray,
                Uf: np.ndarray | cp.Variable) -> np.ndarray:
        """ Use the predictive model for generating an (open-loop) prediction

        Args:
            Up (np.ndarray): Past input data (shape: (ρ, n_in))
            Yp (np.ndarray): Past output data (shape: (ρ, n_out))
            Uf (np.ndarray | cp.Variable): Future input data (shape: (T, n_in))

        Returns:
            np.ndarray: The predicted output data (shape: (T, n_out))
        """
        
        # Built the regression vector φ = [ U_past; Y_past; U_future ], see (7)
        U_past = Up[-self.rho:, :].reshape(-1, 1)
        Y_past = Yp[-self.rho:, :].reshape(-1, 1)

        if isinstance(Uf, np.ndarray):
            Phi = np.vstack((U_past, Y_past, Uf.reshape(-1, 1)))
        else:
            Uf_ = cp_flatten(Uf)
            Phi = cp.vstack((U_past, Y_past, Uf_))
            
        # Future output
        Y = self.Theta @ Phi
        return Y.reshape((-1, Yp.shape[1]))
    

    def _construct_problem(self, 
                           slack_in_stage_cost: bool = True, 
                           slack_in_constraint: bool = True, 
                           output_constraints: bool = False,
                           phi_range_constraint: bool = False,
                           phi_regularization: bool = False) -> None:
        """Construct the DPP optimization problem

        Args:
            slack_in_stage_cost (bool, optional): Whether to account for the Δy in the MPC stage cost. Defaults to True.
            slack_in_constraint (bool, optional): Whether to account for the Δy in the output constraints. Defaults to True.
            output_constraints (bool, optional): Whether output constraints need to be setup. Defaults to False.
            phi_range_constraint (bool, optional): Whether to enforce the range constraint on the φ vector. Defaults to False.
            phi_regularization (bool, optional): Whether to include the regularization on φ. Defaults to False. 
        """
        self._u = cp.Variable((self.T, self.n_in))
        self._u_past = cp.Parameter((self.rho, self.n_in))
        self._y_past = cp.Parameter((self.rho, self.n_out))
        self._r = cp.Parameter((self.n_out,))

        # Auxiliary variable to account for the range constraint on the slack variable
        # Δy = Σ_Δ * c ensures that the slack Δy lies in the column space of Σ_Δ
        self._c = cp.Variable((self.T, self.n_out))

        self._phi = cp.vstack((cp_flatten(self._u_past), 
                               cp_flatten(self._y_past),
                               cp_flatten(self._u),))


        self._u_lb = cp.Parameter((1, self.n_in))
        self._u_ub = cp.Parameter((1, self.n_in))
        self._y_lb = cp.Parameter((1, self.n_out))
        self._y_ub = cp.Parameter((1, self.n_out))

        # The stage cost matrices (per time-step)
        self._Q = cp.Parameter((self.n_out, self.n_out), PSD=True)        
        self._S = cp.Parameter((self.n_out, self.n_out), PSD=True)  # Terminal cost matrix
        self._R = cp.Parameter((self.n_in, self.n_in), PSD=True)

        # _Sigma_Delta is just a parameter that will be set to Σ_Δ
        self._Sigma_Delta = cp.Parameter((self.T * self.n_out, self.T * self.n_out), PSD=True)

        # If Σ_Φ is not invertible, we need to include the range constraint
        if phi_range_constraint:
            self._Sigma_Phi = cp.Parameter((self.rho * (self.n_in + self.n_out) + self.n_in * self.T, self.rho * (self.n_in + self.n_out) + self.n_in * self.T), PSD=True)
            self._cphi = cp.Variable((self.rho * (self.n_in + self.n_out) + self.n_in * self.T, 1))

        # Regularization weights
        self._lambda2 = cp.Parameter(nonneg=True)

        # Model's output (shape: (T, n_y)) is Θ*φ 
        self._ym = cp_unflatten(self.Theta @ self._phi, self.n_out)

        # Slacked ouput: y = Θ*φ + Δy (see (23))
        self._y = self._ym + self._Sigma_Delta @ cp_flatten(self._c)
        
        # Input and output constraints
        self._constraints = [ self._u >= self._u_lb,
                               self._u <= self._u_ub ]
        
        if output_constraints:
            # If the slack_in_constraint is True, we use the slacked output in the constraints (as in DeePC)
            # else we use the predicted output ΘΦ (as in SPC)
            y_cstr = self._y if slack_in_constraint else self._ym
            self._constraints += [ y_cstr >= self._y_lb,
                                   y_cstr <= self._y_ub ]
            
        # If we are supposed to enforce the range constraint on φ, we do it here
        if phi_range_constraint:
            self._constraints += [ self._Sigma_Phi @ self._cphi == self._phi ]
        
        # NOTE: We do not use the regularization on φ because it is not needed
        self._cost = self._lambda2 / self.N * cp.quad_form(cp_flatten(self._c), self._Sigma_Delta) 
        
        # Regularization on φ. We include this only if really necessary
        if phi_regularization:
            self._lambda1 = cp.Parameter(nonneg=True)

            if phi_range_constraint:
                # If we are enforcing the range constraint on φ, it is more efficient to use the cphi variable for the regularization
                self._cost += self._lambda1 / self.N * cp.quad_form(cp_flatten(self._cphi), self._Sigma_Phi) 
            else: 
                # But, if we are not enforcing the range constraint (e.g. because Σ_Φ is invertible), we use the full expression
                self._Sigma_Phi_dagger = cp.Parameter((self.rho * (self.n_in + self.n_out) + self.n_in * self.T, self.rho * (self.n_in + self.n_out) + self.n_in * self.T), PSD=True)
                self._cost += self._lambda1 / self.N * cp.quad_form(self._phi, self._Sigma_Phi_dagger)

        # Split Theta into T chunks of n_out rows at time
        for t in range(self.T):
            _yt = self._y[t, :] if slack_in_stage_cost else self._ym[t, :]

            # If the last time-step, use the terminal cost matrix S, else Q
            # As for the input, we use the R matrix
            self._cost += cp.quad_form(_yt - self._r, self._S if t == self.T - 1 else self._Q) \
                            + cp.quad_form(self._u[t, :], self._R)

        # TODO: We should probably formulate this as a DPP problem to speed up the computation!
        self._problem = cp.Problem(cp.Minimize(self._cost), self._constraints)
        
    @property
    def slack_regularization(self):
        return self._lambda2.value
    
    @property
    def _delta(self):
        """The slack variable Δy = Σ_Δ * c"""
        return cp_unflatten(self._Sigma_Delta @ cp_flatten(self._c), self.n_out)


    def mpc_setup(self,
                  u_lb = -np.inf, 
                  u_ub = np.inf, 
                  y_lb = -np.inf, 
                  y_ub = np.inf, 
                  Q: float | np.ndarray = 1.0,
                  R: float | np.ndarray = 0.1,
                  S: float | np.ndarray = 1.0,
                  lambda1: float = 0,
                  lambda2: float = None,
                  slack_in_stage_cost: bool = True, 
                  slack_in_constraint: bool = True) -> None:
        """Setup the MPC problem

        Args:
            u_lb (float, optional): Lower bound on the input. Defaults to -np.inf.
            u_ub (float, optional): Upper bound on the input. Defaults to np.inf.
            y_lb (float, optional): Lower bound on the output. Defaults to -np.inf.
            y_ub (float, optional): Upper bound on the output. Defaults to np.inf.
            q (float | np.ndarray, optional): Weight on the output error. If a scalar, it will be converted to a diagonal matrix. Defaults to 1.0.
            r (float | np.ndarray, optional): Weight on the input. If a scalar, it will be converted to a diagonal matrix. Defaults to 0.1.
            s (float | np.ndarray, optional): Weight on the terminal output error. If a scalar, it will be converted to a diagonal matrix. Defaults to 1.0.
            lambda1 (float, optional): Regularization weight on φ. Defaults to 0.
            lambda2 (float, optional): Regularization weight on the slack variable. Defaults to None.
            slack_in_stage_cost (bool, optional): Whether to account for the Δy in the MPC stage cost. Defaults to True (DeePC).
            slack_in_constraint (bool, optional): Whether to account for the Δy in the output constraints. Defaults to True (DeePC).
        """
        # Whether output constraints are really there
        out_constraint = y_lb != -np.inf or y_ub != np.inf
        # Whether the range constraint must be imposed on φ 
        phi_range = get_data_regime(self.U_train.shape[0], self.T, self.rho, self.n_in, self.n_out) < 0 or np.linalg.eigvalsh(self.Sigma_Phi)[-1] < 1e-12
        phi_regularization = lambda1 > 0

        self._construct_problem(slack_in_stage_cost=slack_in_stage_cost, 
                                slack_in_constraint=slack_in_constraint, 
                                output_constraints=out_constraint, 
                                phi_range_constraint=phi_range,
                                phi_regularization=phi_regularization)

        self._u_lb.value = format_constraint(u_lb, self.n_in, -np.inf)
        self._u_ub.value = format_constraint(u_ub, self.n_in, np.inf)
        self._y_lb.value = format_constraint(y_lb, self.n_out, -np.inf)
        self._y_ub.value = format_constraint(y_ub, self.n_out, np.inf)

        self._lambda2.value = lambda2 if lambda2 is not None else 1.0
        
        self._Q.value = Q * np.eye(self.n_out) if np.isscalar(Q) else Q
        self._R.value = R * np.eye(self.n_in) if np.isscalar(R) else R
        self._S.value = S * np.eye(self.n_out) if np.isscalar(S) else S

        # Set Σ_Δ to the computed value
        self._Sigma_Delta.value = self.Sigma_Delta

        if phi_regularization:
            self._lambda1.value = lambda1

        if phi_range:
            self._Sigma_Phi.value = self.Sigma_Phi
        if phi_regularization and not phi_range:
            self._Sigma_Phi_dagger.value = la.pinvh(self.Sigma_Phi)


    def mpc_step(self, 
                 reference: np.ndarray,
                 Up: np.ndarray,
                 Yp: np.ndarray,
                 slack_override: float = None) -> tuple[np.ndarray, np.ndarray]:
        """Perform one step of the MPC

        Args:
            reference: (np.ndarray): Reference signal (shape: (n_out, ))
            Up (np.ndarray): Past inputs (shape: (rho, n_in))
            Yp (np.ndarray): Past outputs (shape: (rho, n_out))
            slack_override (float, optional): Override the regularization weight. Defaults to None (use the one set in mpc_setup)

        Returns:
            np.ndarray: The generated optimal input sequence  (shape: (T, n_in))
            np.ndarray: The predicted output  (shape: (T, n_out))
        """
        self._r.value = reference.reshape((self.n_out, )) if isinstance(reference, np.ndarray) else np.array([reference] * self.n_out).reshape((self.n_out,))
        self._u_past.value = Up
        self._y_past.value = Yp

        if slack_override is not None:
            prev_reg = self._lambda2.value
            self._lambda2.value = slack_override
            self._problem.solve(solver=self.solver)
            self._lambda2.value = prev_reg
        else:
            self._problem.solve(solver=self.solver)

        if self._problem.status != 'optimal' and get_data_regime(self.U_train.shape[0], self.T, self.rho, self.n_in, self.n_out) < 0 :
            raise NotImplementedError('Problem is not feasible (N too small). Slack variables must be included in the range constraint on φ.')

        return self._u.value, self._y.value


class DeePC(AbstractDDPC):
    def __init__(self, 
                 U_train: np.ndarray,   # (shape: (N_bar, n_in))
                 Y_train: np.ndarray,   # (shape: (N_bar, n_out))
                 T: int = 50,
                 rho: int = 5,
                 N: int = -1,
                 solver: str = None,
                 name: str = 'DeePC') -> None:
        """Build up our DDPC using the DeePC algorithm

        Args:
            U_train (np.ndarray): Training input (shape: (N_bar, n_in))
            Y_train (np.ndarray): Training output (shape: (N_bar, n_out))
            T (int, optional): Prediction horizon. Defaults to 50.
            rho (int, optional): Length of the regression window. Defaults to 5.
            N (int, optional): Number of training samples. Defaults to -1 (use all the samples).
            solver (str, optional): The solver to be used. To let CVXPY choose the best solver, set it to None. Defaults to None.
        """
        super().__init__(U_train, Y_train, T, rho, name)
        self.solver = solver

        # Store the training data, as we need it.
        self.store_data(U_train, Y_train)


    def store_data(self, U: np.ndarray, Y: np.ndarray) -> None:
        """Store the training data

        Args:
            U (np.ndarray): Input data matrix of shape (N_bar, n_in)
            Y (np.ndarray): Output data matrix of shape (N_bar, n_out)
        """
        U_past = hankel(U, self.rho, self.N)
        Y_past = hankel(Y, self.rho, self.N)
        U_future = hankel(U, self.T, self.N, start=self.rho)

        self.Phi = np.vstack((U_past, Y_past, U_future))
        self.Yf = hankel(Y, self.T, self.N, start=self.rho)

        # The projection operator Π = Φ^†  Φ
        Pi_Phi = la.pinv(self.Phi) @ self.Phi
        # The complementary projection operator I - Φ^†  Φ
        self._Pi_ortho = np.eye(*Pi_Phi.shape) - Pi_Phi 


    def _construct_problem(self, reg_mode: str = 'proj', out_constraint: bool = True):
        """Construct the DeePC optimization problem
        """
        # We define u and y as optimizaton bariables, and then we constrain them (they are determined by g)
        self._g = cp.Variable((self.Phi.shape[1], 1))
        self._u = cp.Variable((self.T, self.n_in))
        self._y = cp.Variable((self.T, self.n_out))

        self._u_ctx = cp.Parameter((self.rho, self.n_in))
        self._y_ctx = cp.Parameter((self.rho, self.n_out))

        self._r = cp.Parameter((self.n_out,))
        self._Q = cp.Parameter((self.n_out, self.n_out), PSD=True)
        self._R = cp.Parameter((self.n_in, self.n_in), PSD=True)
        self._S = cp.Parameter((self.n_out, self.n_out), PSD=True)

        self._u_lb = cp.Parameter((1, self.n_in))
        self._u_ub = cp.Parameter((1, self.n_in))
        self._y_lb = cp.Parameter((1, self.n_out))
        self._y_ub = cp.Parameter((1, self.n_out))
        self._beta = cp.Parameter(nonneg=True)

        # Regularization, see (19) and (20) in the paper
        if reg_mode == 'proj':
            self._Pi = cp.Parameter((self.Phi.shape[1], self.Phi.shape[1]), PSD=True)
            self._cost = self._beta * cp.sum_squares(self._Pi @ self._g)
        else:
            self._cost = self._beta * cp.sum_squares(self._g)
            
        phi = cp.vstack((cp_flatten(self._u_ctx),
                         cp_flatten(self._y_ctx),
                         cp_flatten(self._u)))
        
        self._constraints = [ self.Phi @ self._g == phi,                # DeePC constraint (see (14) in the paper)
                              self.Yf @ self._g == cp_flatten(self._y),  # DeePC constraint (see (15) in the paper)
                              self._u >= self._u_lb,
                              self._u <= self._u_ub ]
        
        # To avoid numerical problems, we enforce output constraints only if they are necessary
        if out_constraint:
            self._constraints += [  self._y >= self._y_lb,
                                    self._y <= self._y_ub ] 

        for t in range(self.T):
            _yt = self._y[t, :]

            # Stage cost
            self._cost += cp.quad_form(_yt - self._r, self._S if t == self.T - 1 else self._Q) \
                            + cp.quad_form(self._u[t, :], self._R)
            
        # TODO: We should probably formulate this as a DPP problem to speed up the computation!
        self._problem = cp.Problem(cp.Minimize(self._cost), self._constraints)


    @staticmethod
    def format_constraint(x = None, n = None, default = None):
        if x is None:
            x = default

        if isinstance(x, np.ndarray):
            return np.nan_to_num(x)
        else:
            return np.nan_to_num(np.array([x] * n).reshape(1, -1))
        

    def mpc_setup(self,
                 u_lb = -np.inf, 
                 u_ub = np.inf, 
                 y_lb = -np.inf,
                 y_ub = np.inf,
                 Q: float | np.ndarray = 1.0,
                 R: float | np.ndarray = 0.1,
                 S: float | np.ndarray = 1.0,
                 beta: float = 10.0,
                 reg_mode: str = 'proj') -> None:
        """Setup the DeePC problem

        Args:
            u_lb (float, optional): Lower bound on the input. Defaults to -np.inf.
            u_ub (float, optional): Upper bound on the input. Defaults to np.inf.
            y_lb (float, optional): Lower bound on the output. Defaults to -np.inf.
            y_ub (float, optional): Upper bound on the output. Defaults to np.inf.
            Q (float | np.ndarray, optional): Weight on the output error. If a scalar, it will be converted to a diagonal matrix. Defaults to 1.0.
            R (float | np.ndarray, optional): Weight on the input. If a scalar, it will be converted to a diagonal matrix. Defaults to 0.1.
            S (float | np.ndarray, optional): Weight on the terminal output error. If a scalar, it will be converted to a diagonal matrix. Defaults to 1.0.
            beta (float, optional): Regularization weight. Defaults to 10.0.
            reg_mode (str, optional): Regularization mode ('proj' or 'l2'). Defaults to 'proj'.
        """
        out_constraint = y_lb != -np.inf or y_ub != np.inf

        self._construct_problem(reg_mode=reg_mode, out_constraint=out_constraint)

        self._u_lb.value = format_constraint(u_lb, self.n_in, -np.inf)
        self._u_ub.value = format_constraint(u_ub, self.n_in, np.inf)
        self._y_lb.value = format_constraint(y_lb, self.n_out, -np.inf)
        self._y_ub.value = format_constraint(y_ub, self.n_out, np.inf)
        self._Q.value = Q * np.eye(self.n_out) if np.isscalar(Q) else Q
        self._R.value = R * np.eye(self.n_in) if np.isscalar(R) else R
        self._S.value = S * np.eye(self.n_out) if np.isscalar(S) else S
        self._beta.value = beta

        if reg_mode == 'proj':
            self._Pi.value = self._Pi_ortho


    def mpc_step(self, 
                 reference: np.ndarray,
                 Up: np.ndarray,
                 Yp: np.ndarray, 
                 slack_override: float = None) -> np.ndarray:
        """Perform one step of the MPC

        Args:
            reference (np.ndarray): Reference signal (shape: (n_out, ))
            Up (np.ndarray): Past inputs (shape: (rho, n_in))
            Yp (np.ndarray): Past outputs (shape: (rho, n_out))
            slack_override (float, optional): Override the regularization weight. Defaults to None (use the one set in mpc_setup)

        Returns:
            np.ndarray: The generated optimal input sequence  (shape: (T, n_in))
            np.ndarray: The predicted output sequence  (shape: (T, n_out))
        """
        self._r.value = reference.reshape((self.n_out, )) if isinstance(reference, np.ndarray) else np.array([reference] * self.n_out).reshape((self.n_out,))
        self._u_ctx.value = Up
        self._y_ctx.value = Yp

        if slack_override is not None:
            # Make sure to restore the previous regularization after the solve
            prev_reg = self._beta.value
            self._beta.value = slack_override
            self._problem.solve(solver=self.solver)
            self._beta.value = prev_reg
        else:
            self._problem.solve(solver=self.solver)

        return self._u.value, self._y.value
    
    @property
    def slack_regularization(self):
        return self._beta.value
    

class Oracle(AbstractDDPC):
    def __init__(self, sys: control.ss, T: int = 50, solver: str = None, name: str = 'Oracle'):
        """Oracle MPC algorithm. This knows exactly the deterministic system dynamics and serves as a benchmark.

        Args:
            sys (control.ss): The discrete-time system to be used as model
            T (int, optional): The prediction horizon. Defaults to 50.
            solver (str, optional): The solver. If None, let CVXPY find the most suitable solver.
            name (str, optional): Name. Defaults to 'Oracle'.
        """
        self.A = sys.A
        self.B = sys.B
        self.C = sys.C
        self.D = sys.D
        self.T = T

        self.n_in = sys.ninputs
        self.n_out = sys.noutputs
        self.n_x = sys.A.shape[0]
        self.solver = solver
        self.name = name
    

    def _construct_problem(self, out_constraints: bool = False):
        """Construct the Oracle MPC optimization problem
        """
        self._u = cp.Variable((self.T, self.n_in))
        self._y = cp.Variable((self.T, self.n_out))
        self._x = cp.Variable((self.T + 1, self.n_x))

        self._r = cp.Parameter((self.n_out,))
        self._x0 = cp.Parameter((self.n_x,))

        self._u_lb = cp.Parameter((1, self.n_in))
        self._u_ub = cp.Parameter((1, self.n_in))
        self._y_lb = cp.Parameter((1, self.n_out))
        self._y_ub = cp.Parameter((1, self.n_out))

        self._Q = cp.Parameter((self.n_out, self.n_out), PSD=True)        
        self._S = cp.Parameter((self.n_out, self.n_out), PSD=True)
        self._R = cp.Parameter((self.n_in, self.n_in), PSD=True)

        self._constraints = [ self._u >= self._u_lb,
                              self._u <= self._u_ub,
                              self._x[0, :] == self._x0 ]
        
        if out_constraints:
            self._constraints += [ self._y >= self._y_lb,
                                   self._y <= self._y_ub ]
        
        self._cost = 0
        for t in range(self.T):
            # Predictive model constrains the dynamics
            self._constraints.append(self._y[t, :] == self.C @ self._x[t, :] + self.D @ self._u[t, :])
            self._constraints.append(self._x[t+1, :] == self.A @ self._x[t, :] + self.B @ self._u[t, :])

            self._cost += cp.quad_form(cp_flatten(self._y[t, :] - self._r), self._S if t == self.T - 1 else self._Q) \
                            + cp.quad_form(self._u[t, :], self._R)
           
        # TODO: We should probably formulate this as a DPP problem to speed up the computation!
        self._problem = cp.Problem(cp.Minimize(self._cost), self._constraints)


    def mpc_setup(self,
        u_lb = -np.inf, 
        u_ub = np.inf, 
        y_lb = -np.inf,
        y_ub = np.inf,
        Q: float | np.ndarray = 1.0,
        R: float | np.ndarray = 0.1,
        S: float | np.ndarray = 1.0) -> None:
        """Setup the MPC problem!

        Args:
            u_lb (float, optional): Lower bound on the input. Defaults to -np.inf.
            u_ub (float, optional): Upper bound on the input. Defaults to np.inf.
            y_lb (float, optional): Lower bound on the output. Defaults to -np.inf.
            y_ub (float, optional): Upper bound on the output. Defaults to np.inf.
            Q (float | np.ndarray, optional): Weight on the output error. If a scalar, it will be converted to a diagonal matrix. Defaults to 1.0.
            R (float | np.ndarray, optional): Weight on the input. If a scalar, it will be converted to a diagonal matrix. Defaults to 0.1.
            S (float | np.ndarray, optional): Weight on the terminal output error. If a scalar, it will be converted to a diagonal matrix. Defaults to 1.0.
        """
        out_constraint = y_lb != -np.inf or y_ub != np.inf
        self._construct_problem(out_constraints=out_constraint)

        self._u_lb.value = format_constraint(u_lb, self.n_in, -np.inf)
        self._u_ub.value = format_constraint(u_ub, self.n_in, np.inf)
        self._y_lb.value = format_constraint(y_lb, self.n_out, -np.inf)
        self._y_ub.value = format_constraint(y_ub, self.n_out, np.inf)
        
        self._Q.value = Q * np.eye(self.n_out) if np.isscalar(Q) else Q
        self._S.value = S * np.eye(self.n_out) if np.isscalar(S) else S
        self._R.value = R * np.eye(self.n_in) if np.isscalar(R) else R


    def mpc_step(self, 
                 reference: np.ndarray,
                 x0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Perform one step of the MPC

        Args:
            reference: (np.ndarray): Reference signal (shape: (n_out, ))
            x0 (np.ndarray): Current state of the system (shape: (n_x, ))

        Returns:
            np.ndarray: The generated optimal input sequence  (shape: (T, n_in))
            np.ndarray: The predicted output  (shape: (T, n_out))
        """
        self._r.value = reference.reshape((self.n_out, )) if isinstance(reference, np.ndarray) else np.array([reference] * self.n_out).reshape((self.n_out,))
        self._x0.value = x0

        self._problem.solve(solver=self.solver)

        if self._problem.status != 'optimal' and get_data_regime(self.U_train.shape[0], self.T, self.rho, self.n_in, self.n_out) < 0 :
            raise NotImplementedError('Problem is not feasible (N too small). Slack variables must be included in the range constraint on φ.')
        
        return self._u.value, self._y.value
    


def get_data_regime(N_bar, T, rho, n_in, n_out):
    """Return the data-regime where the DDPC operates

    Args:
        N_bar (int): Number of training samples
        T (int): Prediction horizon
        rho (int): Length of the regression window
        n_in (int): Number of inputs
        n_out (int): Number of outputs

    Returns:
        int: Data regime (-1: insufficient data, 0: underdetermined (overparametrization), 1: overdetermined (underparametrization))
    """
    N = N_bar - T - rho + 1

    if N >= (T + rho) * (n_in + n_out):
        return 1
    elif N < (T + rho) * n_in + rho * n_in:
        return -1
    else:
        return 0
    

def get_size_for_regime(regime, T, rho, n_in, n_out):
    """Get the size of the training data to operate in a certain regime

    Args:
        regime (int): Regime where we want to operate
        T (int): Prediction horizon
        rho (int): Length of the regression window
        n_in (int): Number of inputs
        n_out (int): Number of outputs

    Returns:
        tuple[int, int]: (min_size, max_size)
    """
    
    tresh_overd = np.int32(np.ceil((T + rho) * (n_in + n_out) + T + rho - 1))
    tresh_welld = np.int32(np.ceil((T + rho) * n_in + rho * n_out + T + rho))

    match regime:
        case 1:
            return (tresh_overd, np.inf)
        case 0:
            return (tresh_welld, tresh_overd-1)
        case -1:
            return (0, tresh_welld-1)

def format_constraint(x = None, n = None, default = None):
    """Just make sure that the constraints are in the correct format"""
    if x is None:
        x = default

    if isinstance(x, np.ndarray):
        return np.nan_to_num(x)
    else:
        return np.nan_to_num(np.array([x] * n).reshape(1, -1)) 