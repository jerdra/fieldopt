"""
Bayesian optimizer
"""

from collections import deque
import wrapt
import time

import numpy as np

from moe.optimal_learning.python.cpp_wrappers.domain import (
    TensorProductDomain as cTensorProductDomain)
from moe.optimal_learning.python.python_version.domain import (
    TensorProductDomain)
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.cpp_wrappers.expected_improvement import (
    ExpectedImprovement)
from moe.optimal_learning.python.cpp_wrappers.expected_improvement import (
    multistart_expected_improvement_optimization as meio)
from moe.optimal_learning.python.data_containers import (HistoricalData,
                                                         SamplePoint)
from moe.optimal_learning.python.cpp_wrappers.log_likelihood_mcmc import (
    GaussianProcessLogLikelihoodMCMC)
from moe.optimal_learning.python.default_priors import DefaultPrior
from moe.optimal_learning.python.base_prior import BasePrior
from moe.optimal_learning.python.cpp_wrappers.optimization import (
    GradientDescentOptimizer as cGDOpt, GradientDescentParameters as cGDParams)
from moe.optimal_learning.python.base_prior import TophatPrior, NormalPrior

from .base import IterableOptimizer

import logging

logger = logging.getLogger(__name__)

# Estimated from initial hyper-parameter optimization
DEFAULT_LENGTHSCALE_PRIOR = TophatPrior(-2, 5)
DEFAULT_CAMPL_PRIOR = NormalPrior(12.5, 1.6)
DEFAULT_PRIOR = DefaultPrior(n_dims=3 + 2, num_noise=1)
DEFAULT_PRIOR.tophat = DEFAULT_LENGTHSCALE_PRIOR
DEFAULT_PRIOR.ln_prior = DEFAULT_CAMPL_PRIOR

# Default SGD
DEFAULT_SGD_PARAMS = {
    "num_multistarts": 200,
    "max_num_steps": 50,
    "max_num_restarts": 5,
    "num_steps_averaged": 4,
    "gamma": 0.7,
    "pre_mult": 1.0,
    "max_relative_change": 0.5,
    "tolerance": 1.0e-10
}


@wrapt.decorator
def _check_initialized(wrapped, instance, args, kwargs):
    if instance.gp_loglikelihood is None:
        logging.error("Model has not been initialized! "
                      "Use '.step() or .initialize_model()' "
                      "to initialize optimizer")
    else:
        return wrapped(*args, **kwargs)


class BayesianMOEOptimizer(IterableOptimizer):
    """
    Initialize default parameters for running a Bayesian optimization
    algorithm on an objective function with parameters.

    Initializes a squared exponential covariance prior using a
    - tophat prior on the lengthscale
    - lognormal prior on the covariance amplitude
    """
    def __init__(self,
                 objective_func,
                 samples_per_iteration,
                 bounds,
                 minimum_samples=10,
                 prior=None,
                 sgd_params=DEFAULT_SGD_PARAMS,
                 maximize=False,
                 max_iterations=None,
                 min_iterations=None,
                 epsilon=1e-3):
        '''
        Arguments:
            objective_func (callable): Objective function
            samples_per_iteration (int): Number of samples to propose
                per iteration
            bounds (ndarray): (P,2) array where each row
                corresponds to the (min, max) of feature :math:`p`
            minimum_samples (int): Minimum number of samples to collect
                before performing convergence checks
            prior (BasePrior): Cornell MOE BasePrior subclass
                to use for lengthscale and covariance amplitude
            sgd_params (GradientDescentParameters): Stochastic
                Gradient Descent parameters (see: cornell MOE's
                GradientDescentParameters)
            maximize (bool): Maximize if true, else minimize
            max_iterations (int): Maximum number of iterations before
              stopping optimization
            min_iterations (int): Perform at least `min_iterations` before
              stopping optimizations
            epsilon (float): Standard deviation convergence threshold
        '''

        super(BayesianMOEOptimizer, self).__init__(objective_func, maximize)

        self.epsilon = epsilon

        self.dims = bounds.shape[0]
        self.best_point_history = []
        self.convergence_buffer = deque(maxlen=minimum_samples)

        if max_iterations < min_iterations:
            raise ValueError("Minimum number of iterations exceeds "
                             "maximum number of iterations!")

        self.max_iter = max_iterations
        self.min_iter = min_iterations

        # non-C++ wrapper needed since it has added functionality
        # at cost of speed
        moe_bounds = [ClosedInterval(mn, mx) for mn, mx in bounds]
        self.search_domain = TensorProductDomain(moe_bounds)
        self.c_search_domain = cTensorProductDomain(moe_bounds)

        # TODO: Noise modelling will be supported later
        if prior is None:
            logging.warning("Using default prior from Cornell MOE")
            logging.warning("Prior may be sub-optimal for problem " "domain!")
            self.prior = DefaultPrior(n_dims=self.dims + 2, num_noise=1)
        elif not isinstance(prior, BasePrior):
            raise ValueError("Prior must be of type BasePrior!")
        else:
            self.prior = prior

        self.sgd = cGDParams(**sgd_params)
        self.gp_loglikelihood = None

        self.num_samples = samples_per_iteration

    def get_history(self):
        '''
        Retrieve current optimization history

        Returns:
            history (ndarray): [N, (*inputs, value)] array, where
                N is the number of points that have been sampled
        '''
        history = []
        for c, v in self.best_point_history:
            history.append(np.array([*c, v]))

        return np.array(history)

    def __str__(self):
        return f'''
        Configuration:
        Samples/iteration: {self.num_samples}
        Minimum Samples: {self.convergence_buffer.maxlen}
        Epsilon: {self.epsilon}
        Bounds: {str(self.bounds)}
        Maximize: {self.maximize}

        State:
        Iteration: {self.iteration}
        Current Best: {self.current_best}
        Convergence: {self.convergence}
        '''

    @property
    def gp(self):
        '''
        Gaussian process model from the current ensemble

        Returns:
            gp (GaussianProcessLogLikelihoodMCMC): :math:`GP` model
        '''
        if self.gp_loglikelihood is None:
            logging.warning("Model has not been initialized "
                            "Use .initialize_model() or .step() to "
                            "initialize GP model")
            return
        return self.gp_loglikelihood.models[0]

    @property
    def current_best(self):
        '''
        Get current estimate of optimal value

        Returns:
            best_coord (ndarray): (P,) best input coordinate
            best_value (float): Objective function evaluated at `best_coord`
        '''
        if self.gp_loglikelihood is None:
            logging.warning("Model has not been initialized "
                            "Use .initialize_model() or .step() to "
                            "initialize GP model")
            return
        history = self.gp.get_historical_data_copy()
        best_value = np.min(history._points_sampled_value)
        best_index = np.argmin(history._points_sampled_value)
        best_coord = history.points_sampled[best_index]
        return best_coord, best_value

    @property
    def _buffer_filled(self):
        return len(self.convergence_buffer) == self.convergence_buffer.maxlen

    @property
    def converged(self):
        '''
        Evaluates whether Bayesian optimization has converged.
        Examines whether the standard deviation of the history of length
        `self.min_samples` is below `self.epsilon`

        Returns:
            converged (bool): False if stop criterion has not been met,
            else True
        '''
        # Minimum number of iterations have not been met
        if (self.max_iter is not None) and (self.iteration == self.max_iter):
            logging.info(f"Reached maximum number of iterations,"
                         " N={self.max_iter}")
            return True

        if not self._buffer_filled:
            return False

        if self.min_iter is not None and (self.iteration < self.min_iter):
            return False

        if self.gp_loglikelihood is None:
            return False

        criterion = self._compute_convergence_criterion()
        logging.debug(f"Buffer standard deviation: {criterion}")
        return criterion < self.epsilon

    @_check_initialized
    def _compute_convergence_criterion(self):
        '''
        Compute the convergence criterion (sample standard deviation)
        of the past self.min_samples evaluations
        '''
        _, best = self.current_best
        deviation = np.linalg.norm(np.array(self.convergence_buffer) - best)
        return deviation

    def initialize_model(self):
        '''
        Initialize the GaussianProcessLogLikelihood model using
        an initial set of observations

        Returns:
            init_pts (ndarray): (N,P) Initial sampling points
            res (ndarray): (N,) Objective function evaluations of `init_pts`
        '''

        self.iteration = 0
        logging.debug(f"Initializing model with {self.num_samples} samples")
        init_pts = self.search_domain\
            .generate_uniform_random_points_in_domain(self.num_samples)

        res = self.evaluate_objective(init_pts)
        logging.debug(f"Initial samples: {init_pts}")

        history = HistoricalData(dim=self.dims, num_derivatives=0)
        history.append_sample_points(
            [SamplePoint(i, o, 0.0) for i, o in zip(init_pts, res)])

        self.gp_loglikelihood = GaussianProcessLogLikelihoodMCMC(
            historical_data=history,
            derivatives=[],
            prior=self.prior,
            chain_length=1000,
            burnin_steps=2000,
            n_hypers=2**4,
            noisy=False)

        self.gp_loglikelihood.train()
        return init_pts, res

    @_check_initialized
    def _update_model(self, evidence):
        '''
        Updates the current ensemble of models with
        new data

        Arguments:
            evidence (SamplePoint): New SamplePoint data
        '''
        self.gp_loglikelihood.add_sampled_points(evidence)
        self.gp_loglikelihood.train()
        return

    @_check_initialized
    def _update_history(self):
        '''
        Update the history of best points with the
        current best point and value
        '''
        best_coord, best_value = self.current_best
        self.best_point_history.append((best_coord, best_value))

    @_check_initialized
    def propose_sampling_points(self):
        '''
        Performs stochastic gradient descent to optimize qEI function
        returning a list of optimal candidate points for the current
        set of ensemble models

        Returns:
            samples (ndarray): (N,P) Set of q-EI optimal samples to evaluate
            ei (float): q-Expected improvement
        '''
        samples, ei = _gen_sample_from_qei(self.gp, self.c_search_domain,
                                           self.sgd, self.num_samples)
        return samples, ei

    def step(self):
        '''
        Performs one iteration of Bayesian optimization:
        1. get sampling points via maximizing qEI
        2. Evaluate objective function at proposed sampling points
        3. Update ensemble of Gaussian process models
        4. Store current best in the history of best points
        5. Increment iteration counter

        Returns:
            sampling_points (ndarray): Points sampled
            res (ndarray): Objective function evaluations at `sampling_points`
            qEI (float): q-Expected Improvement of `sampling_points`
        '''
        if self.iteration == 0:
            sampling_points, res = self.initialize_model()
            qEI = None
            logging.debug("Model has not yet been built! "
                          "Initializing model..")
        else:
            sampling_points, qEI = self.propose_sampling_points()
            logging.debug(f"Sampling points: {str(sampling_points)}")
            logging.debug(f"q-Expected Improvement: {qEI}")
            res = self.evaluate_objective(sampling_points)
            evidence = [
                SamplePoint(c, v, 0.0) for c, v in zip(sampling_points, res)
            ]
            self._update_model(evidence)
        self._update_history()

        _, best = self.current_best
        self.convergence_buffer.append(best)
        self._increment()

        return sampling_points, res, qEI

    def iter(self, print_status=False):
        '''
        Generator for end-to-end optimization

        Yields:
            iter_result (dict): Iteration tracking information
        '''
        while not self.converged:

            start = time.time()
            sampling_points, res, qEI = self.step()
            best_point, best_val = self.current_best

            if self._buffer_filled:
                criterion = self._compute_convergence_criterion()
            else:
                criterion = None

            out = {
                "best_point": best_point,
                "best_value": best_val,
                "iteration": self.iteration,
                "samples": sampling_points,
                "result": res,
                "qei": qEI,
                "criterion": criterion,
                "converged": self.converged
            }

            if print_status:
                logging.info(f"Duration: {time.time() - start}")
                logging.info(f"Iteration: {self.iteration}")
                logging.info(f"Best Value: {self.sign * best_val}")
                logging.info(f"Criterion: {criterion}")
                logging.info(f"qEI: {qEI}")
                logging.info(
                    f"Converged" if self.converged else "Not Converged")
                logging.info("-----------------------------------------------")

            yield out


def get_default_tms_optimizer(f,
                              num_samples,
                              minimum_samples=10,
                              max_iterations=None,
                              min_iterations=None):
    '''
    Construct BayesianMOEOptimizer using pre-configured
    prior hyperparameters

    Arguments:
        f (fieldopt.objective.FieldFunc): FieldFunc objective function
        num_samples (int): Number of samples to evaluate in parallel
        minimum_samples (int): Minimum number of samples to evaluate before
            performing convergence checks
        max_iterations (int): Maximum number of iterations before cutting
            off optimization

    Returns:
        optimizer (fieldopt.optimization.bayes_moe.BayesianMOEOptimizer):
            Configured optimizer


    Note:
        Optimizer is initialized with a squared exponential
        covariance function with the following priors:
            - Length scale with a TopHat(-2, 5)
            - Log-normal covariance amplitude Ln(Normal(12.5, 1.6))
    '''
    # Set standard TMS bounds
    bounds = f.domain.bounds
    bounds[2, :] = np.array([0, 180])

    return BayesianMOEOptimizer(objective_func=f.evaluate,
                                samples_per_iteration=num_samples,
                                bounds=bounds,
                                minimum_samples=minimum_samples,
                                prior=DEFAULT_PRIOR,
                                maximize=True,
                                max_iterations=max_iterations,
                                min_iterations=min_iterations)


def _gen_sample_from_qei(gp,
                         search_domain,
                         sgd_params,
                         num_samples,
                         num_mc=1e4):
    '''
    Perform multistart stochastic gradient descent (MEIO)
    on the q-EI of a gaussian process model with
    prior models on its hyperparameters

    Arguments:
        gp              Gaussian process model
        search_domain   Input domain of gaussian process model
        sgd_params      Stochastic gradient descent parameters
        num_samples     Number of samples to maximize over
        num_mc          Number of monte carlo sampling iterations to
                        perform to compute integral

    Returns:
        points_to_sample    Optimal samples to evaluate
        qEI                 q-Expected Improvement of `points_to_sample`
    '''

    qEI = ExpectedImprovement(gaussian_process=gp,
                              num_mc_iterations=int(num_mc))

    # lhc_iter=2e4 doesn't actually matter since we're using SGD
    optimizer = cGDOpt(search_domain, qEI, sgd_params, int(2e4))
    points_to_sample = meio(optimizer,
                            None,
                            num_samples,
                            use_gpu=False,
                            which_gpu=0,
                            max_num_threads=8)
    qEI.set_current_point(points_to_sample[0])

    return points_to_sample, qEI.compute_expected_improvement()
