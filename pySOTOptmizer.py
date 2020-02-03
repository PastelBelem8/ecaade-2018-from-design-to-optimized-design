from SimpleOptimizer import SimpleOptimizer
from pySOT import *
from poap.controller import SerialController, BasicWorkerThread
import numpy as np
from datetime import datetime


class PySOTOptimizer(SimpleOptimizer):
    SAMPLING_ALGORITHMS = {
        "LHS": LatinHypercube,
        "SLHS": SymmetricLatinHypercube,
        "2FACT": TwoFactorial,
        "BoxBehnken": BoxBehnken
    }
    CANDIDATE_SAMPLING_ALGORITHMS = {
        "DYCORS": CandidateDYCORS,
        "DDS": CandidateDDS,
        "SRBF": CandidateSRBF,
        "UNIFORM": CandidateUniform
    }
    SURROGATE_MODELS = {
        "RBF": RBFInterpolant,
        # "MARS": MARSInterpolant,
        "GP": GPRegression,
        # "PolyRegression": PolyRegression,
        "Ensemble": EnsembleSurrogate
    }

    def _prepare_model(self, model, alg_params):
        import inspect
        model_params = inspect.signature(model).parameters

        res = {}
        for param, val in enumerate(alg_params):
            if param in model_params:
                res[param] = val

        return res

    def _resolve_model(self, model, alg_options, max_eval):
        alg_options["maxp"] = max_eval
        alg_options = self._prepare_model(model, alg_options)

        return model(**alg_options)

    def __init__(self, ext_fitness_function, params_name, params_cont, params_int, bounds, results_folder, time_limit, eval_limit,
                 stop_val, minimize, n_runs, print_freq, plot_freq, plot_store, alg_name, alg_options,
                 initial_sampling_alg, n_initial_samples, sampling_candidate_alg, n_candidate_samples):

        super().__init__(ext_fitness_function, params_name, bounds, results_folder, time_limit, eval_limit,
                         stop_val, minimize, n_runs, print_freq, plot_freq, plot_store, alg_name)

        # <editor-fold desc="Problem Definition">
        def maximize(x):
            return -1 * self.fitness_function(x)
        fitness_function = self.fitness_function if minimize else maximize
        self.data = OptimizerProblem(fitness_function, self.lower_bounds, self.upper_bounds,
                                     self.n_params,
                                     np.array(params_int),  # Consider integer variables as well
                                     np.array(params_cont))

        # Check that the optimization problem follows pySOT standard
        check_opt_prob(self.data)
        # </editor-fold>

        # <editor-fold desc="Sampling algorithms definition">
        # Experimental design algorithm
        self.initial_sampling_alg = self.SAMPLING_ALGORITHMS[initial_sampling_alg](self.n_params, n_initial_samples)

        # Adaptive Sampling
        self.candidate_sampling_alg = \
            self.CANDIDATE_SAMPLING_ALGORITHMS[sampling_candidate_alg](self.data, n_candidate_samples)
        # </editor-fold>

        # <editor-fold desc="Surrogate Definition">
        assert self.eval_limit > 0, "Error Max number of evaluations should be positive"
        assert isinstance(alg_options, dict), "Error alg_options is not a dictionary"

        model = self.SURROGATE_MODELS[alg_name]
        self.surrogate = self._resolve_model(model, alg_options, eval_limit)
        # </editor-fold>

    def __str__(self):
        return "pySOTOptimizer: \n{}-dimensional problem \nLower bounds: {}\nUpper bounds: {}" \
               "\n Max #Evals: {}\n Max Time: {}".format(self.n_params, self.lower_bounds,
                                                         self.upper_bounds, self.eval_limit, self.time_limit)

    def run(self):
        print("\n[%s] Running %s  pySOTOptimizer algorithm" % (datetime.now().strftime('%H:%M:%S'), self.alg_name))

        for i in range(self.n_runs):
            print("\t[%s] Starting Run %s" % (datetime.now().strftime('%H:%M:%S'), i + 1))
            self.reset_run_counters(i)

            controller = SerialController(self.data.obj_function)
            strategy = SyncStrategyNoConstraints(
                worker_id=0, data=self.data, maxeval=self.eval_limit, nsamples=1,
                exp_design=self.initial_sampling_alg, response_surface=self.surrogate,
                sampling_method=self.candidate_sampling_alg)
            controller.strategy = strategy

            self.start_time = time.time()
            # Run the optimization strategy
            result = controller.run()
            self.final_time = time.time() - self.start_time

            self.update_results()
            self.print_run_results()

        self.avg_best_result /= self.n_runs
        self.avg_eval_count /= self.n_runs
        self.avg_time /= self.n_runs
        self.print_final_results()


class OptimizerProblem:
    def __init__(self, obj_function, xlow, xup, dim, integer, continuous):
        self.obj_function = obj_function
        self.xlow = xlow if isinstance(xlow, np.ndarray) else np.array(xlow)
        self.xup = xup if isinstance(xup, np.ndarray) else np.array(xup)
        self.dim = dim
        self.integer = integer
        self.continuous = continuous

    def objfunction(self, x):
        return self.obj_function(x)


def ackley_obj_function(x):
    n = float(len(x))
    return -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n)) - \
           np.exp(np.sum(np.cos(2.0 * np.pi * x)) / n) + 20 + np.exp(1)


def create_bounds(l, u, dim):
    res = []
    for _ in range(dim):
        res.append(-1)
        res.append(1)
    return res


def main():
    dim = 2
	
    optimizer = PySOTOptimizer(ext_fitness_function='C:\\Users\\CatarinaB\\Desktop\\analises\\2018_InesPereira\\PavPreto_0605.rkt', 
							params_name = ["raio", "n-aberturas"],
							params_cont = [0],
							params_int = [1],
                            bounds=[0.05, 2, 2, 7],
                            results_folder="results/ines-pavpreto-GP",
                            time_limit=-1,
                            eval_limit=65,
                            stop_val=math.inf,
                            minimize=False,
                            n_runs=2,
                            print_freq=1,
                            plot_freq=1,
                            plot_store=True,
                            alg_name="GP",
                            alg_options= {},#{"kernel": CubicKernel, "tail": ConstantTail},
                            initial_sampling_alg="SLHS",
                            n_initial_samples=2 * dim + 1,
                            sampling_candidate_alg="SRBF",
                            n_candidate_samples=100 * dim)
    optimizer.run()

if __name__ == '__main__':
    main()
