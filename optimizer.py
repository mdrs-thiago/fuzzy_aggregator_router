from optimizer_utils import plot_response
import numpy as np 
import matplotlib.pyplot as plt 
from geneticalgorithm import geneticalgorithm as ga


def run_experiment(num, den, evaluator, control_system, trials=1, popsize=100, epochs=100, crossover_prob = 0.8, mutation_prob = 0.1, T=15, plot=False):

    R_VALUES = [10, 11, 12, 13, 15, 16, 18, 20, 22, 24, 27, 30, 33, 36, 39, 43, 47, 51, 56, 62, 68, 75, 82, 91]
    C_VALUES = [10, 11, 12, 13, 15, 16, 18, 20, 22, 24, 27, 30, 33, 36, 39, 43, 47, 51, 56, 62, 68, 75, 82, 91]


    varbound=np.array([[0,23], [2,6]]*4 + [[0,23], [-12, -6]]*2)

    algorithm_param = {'max_num_iteration': epochs,\
                    'population_size':popsize,\
                    'mutation_probability':mutation_prob,\
                    'elit_ratio': 0.01,\
                    'crossover_probability': crossover_prob,\
                    'parents_portion': 0.4,\
                    'crossover_type':'uniform',\
                    'max_iteration_without_improv':None}

    results_1 = None 

    for i in range(trials):
        model_1=ga(function=evaluator,\
                    dimension=12,\
                    variable_type='int',\
                    variable_boundaries=varbound,\
                    convergence_curve = False, \
                    algorithm_parameters=algorithm_param)

        model_1.run();

        optimal_ctrl_system = control_system(num, den)

        #Get geneticalgorithm best solution
        v = model_1.output_dict['variable'].astype(int)
        R = np.array([R_VALUES[v[i]]*10**(v[i+1]) for i in range(0, 8, 2)], dtype=np.int64)
        C = np.array([float(C_VALUES[v[i]])*10**(float(v[i+1])) for i in range(8, 11, 2)], dtype=np.float64)
        
        Kp = R[3]*(R[0]*C[0] + R[1]*C[1])/(R[2]*R[0]*C[1])
        Kd = C[0]*R[3]*R[1]/R[2]
        Ki = R[3]/(C[1]*R[0]*R[2])
        print(R)
        print(C)
        print(f'Kp = {Kp}, Kd = {Kd}, Ki = {Ki}')
        a, b, c, time, response, system, _ = optimal_ctrl_system(v)
        res = {'Kp': Kp, 'Kd': Kd, 'Ki':Ki, 'ST':a, 'RT':b, 'OS':c, 'res':response, 't': time}
        if plot:
            try:
                plot_response(system, T=T)
            except:
                print('Could not run function plot response')

        if results_1 is None:
            results_1 = np.array(model_1.report) 
            metrics = [res]
        else:
            results_1 = np.vstack((results_1,np.array(model_1.report)))
            metrics.append(res)

    return results_1, metrics
