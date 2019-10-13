#%%
import numpy as np
from plotter import tickinit, tick, manyRuns, oneRun
import time
from extimecalc import chrono
import matplotlib.pyplot as plt
# plt.style.use('seaborn-ticks')
plt.style.use('seaborn-whitegrid')

import mlrose as mlr
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from scraper import import_wine_quality
from analysis import analyse

import winsound
frequency = 1000  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second



#%% #region[white] LOADING THE DATASET
print("----------Loading dataset----------")
XXX, yyy = import_wine_quality(subset_size=0)
X_train, X_test, y_train, y_test = train_test_split(XXX, yyy, test_size=0.15, random_state=0)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
fold = 4
print("----------Done loading dataset----------")
#endregion



#%% #region[black] NN OPTIMIZATION WITH SA

nn_model1 = mlr.NeuralNetwork(hidden_nodes = [100], activation = 'relu', 
                                 algorithm = 'simulated_annealing', 
                                 max_iters = 4000, bias = True, is_classifier = True, 
                                 learning_rate = 0.75, early_stopping = True, 
                                 clip_max = 5, max_attempts = 100, random_state = None,
                                 schedule=mlr.ExpDecay())

nn_model1.fit(X_train_scaled, y_train)

y_train_pred = nn_model1.predict(X_train_scaled)
y_train_accuracy = accuracy_score(y_train, y_train_pred)
print(y_train_accuracy)

y_test_pred = nn_model1.predict(X_test_scaled)
y_test_accuracy = accuracy_score(y_test, y_test_pred)
print(y_test_accuracy)

winsound.Beep(frequency, duration)

analyse(nn_model1, fold, XXX, yyy, X_train, y_train, X_test, y_test)

winsound.Beep(frequency, duration)
#endregion


#%% #region[black] NN OPTIMIZATION WITH GA

nn_model1 = mlr.NeuralNetwork(hidden_nodes = [100], activation = 'relu', 
                                 algorithm = 'genetic_alg', 
                                 max_iters = 4000, bias = True, is_classifier = True, 
                                 learning_rate = 0.75, early_stopping = True, 
                                 clip_max = 5, max_attempts = 100, random_state = 3,
                                 pop_size=200, mutation_prob=0.1)

nn_model1.fit(X_train_scaled, y_train)

y_train_pred = nn_model1.predict(X_train_scaled)
y_train_accuracy = accuracy_score(y_train, y_train_pred)
print(y_train_accuracy)

y_test_pred = nn_model1.predict(X_test_scaled)
y_test_accuracy = accuracy_score(y_test, y_test_pred)
print(y_test_accuracy)

winsound.Beep(frequency, duration)

analyse(nn_model1, fold, XXX, yyy, X_train, y_train, X_test, y_test)

winsound.Beep(frequency, duration)
#endregion


#%% #region[black] NN OPTIMIZATION WITH RHC

nn_model1 = mlr.NeuralNetwork(hidden_nodes = [100], activation = 'relu', 
                                 algorithm = 'random_hill_climb', 
                                 max_iters = 4000, bias = True, is_classifier = True, 
                                 learning_rate = 0.1, early_stopping = True, 
                                 clip_max = 5, max_attempts = 100, random_state = None,
                                 restarts=10)

nn_model1.fit(X_train_scaled, y_train)

y_train_pred = nn_model1.predict(X_train_scaled)
y_train_accuracy = accuracy_score(y_train, y_train_pred)
print(y_train_accuracy)

y_test_pred = nn_model1.predict(X_test_scaled)
y_test_accuracy = accuracy_score(y_test, y_test_pred)
print(y_test_accuracy)

winsound.Beep(frequency, duration)

analyse(nn_model1, fold, XXX, yyy, X_train, y_train, X_test, y_test)

winsound.Beep(frequency, duration)
#endregion



#%% #region[white] ONEMAX
def gen_problem_onemax(problem_size):
    fitness = mlr.OneMax()
    maximize = True
    problem = mlr.DiscreteOpt(length=problem_size, fitness_fn=fitness, maximize=maximize)
    return problem, maximize

gen_problem = gen_problem_onemax
problem, maximize = gen_problem(50)
#endregion



#%% #region[white] FOURPEAKS
def gen_problem_fourpeaks(problem_size):
    fitness = mlr.FourPeaks(t_pct=0.25)
    maximize = True
    problem = mlr.DiscreteOpt(length=problem_size, fitness_fn=fitness, maximize=maximize)
    return problem, maximize

gen_problem = gen_problem_fourpeaks
problem, maximize = gen_problem(5)
#endregion




#%% #region[white] PAIRIODIC
def pairiodic4(state):
    score = 0
    l = len(state)
    p = int(l/4)
    for k in range(p):
        x = state[k]
        nbr_rep = int(x==state[p+k]) + int(x==state[2*p+k]) + int(x==state[3*p+k])
        score += [1,2,0,5][nbr_rep]
    return score

def pairiodic6(state):
    score = 0
    l = len(state)
    p = int(l/6)
    for k in range(p):
        x = state[k]
        nbr_rep = int(x==state[p+k]) + int(x==state[2*p+k]) + int(x==state[3*p+k]) + int(x==state[4*p+k]) + int(x==state[5*p+k])
        score += [2,4,0,8,0,16][nbr_rep]
    return score

def pairiodic8(state):
    score = 0
    l = len(state)
    p = int(l/8)
    for k in range(p):
        x = state[k]
        nbr_rep = int(x==state[p+k]) + int(x==state[2*p+k]) + int(x==state[3*p+k]) + int(x==state[4*p+k]) + int(x==state[5*p+k]) + int(x==state[6*p+k]) + int(x==state[7*p+k])
        score += [2,4,0,8,0,16,0,32][nbr_rep]
    return score

def pairiodic10(state):
    score = 0
    l = len(state)
    p = int(l/10)
    for k in range(p):
        x = state[k]
        nbr_rep = int(x==state[p+k]) + int(x==state[2*p+k]) + int(x==state[3*p+k]) + int(x==state[4*p+k]) + int(x==state[5*p+k]) + int(x==state[6*p+k]) + int(x==state[7*p+k]) + int(x==state[8*p+k]) + int(x==state[9*p+k])
        score += [2,4,0,8,0,16,0,32,0,64][nbr_rep]
    return score

def gen_problem_pairiodic(problem_size):
    fitness = mlr.CustomFitness(fitness_fn=pairiodic6, problem_type='discrete')
    maximize = True
    problem = mlr.DiscreteOpt(length=problem_size, fitness_fn=fitness, maximize=maximize, max_val=2)
    return problem, maximize

gen_problem = gen_problem_pairiodic
problem, maximize = gen_problem(300)
#endregion



#%% #region[blue] RANDOMIZED HILL CLIMBING

print("► RANDOMIZED HILL CLIMBING")
hil = mlr.random_hill_climb
hil_params = {'max_attempts':50, 'max_iters':np.inf, 'restarts':100, 'init_state':None}
manyRuns(hil, problem, hil_params, color="SteelBlue", nbr_runs=25, maximize=maximize)
oneRun(hil,problem,hil_params,color="SteelBlue")
#endregion



#%% #region[red] SIMULATED ANNEALING

print("► SIMULATED ANNEALING")
ann = mlr.simulated_annealing
ann_params = {'schedule':mlr.ExpDecay(), 'max_attempts':50, 'max_iters':np.inf, 'init_state':None}
manyRuns(ann, problem, ann_params, color="IndianRed", nbr_runs=25, maximize=maximize)
oneRun(ann,problem,ann_params,color="IndianRed")
#endregion



#%% #region[green] GENETIC ALGORITHM

print("► GENETIC ALGORITHM")
gen = mlr.genetic_alg
gen_params = {'pop_size':200, 'mutation_prob':0.1, 'max_attempts':50, 'max_iters':10000}
manyRuns(gen, problem, gen_params, color="DarkSeaGreen", nbr_runs=25, maximize=maximize)
oneRun(gen,problem,gen_params,color="DarkSeaGreen")
#endregion



#%% #region[yellow] MIMIC
print("► MIMIC")
mim = mlr.mimic
mim_params = {'pop_size':200, 'keep_pct':0.3, 'max_attempts':50, 'max_iters':np.inf, 'fast_mimic':True}
manyRuns(mim, problem, mim_params, color="Goldenrod", nbr_runs=25, maximize=maximize)
oneRun(mim,problem,mim_params,color="Goldenrod")
#endregion



#%% #region[black] ALGORITHMS AND PARAMETERS TO RUN BEFORE THE NEXT CELLS                               #endregion
hil = mlr.random_hill_climb                                                                             #region[blue]
hil_params = {'max_attempts':50, 'max_iters':np.inf, 'restarts':50, 'init_state':None}                  #endregion
ann = mlr.simulated_annealing                                                                           #region[red]
ann_params = {'schedule':mlr.ExpDecay(), 'max_attempts':200, 'max_iters':np.inf, 'init_state':None}     #endregion
gen = mlr.genetic_alg                                                                                   #region[green]
gen_params = {'pop_size':200, 'mutation_prob':0.1, 'max_attempts':200, 'max_iters':np.inf}              #endregion
mim = mlr.mimic                                                                                         #region[yellow]
mim_params = {'pop_size':200, 'keep_pct':0.3, 'max_attempts':200, 'max_iters':np.inf, 'fast_mimic':True}#endregion
algos = [                                                                                               #region[black]                                                                                               
    {'key':"hil", 'algorithm':hil, 'params':hil_params, 'nbr_runs':10, 'color':"SteelBlue", 'name':"RANDOMIZED HILL CLIMBING"},
    {'key':"ann", 'algorithm':ann, 'params':ann_params, 'nbr_runs':10, 'color':"IndianRed", 'name':"SIMULATED ANNEALING"},
    {'key':"gen", 'algorithm':gen, 'params':gen_params, 'nbr_runs':10, 'color':"DarkSeaGreen", 'name':"GENETIC ALGORITHM"},
    {'key':"mim", 'algorithm':mim, 'params':mim_params, 'nbr_runs':10, 'color':"Goldenrod", 'name':"MIMIC"}
]
#endregion



#%% #region[black] FITNESS-ITERATIONS (NOT VERY INTERESTING)

problem_size = 10
problem, maximize = gen_problem(problem_size)

curvess = {algo['key']:[] for algo in algos}
best_curves = {algo['key']:[] for algo in algos}

for algo in algos:
    print(f"► {algo['name']}")
    nbr_runs = algo['nbr_runs']
    algorithm = algo['algorithm']
    params = algo['params']
    best_states = []
    best_fitnesses = []
    curves = []
    ticksize, tickmarks = tickinit(nbr_runs)
    chrono()
    for run_nbr in range(nbr_runs):
        best_state, best_fitness, fit_curve, time_curve = algorithm(problem, curve=True, random_state=run_nbr, **params)
        best_states.append(best_state)
        best_fitnesses.append(best_fitness)
        curves.append(fit_curve)
        tick(ticksize, tickmarks, run_nbr)
    print("│  {0:.2f}s\n".format(chrono()))
    best_fitness = np.amax(best_fitnesses) if maximize else np.amin(best_fitnesses)
    average_fitness = sum(best_fitnesses)/len(best_fitnesses)
    worst_fitness = np.amin(best_fitnesses) if maximize else np.amax(best_fitnesses)
    best_state = best_states[np.argmax(best_fitnesses)] if maximize else best_states[np.argmin(best_fitnesses)]
    best_curve = curves[np.argmax(best_fitnesses)] if maximize else curves[np.argmin(best_fitnesses)]
    curvess[algo['key']] = curves
    best_curves[algo['key']] = best_curve

for algo in algos:
    curves = curvess[algo['key']]
    best_curve = best_curves[algo['key']]
    nbr_runs = algo['nbr_runs']
    color = algo['color']

    max_len = max(map(len, curves))
    for curve in curves:
        while (len(curve)<max_len):
            curve += [curve[-1]]
    avg_curve = [sum(x)/len(curve) for x in zip(*curve)] 

    alpha = nbr_runs**-0.6
    for curve in curves:
        plt.plot(range(len(curve)),curve,color=color,alpha=alpha)
    plt.plot(range(len(avg_curve)),avg_curve,color=color,linewidth=2.0)
    plt.xlabel('Number of iterations')
    plt.ylabel('Fitness')
    plt.xscale('log')
plt.show()
#endregion



#%% #region[black] FITNESS-PROBLEM SIZE

problem_sizes = [3,10,35,100,350]

fitnesses_max = {algo['key']:[] for algo in algos}
fitnesses_avg = {algo['key']:[] for algo in algos}
fitnesses_min = {algo['key']:[] for algo in algos}

for problem_size in problem_sizes:
    print("╔═════════════════════╗")
    print("║ PROBLEM SIZE = {:<5}║".format(problem_size))
    print("╚═════════════════════╝")
    
    problem, maximize = gen_problem(problem_size)
    for algo in algos:
        print(f"► {algo['name']}")
        nbr_runs = algo['nbr_runs']
        algorithm = algo['algorithm']
        params = algo['params']
        best_states = []
        best_fitnesses = []
        ticksize, tickmarks = tickinit(nbr_runs)
        chrono()
        for run_nbr in range(nbr_runs):
            best_state, best_fitness, _, __, ___ = algorithm(problem, random_state=run_nbr, **params, curve=True)
            best_fitnesses.append(best_fitness)
            tick(ticksize, tickmarks, run_nbr)
        print("│  {0:.2f}s\n".format(chrono()))
        best_fitness = np.amin(best_fitnesses) if maximize else np.amax(best_fitnesses)
        average_fitness = sum(best_fitnesses)/len(best_fitnesses)
        worst_fitness = np.amax(best_fitnesses) if maximize else np.amin(best_fitnesses)
        fitnesses_max[algo['key']].append(best_fitness)
        fitnesses_avg[algo['key']].append(average_fitness)
        fitnesses_min[algo['key']].append(worst_fitness)

for algo in algos:
    plt.plot(problem_sizes, fitnesses_avg[algo['key']], algo['color'], linewidth=3.0)
    plt.fill_between(problem_sizes, fitnesses_min[algo['key']], fitnesses_max[algo['key']], color=algo['color'], alpha=0.15)
    plt.xlabel('Problem size')
    plt.xscale('log')
    plt.ylabel('Fitness')
plt.show()
#endregion



#%% #region[black] FITNESS-TIME

time_limit = 10
time_step = 0.01

problem_size = 100
problem, maximize = gen_problem(problem_size)

min_fit_curves = {algo['key']:[] for algo in algos}
avg_fit_curves = {algo['key']:[] for algo in algos}
max_fit_curves = {algo['key']:[] for algo in algos}
time_curvess = {algo['key']:[] for algo in algos}

for algo in algos:
    print(f"► {algo['name']}")
    nbr_runs = algo['nbr_runs']
    algorithm = algo['algorithm']
    params = algo['params']
    params['max_time'] = time_limit
    best_states = []
    best_fitnesses = []
    fit_curves = []
    time_curves = []
    ticksize, tickmarks = tickinit(nbr_runs)
    chrono()
    for run_nbr in range(nbr_runs):
        best_state, best_fitness, fit_curve, time_curve,_ = algorithm(problem, random_state=run_nbr, **params, curve=True)
        best_fitnesses.append(best_fitness)
        fit_curves.append(fit_curve)
        time_curves.append(time_curve)
        tick(ticksize, tickmarks, run_nbr)
    print("│  {0:.2f}s\n".format(chrono()))
    min_time = min([time_curve[0] for time_curve in time_curves])
    max_time = max([time_curve[-1] for time_curve in time_curves])
    time_interp = np.arange(min_time, max_time, time_step)
    fit_interps = []
    for fit_curve,time_curve in zip(fit_curves,time_curves):
        fit_interps.append(np.interp(time_interp, time_curve, fit_curve))
    time_curvess[algo['key']] = time_interp
    avg_fit_curves[algo['key']] = np.mean(fit_interps, axis=0)
    min_fit_curves[algo['key']] = np.amin(fit_interps, axis=0)
    max_fit_curves[algo['key']] = np.amax(fit_interps, axis=0)

for algo in algos:
    plt.plot(time_curvess[algo['key']], avg_fit_curves[algo['key']], algo['color'], linewidth=3.0)
    plt.fill_between(time_curvess[algo['key']], min_fit_curves[algo['key']], max_fit_curves[algo['key']], color=algo['color'], alpha=0.15)
    plt.xlabel('Maximum time')
    plt.xscale('log')
    plt.ylabel('Fitness')
plt.show()
#endregion



#%%
