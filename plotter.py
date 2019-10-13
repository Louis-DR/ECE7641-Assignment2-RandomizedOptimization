import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('seaborn-ticks')
plt.style.use('seaborn-whitegrid')

#%%
def combine(fitnesses, times):
    full_fitnesses = [j for i in fitnesses for j in i]
    full_times = [j for i in times for j in i]
    full_combined = sorted(zip(full_times,full_fitnesses))
    sorted_fitnesses = [el[1] for el in full_combined]
    sorted_times = [el[0] for el in full_combined]

    return sorted_fitnesses, sorted_times
#%%

def tickinit(nbr_runs):
    ticksize=0
    tickmarks=[]
    if nbr_runs>=20:
        ticksize = 1
        tickmarks = [int(k*nbr_runs/20) for k in range(20)]
    elif nbr_runs>=10:
        ticksize = 2
        tickmarks = [int(k*nbr_runs/10) for k in range(10)]
    elif nbr_runs>=5:
        ticksize = 4
        tickmarks = [int(k*nbr_runs/5) for k in range(5)]
    else:
        ticksize = 20
        tickmarks = [nbr_runs-1]
    print("┌"+20*"─"+"┐")
    print("│",end="")
    return ticksize, tickmarks

def tick(ticksize, tickmarks, run_nbr):
    if (run_nbr in tickmarks) : print(ticksize*"▓",end="")

def oneRun(algorithm, problem, params, verbose=True, plot=True, random_state=None, color="gray"):
    best_state, best_fitness, fit_curve, time_curve, nbr_call = algorithm(problem, curve=plot, random_state=random_state, **params)
    if (verbose): print(f'The best state found is : {best_state}')
    if (verbose): print(f'The fitness at the best state is : {best_fitness}')
    if (verbose): print(f'Number of fitness calls : {nbr_call}')
    if (plot): 
        plt.plot(range(len(fit_curve)),fit_curve,color=color)
        plt.xlabel('Number of iterations')
        plt.ylabel('Fitness')
        plt.show()
        plt.plot(time_curve,fit_curve,color=color)
        plt.xlabel('Computation time (s)')
        plt.ylabel('Fitness')
        plt.show()
    return fit_curve, time_curve

def manyRuns(algorithm, problem, params, verbose=True, plot=True, hist=True, color="gray", nbr_runs=5, maximize=True):
    best_states = []
    best_fitnesses = []
    curves = []
    ticksize, tickmarks = tickinit(nbr_runs)
    for run_nbr in range(nbr_runs):
        best_state, best_fitness, fit_curve, _ = algorithm(problem, curve=plot, random_state=run_nbr, **params)
        best_states.append(best_state)
        best_fitnesses.append(best_fitness)
        curves.append(fit_curve)
        tick(ticksize, tickmarks, run_nbr)
    print("│")
    best_fitness = np.amax(best_fitnesses) if maximize else np.amin(best_fitnesses)
    average_fitness = sum(best_fitnesses)/len(best_fitnesses)
    worst_fitness = np.amin(best_fitnesses) if maximize else np.amax(best_fitnesses)
    best_state = best_states[np.argmax(best_fitnesses)] if maximize else best_states[np.argmin(best_fitnesses)]
    best_curve = curves[np.argmax(best_fitnesses)] if maximize else curves[np.argmin(best_fitnesses)]

    if (verbose): print(f'Number of different optimal states :  {len(set(tuple(best_state) for best_state in best_states))}')
    if (verbose): print(f'Best optimal fitness :                {best_fitness}')
    if (verbose): print(f'Average optimal fitness :             {average_fitness}')
    if (verbose): print(f'Worst optimal fitness :               {worst_fitness}')
    if (verbose): print(f'Best optimal state :\n  {best_state}')
    if (plot): 
        alpha = nbr_runs**-0.6
        for curve in curves:
            plt.plot(range(len(curve)),curve,color=color,alpha=alpha)
        plt.plot(range(len(best_curve)),best_curve,color=color,linewidth=2.0)
        plt.xlabel('Number of iterations')
        plt.ylabel('Fitness')
        plt.show()
    if (hist):
        plt.hist(best_fitnesses, bins=10, color=color)
        plt.xlabel('Fitness')
        plt.ylabel('Number of optima found')
        plt.show()
