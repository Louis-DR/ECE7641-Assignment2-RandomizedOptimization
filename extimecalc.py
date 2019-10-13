import time

time1 = time.time()

def chrono(n=0, prt=False):
    global time1
    rt = time.time() - time1
    if (prt) : print(" "*n+"Computation time : {0:.2f} seconds".format(rt))
    time1 = time.time()
    return rt