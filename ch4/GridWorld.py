import sys
import pandas as pd
import numpy as np

ACTION_MOVE_TABLE = [(0,1), (0,-1), (-1,0), (1,0)]
THETA = 0.001
DIMENSIONS = [4, 4]

def print_grid(a):
    print(a.view().reshape(DIMENSIONS))

def readGrid(filename):
    grid = None
    with open(filename, "r") as f:
        DIMENSIONS = list(map(int, f.readline().split(',')))
        #print(DIMENSIONS)
        grid = np.empty(shape=(DIMENSIONS[0]*DIMENSIONS[1]), dtype=object)
        df = pd.read_csv(f, header=None, index_col=False)
        for index, row in df.iterrows():
            #print(index/DIMENSIONS[0], index%DIMENSIONS[0])
            grid[index] = {"is_dest": row[0], "paths":(row[1:3].tolist(), row[3:5].tolist(), row[5:7].tolist(), row[7:].tolist())}
    return grid

def initPolicy(shape):
    P = np.empty(shape=grid.shape, dtype=object)
    for index, p in np.ndenumerate(P):
        P[index] = np.array([0.25]*4)
    return P
    
def PolicyEvaluation(grid, V, P, discount):
    Vnext = np.zeros(shape=V.shape)
    n_iteration = 0
    while True:
        delta = 0.0
        n_iteration += 1
        for index, x in np.ndenumerate(grid):
            Vnext[index] = 0.0
            if grid[index]["is_dest"] == 1:
                continue
            for a, p in np.ndenumerate(P[index]):
                path = grid[index]["paths"][a[0]]
                Vnext[index] += p*(path[1]+discount*V[path[0]])
            delta = max(delta, abs(V[index]-Vnext[index]))
        #print_grid(Vnext)
        np.copyto(V, Vnext)
        if delta < THETA:
            break
    return n_iteration   
    
def PolicyImprovement(grid, V, P):
    delta = 0.0
    for index, x in np.ndenumerate(grid):
        v = np.array( list(map(lambda z: V[grid[index]["paths"][z][0]], range(4))) )
        #print(v)
        winners = np.argwhere(v == np.amax(v)).flatten().tolist()
        p = 1.0/len(winners)
        pnew = np.array([0.0]*4)
        for i in winners:
            pnew[i] = p
        delta = max(delta, np.linalg.norm(P[index]-pnew))
        P[index] = pnew
    return delta
    
def PolicyIteration(grid, V, P):
    n_iteration = 0
    while True:
        n_iteration += 1
        print("@iteration ", n_iteration)
        n_eva = PolicyEvaluation(grid, V, P, 0.9)
        print_grid(V)
        print(n_eva)
        policy_change = PolicyImprovement(grid, V, P)
        print_grid(P)
        print(policy_change)
        if policy_change == 0.0:
            break
            
def ValueIteration(grid, V, P, discount):
    Vnext = np.zeros(shape=V.shape)
    n_iteration = 0
    while True:
        delta = 0.0
        n_iteration += 1
        print("@iteration ", n_iteration)
        for index, x in np.ndenumerate(grid):
            Vnext[index] = 0.0
            if grid[index]["is_dest"] == 1:
                continue
            v = np.array( list(map(lambda z: V[grid[index]["paths"][z][0]], range(4))) )
            r = np.array( list(map(lambda z: grid[index]["paths"][z][1], range(4))) )
            v = r+discount*v
            winners = np.argwhere(v == np.amax(v)).flatten().tolist()
            p = 1.0/len(winners)
            pnew = np.array([0.0]*4)
            for i in winners:
                pnew[i] = p
            P[index] = pnew
            Vnext[index] = np.inner(P[index], v)
            delta = max(delta, abs(V[index]-Vnext[index]))
        #print_grid(Vnext)
        np.copyto(V, Vnext)
        print_grid(V)
        print_grid(P)
        print(delta)
        if delta < THETA:
            break
    return n_iteration
    
def ValueIterationInplace(grid, V, P, discount):
    n_iteration = 0
    while True:
        delta = 0.0
        n_iteration += 1
        print("@iteration ", n_iteration)
        for index, x in np.ndenumerate(grid):
            if grid[index]["is_dest"] == 1:
                continue
            v = np.array( list(map(lambda z: V[grid[index]["paths"][z][0]], range(4))) )
            r = np.array( list(map(lambda z: grid[index]["paths"][z][1], range(4))) )
            v = r+discount*v
            winners = np.argwhere(v == np.amax(v)).flatten().tolist()
            p = 1.0/len(winners)
            pnew = np.array([0.0]*4)
            for i in winners:
                pnew[i] = p
            P[index] = pnew
            Vnext = np.inner(P[index], v)
            delta = max(delta, abs(V[index]-Vnext))
            V[index] = Vnext
        print_grid(V)
        print_grid(P)
        print(delta)
        if delta < THETA:
            break
    return n_iteration   
    
def TestPolicyIteration(grid):
    print(" --- PolicyIteration init ---")
    V = np.zeros(shape=grid.shape)
    print_grid(V)
    P = initPolicy(grid.shape)
    print_grid(P)
    print(" --- PolicyIteration start ---")
    PolicyIteration(grid, V, P)
    
def TestValueIteration(grid):
    print(" --- ValueIteration init ---")
    V = np.zeros(shape=grid.shape)
    print_grid(V)
    P = initPolicy(grid.shape)
    print_grid(P)
    print(" --- ValueIteration start ---")
    ValueIteration(grid, V, P, 0.9)
    
def TestValueIterationInplace(grid):
    print(" --- Inplace ValueIteration init ---")
    V = np.zeros(shape=grid.shape)
    print_grid(V)
    P = initPolicy(grid.shape)
    print_grid(P)
    print(" --- Inplace ValueIteration start ---")
    ValueIterationInplace(grid, V, P, 0.9)
    
if __name__ == "__main__":
    path= sys.argv[1] #normal set
    grid = readGrid(path)
    print("grid:")
    print_grid(grid)
    TestPolicyIteration(grid)
    TestValueIteration(grid)
    TestValueIterationInplace(grid)
    