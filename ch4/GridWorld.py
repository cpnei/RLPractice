import pandas as pd

ACTION_MOVE_TABLE = [(0,1), (0,-1), (-1,0), (1,0)]

def readGrid(filename):
    grid = 
    df = pd.read_csv(filename, header=None, index_col=False )
    for index, row in df.iterrows():
    return grid