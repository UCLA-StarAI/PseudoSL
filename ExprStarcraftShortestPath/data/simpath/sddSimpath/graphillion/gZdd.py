from graphillion import GraphSet
import graphillion.tutorial as tl

fp = open('./output/unreducedGP.zdd', 'w')

gridSize = 6
i = gridSize-1
universe = tl.grid(i,i)
GraphSet.set_universe(universe)
start = 1
goal = (i+1)*(i+1)
gs = GraphSet.paths(start, goal)
gs.dump(fp)

