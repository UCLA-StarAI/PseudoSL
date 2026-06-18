from graphillion import GraphSet
import graphillion.tutorial as tl

for a in range(10):
	i = a+1
	universe = tl.grid(i,i)
	GraphSet.set_universe(universe)
	#tl.draw(universe)

	start = 1
	goal = (i+1)*(i+1)
	paths = GraphSet.paths(start, goal)
	print i, "x", i, "grid has", paths.len(), "simple paths."
