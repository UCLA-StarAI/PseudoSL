from graphillion import GraphSet
import graphillion.tutorial as tl
import sys

#for a in range(10):
#	i = a+1
#	universe = tl.grid(i,i)
#	GraphSet.set_universe(universe)
#	#tl.draw(universe)
#
#	start = 1
#	goal = (i+1)*(i+1)
#	paths = GraphSet.paths(start, goal)
#	print i, "x", i, "grid has", paths.len(), "simple paths."

tokens = []

input = ""
for line in sys.stdin:
	if line[0] == 'c':
		continue
	for word in line.split():
		tokens.append(word)

if len(tokens) % 3 != 0:
	print "Invalid input, # of tokens should be divisible by 3"
	exit()

nodes = (int)(tokens[1])
edges = (int)(tokens[2])

edgeDict = {}
maxN = 0
for i in range(1,len(tokens)/3):
	edge_num = (int)(tokens[i*3+0])
	nodeA = (int)(tokens[i*3+1]) + 1
	nodeB = (int)(tokens[i*3+2]) + 1
	edgeDict[(nodeA,nodeB)] = True
	maxN = max(maxN,nodeA)
	maxN = max(maxN,nodeB)
	#print edge_num, nodeA, nodeB

edgeList = [e for e in edgeDict]
universe = edgeList
GraphSet.set_universe(universe)
gs = GraphSet([edgeList])
start = 1
goal = maxN
paths = GraphSet.paths(start,goal)
print "Counting simple paths from node", start, "to", goal
print "Count is:", paths.len()
