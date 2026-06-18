#!/usr/bin/env python

import os
import sys
from subprocess import call

# ./c2d -in cnfs/sf-0-0.cnf -dt_method 4 -dt_out -in_memory
# ./c2d -in cnfs/sf-0-0.cnf -dt_method 0 -dt_out -in_memory

# for cnf in cnfs/*.cnf; do ./c2d -in $cnf -dt_method 4 -dt_out -in_memory; done

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage: %s [GRAPH_FILENAME]" % sys.argv[0])
        exit(1)

    graph_filename = sys.argv[1]
    if not graph_filename.endswith(".txt") and \
       not graph_filename.endswith(".graph"):
        print("please use .txt or .graph extension")
        exit(1)

    basename = os.path.basename(graph_filename)
    basename = os.path.splitext(basename)[0]
    cnf_filename = "tmp/" + basename + ".cnf"
    dtree_filename = cnf_filename + ".dtree"
    vtree_filename = basename + ".vtree"
    vtree_hg_filename = basename + ".vtree_hg"

    # read graph file
    with open(graph_filename,'r') as f:
        edges = dict()
        for line in f.readlines():
            line = line.strip()
            if line.startswith('c'): continue
            if line == "": continue
            if line.startswith('graph'):
                node_count,edge_count = [ int(token) for token in line.split()[1:] ]
            else:
                index,x,y = [ int(token) for token in line.split() ]
                edges[index] = (x,y)

    # write cnf file
    with open(cnf_filename,'w') as f:
        f.write("p cnf %d %d\n" % (node_count,edge_count))
        for edge in sorted(edges.keys()):
            x,y = edges[edge]
            f.write("%d %d 0\n" % (x+1,y+1))

    # generate dtree (min-fill)
    args = ["bin/c2d","-in",cnf_filename,"-dt_method","4","-dt_out","-in_memory"]
    call(args)

    # dtree to vtree
    dtree = open(dtree_filename,'r')
    vtree = open(vtree_filename,'w')

    index = 0

    for line in dtree.readlines():
        line = line.strip()
        if line == "": continue
        line = line.split(' ')
        if line[0] == 'dtree':
            node_count = int(line[1])
            #var_count = (node_count+1)/2
            vtree.write("vtree %d\n" % node_count)
        elif line[0] == 'L':
            var_id = int(line[1]) + 1
            vtree.write("L %d %d\n" % (index,var_id))
            index += 1
        elif line[0] == 'I':
            left,right = int(line[1]),int(line[2])
            vtree.write("I %d %d %d\n" % (index,left,right))
            index += 1
        else:
            print("warning: unexpected token")
            pass

    dtree.close()
    vtree.close()

    # generate dtree (hypergraph)
    args = ["bin/c2d","-in",cnf_filename,"-dt_method","0","-dt_out","-in_memory"]
    call(args)

    # dtree to vtree
    dtree = open(dtree_filename,'r')
    vtree = open(vtree_hg_filename,'w')

    index = 0

    for line in dtree.readlines():
        line = line.strip()
        if line == "": continue
        line = line.split(' ')
        if line[0] == 'dtree':
            node_count = int(line[1])
            #var_count = (node_count+1)/2
            vtree.write("vtree %d\n" % node_count)
        elif line[0] == 'L':
            var_id = int(line[1]) + 1
            vtree.write("L %d %d\n" % (index,var_id))
            index += 1
        elif line[0] == 'I':
            left,right = int(line[1]),int(line[2])
            vtree.write("I %d %d %d\n" % (index,left,right))
            index += 1
        else:
            print("warning: unexpected token")
            pass

    dtree.close()
    vtree.close()

