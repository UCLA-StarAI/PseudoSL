import cPickle as pickle

# ./c2d -in cnfs/sf-0-0.cnf -dt_method 4 -dt_out -in_memory
# ./c2d -in cnfs/sf-0-0.cnf -dt_method 0 -dt_out -in_memory

# for cnf in cnfs/*.cnf; do ./c2d -in $cnf -dt_method 4 -dt_out -in_memory; done

if __name__ == '__main__':
    with open("pickle/hier_universes.p","rb") as f:
        hier_universes = pickle.load(f)

    universes = hier_universes[0]
    for key in universes:
        universe = universes[key]
        if universe is None: continue
        print key, len(universe)

        variables = set()
        for edge in enumerate(universe):
            variables.add(edge[0])
            variables.add(edge[1])

        filename = "graphs/sf-%d-%d.txt" % key
        with open(filename,'w') as f:
            f.write("c ids of edges start at 1\n")
            f.write("c file syntax:\n")
            f.write("c graph number-of-nodes number-of-edges\n")
            f.write("c id-of-edge id-of-node id-of-node\n")
            f.write("c\n")

            f.write("graph %d %d\n" % (len(variables),len(universe)))
            for i,edge in enumerate(universe):
                index = i + 1
                x,y = edge
                f.write("%d %d %d\n" % (index,x,y))

        filename = "cnfs/sf-%d-%d.cnf" % key
        with open(filename,'w') as f:
            variables = set()
            for edge in enumerate(universe):
                variables.add(edge[0])
                variables.add(edge[1])

            f.write("p cnf %d %d\n" % (len(variables),len(universe)))
            for edge in universe:
                x,y = edge
                f.write("%d %d 0\n" % (x+1,y+1))

