#!/usr/bin/env python

import glob
import os
import sdd

class Struct:
    """Create an instance with argument=value slots.
    This is for making a lightweight object whose class doesn't matter."""
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __cmp__(self, other):
        if isinstance(other, Struct):
            return cmp(self.__dict__, other.__dict__)
        else:
            return cmp(self.__dict__, other)

    def __repr__(self):
        args = ['%s=%s' % (k, repr(v)) for (k, v) in list(vars(self).items())]
        return 'Struct(%s)' % ', '.join(args)

def run(job):
    args,log_filename = job
    cmd = "exe/sddSimpath %s" % args
    os.system('echo %s >> %s' % (cmd,log_filename))
    os.system('/usr/bin/time %s >> %s 2>&1' % (cmd,log_filename))
    return None

#graph_filename = "new-graphs/agraph.txt"
#graph_filename = "new-graphs/sf-0.txt"
#filenames = [graph_filename]
filenames = glob.glob("new-graphs/*.txt")
filenames.sort()

for graph_filename in filenames:
    basename = os.path.basename(graph_filename)
    basename = os.path.splitext(basename)[0]
    print(basename)

    with open(graph_filename,'r') as f:
        for line in f.readlines():
            if line.startswith("graph"): break
        line = line.split()
        nodes,edges = int(line[1]),int(line[2])

    vtree_filename = "new-vtrees/%s.vtree" % basename
    sdd_filename = "new-sdds/%s.sdd" % basename
    #sdd_filename = "/media/mountaindew/new-sdds/%s.sdd" % basename
    output_sdd_filename = "new-sdds/%s-all-pairs.sdd" % basename
    log_filename = "new-sdds/%s.log" % basename
    vtree = sdd.sdd_vtree_read(vtree_filename)
    manager = sdd.sdd_manager_new(vtree)

    total = nodes*(nodes-1)/2
    count = 0

    print(("%d:" % total,))
    alpha = sdd.sdd_manager_false(manager)
    for i in range(nodes):
        for j in range(i+1,nodes):
            count += 1
            if count % 100 == 0:
                print(("%d" % count,))

            args = (graph_filename,vtree_filename,i+1,j+1,sdd_filename)
            args = "%s %s %d %d %s" % args
            job = args,log_filename
            run(job)
            beta = sdd.sdd_read(sdd_filename,manager)
            print(i+1,j+1,sdd.sdd_model_count(beta,manager)) # ACACAC
            alpha = sdd.sdd_disjoin(alpha,beta,manager)
            sdd.sdd_ref(alpha,manager)
            sdd.sdd_manager_garbage_collect(manager)
            sdd.sdd_deref(alpha,manager)
            os.remove(sdd_filename)

    print()
    print(("mc:   %d" % sdd.sdd_model_count(alpha,manager)))
    print(("nc:   %d" % sdd.sdd_count(alpha)))
    print(("size: %d" % sdd.sdd_size(alpha)))
    print("minimizing...")
    sdd.sdd_ref(alpha,manager)
    sdd.sdd_manager_minimize(manager)
    sdd.sdd_deref(alpha,manager)
    print(("nc:   %d" % sdd.sdd_count(alpha)))
    print(("size: %d" % sdd.sdd_size(alpha)))
    print("saving...")
    sdd.sdd_save(output_sdd_filename,alpha)
    sdd.sdd_vtree_save("new-sdds/%s-out.vtree" % basename, sdd.sdd_manager_vtree(manager))
    sdd.sdd_manager_free(manager)
