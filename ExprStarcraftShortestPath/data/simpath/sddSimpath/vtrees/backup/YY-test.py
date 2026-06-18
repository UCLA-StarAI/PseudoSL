import os
import glob

if __name__ == '__main__':

    for dtree_filename in glob.glob("cnfs/*.dtree"):
        vtree_filename = os.path.basename(dtree_filename)
        vtree_filename = os.path.splitext(vtree_filename)[0]
        vtree_filename = "vtrees/" + vtree_filename + ".vtree"
        
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
                print "warning: unexpected token"
                pass

        dtree.close()
        vtree.close()
