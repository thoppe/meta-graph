import sqlite3, logging, argparse, os, collections, ast
import subprocess, itertools
import numpy as np
from src.helper_functions import load_graph_database, grab_vector
from src.invariants import convert_to_numpy, graph_tool_representation
import graph_tool
import pyparsing as pypar

desc   = "Verify the sequences produced are the correct ones"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('n',type=int,default=4,
                    help="graph size n to compute meta")
cargs = vars(parser.parse_args())
n = cargs["n"]
upper_idx = np.triu_indices(n)

# Start the logger
logging.root.setLevel(logging.INFO)

# Connect to the database
conn = load_graph_database(cargs["n"])


def convert_edge_to_adj(A):
    # The string representation of the upper triangular adj matrix
    au = ''.join(map(str,A[upper_idx]))
    # Convert the binary string to an int
    int_index = int(au,2)
    return int_index

# Make a mapping of all the graph id's
IDS = grab_vector(conn, '''SELECT graph_id FROM graph''')
IDS = dict(zip(IDS,range(len(IDS))))
Mn  = len(IDS)
current_mark = len(IDS)-1

from scipy.sparse import *
M = dok_matrix((Mn,Mn), dtype=int)

# Start with the complete graph
KN = np.ones((n,n),dtype=int) - np.diag((1,)*n)

# Find its adj representation
#au = ''.join(map(str,KN[upper_idx]))
#int_index = int(au,2)

g = KN.copy()

import graph_tool.topology
import graph_tool.draw

def iso_difference_in_set(g, graph_set):

    # Check if connected
    if len(graph_tool.topology.label_components(g)[1]) > 1:
        return False

    for h in graph_set:
        if graph_tool.topology.isomorphism(g,h):
            return False
    return True

def find_iso_set_from_cut(h):
    # For every edge in the current graph
    # find isomorphicly distinct set of edge removals

    iso_set = []

    for edge in h.edges():

        hx = h.copy()
        hx.remove_edge(edge)
        if iso_difference_in_set(hx, iso_set):
            iso_set.append(hx)
    return iso_set


def find_iso_match(g, canidate_list):

    print "CANIDATES!",canidate_list

    for idx,adj in canidate_list:
        h = graph_tool_representation(adj,**{"N":n})
        if graph_tool.topology.isomorphism(g,h):
            return idx
    
    raise ValueError("should find match already")


cmd_find = '''
SELECT graph_id,adj FROM graph WHERE special_degree_sequence=(?)'''


cmd_grab_list = '''SELECT graph_id,adj FROM graph'''
GRAPH_LIST = conn.execute(cmd_grab_list).fetchall()

for current_mark, target_adj in GRAPH_LIST[::-1]:

    g = graph_tool_representation(target_adj,**{"N":n})   
    iso_set = find_iso_set_from_cut(g)

    #graph_tool.draw.graphviz_draw(g,vcolor="blue")

    deg_set = [sorted([x.out_degree() for x in h.vertices()]) 
               for h in iso_set]

    for h in iso_set:

        #graph_tool.draw.graphviz_draw(h,vcolor="red")
        deg = sorted([x.out_degree() for x in h.vertices()])

        #print "degree seq", deg
        
        s = str(deg).replace(' ','')
        items = conn.execute(cmd_find,(s,)).fetchall()
        #print "checking match", deg
        match_mark = find_iso_match(h, items)

        i = IDS[current_mark]
        j = IDS[match_mark]
        #print "found link", i,j, g+h
        M[i,j] = 1

m = graph_tool.Graph(directed=False)
m.add_vertex(M.shape[0])
for edge in M.keys():
    m.add_edge(*edge)


f_png = "meta_simple_{}.png".format(n)
graph_tool.draw.graphviz_draw(m,layout="dot",
                              output=f_png,
                              vsize=.4,penwidth=4,size=(30,30))
print m





