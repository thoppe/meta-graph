import sqlite3, logging, argparse, os, collections
import subprocess, itertools
import numpy as np
from src.helper_functions import load_graph_database, grab_vector, grab_scalar
from src.invariants import convert_to_numpy, graph_tool_representation

import graph_tool
import graph_tool.topology

desc   = "Create the simple edge meta-graph of order n"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('n',type=int,default=4,
                    help="graph order")

cargs = vars(parser.parse_args())
n = cargs["n"]
upper_idx = np.triu_indices(n)

# Start the logger
logging.root.setLevel(logging.INFO)

# Connect to the database
conn = load_graph_database(cargs["n"])

save_conn = sqlite3.connect("simple_meta.db")

# Connect to the database
with open("template_meta.sql") as FIN:
    script = FIN.read()
    save_conn.executescript(script)
    save_conn.commit()


def convert_edge_to_adj(A):
    # The string representation of the upper triangular adj matrix
    au = ''.join(map(str,A[upper_idx]))
    # Convert the binary string to an int
    int_index = int(au,2)
    return int_index

# Make a mapping of all the graph id's
IDS = grab_vector(conn, '''SELECT graph_id FROM graph''')
IDS = dict(zip(IDS,range(len(IDS))))



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

    for idx,adj in canidate_list:
        h = graph_tool_representation(adj,**{"N":n})
        if graph_tool.topology.isomorphism(g,h):
            return idx
    
    raise ValueError("should find match already")

def compute_meta_edges(current_mark, target_adj):

    g = graph_tool_representation(target_adj,**{"N":n})   
    iso_set = find_iso_set_from_cut(g)

    deg_set = [sorted([x.out_degree() for x in h.vertices()]) 
               for h in iso_set]

    new_edges = []

    for h in iso_set:

        #graph_tool.draw.graphviz_draw(h,vcolor="red")
        deg = sorted([x.out_degree() for x in h.vertices()])

        #print "degree seq", deg       
        s = str(deg).replace(' ','')
        items = conn.execute(cmd_find,(s,)).fetchall()
        match_mark = find_iso_match(h, items)

        i = IDS[current_mark]
        j = IDS[match_mark]

        #print "found link", i,j
        #M[i,j] = 1

        new_edges.append( (n,i,j) )

    return new_edges



cmd_find = '''
SELECT graph_id,adj FROM graph WHERE special_degree_sequence=(?)'''

cmd_grab_list = '''SELECT graph_id,adj FROM graph'''
GRAPH_LIST = conn.execute(cmd_grab_list).fetchall()

cmd_insert = '''INSERT INTO metagraph VALUES (?,?,?)'''

cmd_check = '''SELECT e0 FROM metagraph WHERE meta_n={} AND e0={} LIMIT 1'''

for current_mark, target_adj in GRAPH_LIST[::-1]:

    i = IDS[current_mark]
    check = save_conn.execute(cmd_check.format(n,i)).fetchone()

    print "Starting meta_{}, edge {}".format(n,i)

    if check == None:

        new_edges = compute_meta_edges(current_mark, target_adj)
        save_conn.executemany(cmd_insert, new_edges)

    if i and i%100==0 :
        save_conn.commit()
        print "Commiting changes"

save_conn.commit()


from scipy.sparse import *
import graph_tool.draw

#Mn  = len(IDS)
#print "N={}, total number of graphs={}".format(n, len(IDS))
#M = dok_matrix((Mn,Mn), dtype=int)

cmd_select = '''SELECT e0,e1 FROM metagraph WHERE meta_n={}'''

m = graph_tool.Graph(directed=False)
m.add_vertex(len(IDS))

cursor = save_conn.execute(cmd_select.format(n))
while cursor:
    block = cursor.fetchmany(1000)
    if not block: break
    for edge in block:
        m.add_edge(*edge)



f_png = "figures/meta_simple_{}.png".format(n)
logging.info("Saving %s"%f_png)

graph_tool.draw.graphviz_draw(m,layout="dot",
                              output=f_png,
                              vsize=.4,penwidth=4,size=(30,30))






