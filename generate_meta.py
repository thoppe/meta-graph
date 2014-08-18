import sqlite3, logging, argparse, os, collections
import subprocess, itertools
import numpy as np
from src.helper_functions import load_graph_database, grab_vector, grab_scalar, grab_all
from src.invariants import convert_to_numpy, graph_tool_representation,compress_input
import src.invariants as invar

import graph_tool
import graph_tool.topology
import graph_tool.spectral

desc   = "Create the simple edge meta-graph of order n"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('N',type=int,default=4,
                    help="graph order")
parser.add_argument('--draw', default=False, action='store_true')

cargs = vars(parser.parse_args())
N = cargs["N"]
upper_idx = np.triu_indices(N)

# Start the logger
logging.root.setLevel(logging.INFO)

# Connect to the database
conn  = load_graph_database(N)
sconn = load_graph_database(N,special=True)

save_conn = sqlite3.connect("simple_meta.db")

# Connect to the database
with open("template_meta.sql") as FIN:
    script = FIN.read()
    save_conn.executescript(script)
    save_conn.commit()

__upper_matrix_index = np.triu_indices(N)

def convert_numpy_to_adj(A):
    # The string representation of the upper triangular adj matrix
    au = ''.join(map(str, A[__upper_matrix_index]))

    # Convert the binary string to an int
    int_index = int(au, 2)

    return int_index

# Make a mapping of all the graph id's
logging.info("Grabbing the graph adj information")
ADJ = dict(grab_all(conn, '''SELECT graph_id,adj FROM graph'''))

# Grab all the Laplacian polynomials
logging.info("Generating the Laplacian database")
single_grab = '''
SELECT x_degree,coeff FROM laplacian_polynomial
WHERE graph_id = (?) ORDER BY x_degree,coeff'''
LPOLY = collections.defaultdict(list)

'''
for gid,adj in ADJ.items():
    g  = graph_tool_representation(adj,N=N)
    gA = graph_tool.spectral.adjacency(g).toarray().astype(int)
    gadj = convert_numpy_to_adj(gA)
    invar.viz_graph(g)
exit()
'''

for gid in ADJ:
    L = tuple(grab_all(sconn, single_grab, (gid,)))   
    LPOLY[L].append(gid)

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
    #invar.viz_graph(h)  

    for edge in h.edges():

        hx = h.copy()
        hx.remove_edge(edge)
        if iso_difference_in_set(hx, iso_set):
            iso_set.append(hx)
            #invar.viz_graph(hx)

    return iso_set


def find_iso_match(g, canidate_list):

    for h_id,h_adj in canidate_list:
        h = graph_tool_representation(h_adj,N=N)
        if graph_tool.topology.isomorphism(g,h):
            return h_id
    
    raise ValueError("should find match already")


def compute_meta_edges(i,target_adj):

    g = graph_tool_representation(target_adj,N=N)
    iso_set = find_iso_set_from_cut(g)
   
    new_edges = []
    isomorphism_check_counter = []

    for h in iso_set:

        hA = graph_tool.spectral.adjacency(h).toarray().astype(int)
        h_adj = convert_numpy_to_adj(hA)
        hL = invar.special_laplacian_polynomial(h_adj,N=N)
        items = [(k,ADJ[k]) for k in LPOLY[hL]]

        isomorphism_check_counter.append(len(items))
        j = find_iso_match(h, items)

        new_edges.append( (N,i,j) )

    #print " + number of isomorphism checks to complete", sum(isomorphism_check_counter)
    return new_edges


cmd_insert = '''INSERT INTO metagraph VALUES (?,?,?)'''
cmd_check  = '''SELECT e0 FROM metagraph WHERE meta_n={} AND e0={} LIMIT 1'''
compute_size = len(ADJ)

cmd_check_complete = '''SELECT meta_n FROM computed'''
complete_n = grab_vector(save_conn,cmd_check_complete)

def process_adj((i,target_adj)):
    #check = save_conn.execute(cmd_check.format(N,i)).fetchone()
    #new_edges = []
    #if check == None:
    if i%1000==0:
        print "Starting meta_{}, edge {}".format(N,i)
    new_edges = compute_meta_edges(i,target_adj)
    return new_edges


import multiprocessing
P = multiprocessing.Pool()

if N not in complete_n:

    sol = P.imap(process_adj,ADJ.items())
    for k,new_edges in enumerate(sol):
        save_conn.executemany(cmd_insert, new_edges)

        #if k and k%100==0 :
        #    save_conn.commit()
        #    logging.info("Commiting changes for edge number {}".format(k))

P.close()
P.join()
    
save_conn.commit()
cmd_mark_complete = '''INSERT INTO computed VALUES (?)'''
save_conn.execute(cmd_mark_complete, (N,))
save_conn.commit()

if not cargs["draw"]:
    exit()


from scipy.sparse import *
import graph_tool.draw

print "Building graph-tool representation"

cmd_select = '''SELECT e0,e1 FROM metagraph WHERE meta_n={}'''

m = graph_tool.Graph(directed=False)
m.add_vertex(len(ADJ))

cursor = save_conn.execute(cmd_select.format(N))
while cursor:
    block = cursor.fetchmany(1000)
    if not block: break
    for edge in block:
        i,j = edge
        m.add_edge(i-1,j-1)

f_png = "figures/meta_simple_{}.png".format(N)
logging.info("Saving %s"%f_png)

print "Saving"
f_save = "reps/meta_{}.gml".format(N)
m.save(f_save)

print "Drawing"
graph_tool.draw.graphviz_draw(m,layout="dot",
                              output=f_png,
                              vsize=.4,penwidth=4,size=(30,30))

#import pylab as plt
#plt.show()




