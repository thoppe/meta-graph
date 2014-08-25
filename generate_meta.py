import sqlite3, logging, argparse, os, collections
import subprocess, itertools
import numpy as np

import graph_tool
import graph_tool.topology
import graph_tool.spectral

from src.helper_functions import load_graph_database, grab_vector, grab_all, select_itr
from src.invariants import graph_tool_representation
import src.invariants as invar

desc   = "Creates the connected simple edge meta-graph of order N"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('N',type=int,default=4,
                    help="graph order")
parser.add_argument('--draw', default=False, action='store_true')
parser.add_argument('--clear', default=False, action='store_true',
                    help="Clears this value from the database")
parser.add_argument('--force', default=False, action='store_true',
                    help="Clears and starts computation")


def is_connected(g):
    component_size = graph_tool.topology.label_components(g)[1]
    return len(component_size)==1

def iso_difference_in_set(g, iso_set):
    ''' Find an isomorphism in a list and return it, otherwise return True'''
    for h in iso_set:
        if graph_tool.topology.isomorphism(g,h):
            return False,h
    return True,None

def connected_cut_iterator(g):
    ''' Iterates over all graphs with one less edge that are still connected '''
    for edge in g.edges():

        gx = g.copy()
        gx.remove_edge(edge)

        # Check if connected
        if is_connected(gx):
            yield gx
    

def find_iso_set_from_cut(g):
    # For every edge in the current graph,
    # find isomorphicly distinct set of edge removals and their cardinality

    iso_set = collections.Counter()

    for h in connected_cut_iterator(g):
       
        is_unique, h_iso = iso_difference_in_set(h, iso_set)
        if is_unique: 
            iso_set[h] += 1
        else: 
            iso_set[h_iso] += 1

    return iso_set

def possible_laplacian_match(L):
    return [ (k,ADJ[k]) for k in  LPOLY[L]]

def identify_match_from_adj(g, match_set):
    # Sanity check here, remove comments for speed
    #if len(match_set) == 1:
    #    return match_set[0][0]

    for idx,adj in match_set:
        h = graph_tool_representation(adj,N=N)
        if graph_tool.topology.isomorphism(g,h):
            return idx

    raise ValueError("should find match already")

def compute_valid_cuts( item ):
    e0, target_adj = item

    # Determine the valid iso_set and compute the invariant (Laplacian)
    g = graph_tool_representation(target_adj,N=N)
    iso_set = find_iso_set_from_cut(g)

    #laplacian_set = collections.Counter()
    laplacian_map = dict()

    HL = []
    for h in iso_set:
        A = graph_tool.spectral.adjacency(h)
        laplacian_map[h] = invar.special_laplacian_polynomial(A,N=N)

    return e0, iso_set, laplacian_map

def process_lap_poly(item):
    e0, iso_set,L_MAP = item

    L_GRAPH_MAP = {}
    for h,L in L_MAP.items():
        match_set = possible_laplacian_match(L)
        L_GRAPH_MAP[h] = match_set
    return (e0, iso_set, L_GRAPH_MAP)

def process_match_set(item):
    e0,iso_set,L_GRAPH_MAP = item

    E1_weights = {}
    for h,match_set in L_GRAPH_MAP.items():
        e1 = identify_match_from_adj(h,match_set)
        E1_weights[e1] = iso_set[h] 
        
    return (e0,E1_weights)

def record_E1_set(item):

    e0,E1_weights = item

    cmd_insert = "INSERT INTO metagraph VALUES (?,?,?,?)"

    def edge_insert_itr():
        for e1 in E1_weights:
            yield (N,e0,e1,E1_weights[e1])

    logging.info("Computed e0 ({})".format(e0))
    meta_conn.executemany(cmd_insert, edge_insert_itr())

cargs = vars(parser.parse_args())
N = cargs["N"]

# Start the logger
logging.root.setLevel(logging.INFO)

# Connect to the database
conn  = load_graph_database(N)
sconn = load_graph_database(N,special=True)

# Build the source generator, tuple of (graph_id, adj)
cmd_grab = '''SELECT graph_id,adj FROM graph'''
source = select_itr(conn,cmd_grab)

# Build the multichain process early while nothing is allocated
from multi_chain import multi_Manager
MULTI_TASKS  = [compute_valid_cuts,process_match_set]
SERIAL_TASKS = [process_lap_poly,record_E1_set]
M = multi_Manager(source, MULTI_TASKS, SERIAL_TASKS, procs=1,chunksize=1)

# Connect to the meta database and template it if needed
meta_conn = sqlite3.connect("simple_meta.db")
with open("template_meta.sql") as FIN:
    script = FIN.read()
    meta_conn.executescript(script)
    meta_conn.commit()

# Process the clear and force command line arguments
if cargs["clear"] or cargs["force"]:
    cmd_clear = '''DELETE FROM metagraph WHERE meta_n = (?)'''
    logging.warning("Clearing metagraph values {}".format(N))
    meta_conn.execute(cmd_clear, (N,))
    meta_conn.commit()

    meta_conn.execute("VACUUM")
    meta_conn.commit()

    cmd_clear_complete = '''DELETE FROM computed WHERE meta_n = (?)'''
    meta_conn.execute(cmd_clear_complete, (N,))

    if not cargs["force"]:
        exit()

# Check if values have been computed, if so, exit early
cmd_check_complete = '''SELECT meta_n FROM computed'''
complete_n = grab_vector(meta_conn,cmd_check_complete)
if N in complete_n:
    msg = "meta {} has already been computed, exiting".format(N)
    raise ValueError(msg)
    
# Make a mapping of all the graph id's
logging.info("Loading the graph adj information")
ADJ = dict(grab_all(conn, '''SELECT graph_id,adj FROM graph'''))

# Grab all the Laplacian polynomials
logging.info("Loading the Laplacian database")
single_grab = '''
SELECT x_degree,coeff FROM laplacian_polynomial
WHERE graph_id = (?) ORDER BY x_degree,coeff'''
LPOLY = collections.defaultdict(list)

for gid in ADJ:
    L = tuple(grab_all(sconn, single_grab, (gid,)))   
    LPOLY[L].append(gid)

num_LPOLY = len(LPOLY)
num_ADJ   = len(ADJ)
if not num_LPOLY:
    msg = "LPOLY database is empty"
    raise ValueError(msg)

if not num_ADJ:
    msg = "ADJ database is empty"
    raise ValueError(msg)


logging.info("Starting edge remove computation")
M.run()

cmd_mark_complete = "INSERT INTO computed VALUES (?)"
meta_conn.execute(cmd_mark_complete,(N,))
meta_conn.commit()

# Sanity check on the number of rows
cmd_rows = "SELECT e0 FROM metagraph WHERE meta_n={}".format(N)
E0_rows = set( grab_vector(meta_conn,cmd_rows) )
print "Missing rows: ", set(ADJ.keys()).difference(E0_rows)

exit()
'''

'''
f_png = "figures/meta_simple_{}.png".format(N)
logging.info("Saving %s"%f_png)

print ("Saving")
f_save = "reps/meta_{}.gml".format(N)
m.save(f_save)

print ("Drawing")
graph_tool.draw.graphviz_draw(m,layout="dot",
                              output=f_png,
                              vsize=.4,penwidth=4,size=(30,30))

#import pylab as plt
#plt.show()
