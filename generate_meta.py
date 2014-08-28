import sqlite3
import logging
import argparse
import collections
import itertools
import multiprocessing

import graph_tool
import graph_tool.topology
import graph_tool.spectral

from src.helper_functions import load_graph_database, grab_vector
from src.helper_functions import grab_all, select_itr, grab_scalar
from src.invariants import graph_tool_representation
import src.invariants as invar

desc = "Creates the connected simple edge meta-graph of order N"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('N', type=int, default=4,
                    help="graph order")
parser.add_argument('--clear', default=False, action='store_true',
                    help="Clears this value from the database")
parser.add_argument('--force', default=False, action='store_true',
                    help="Clears and starts computation")
parser.add_argument('--test', default=False, action='store_true',
                    help="Does not save the computation.")
parser.add_argument('--procs', default= multiprocessing.cpu_count(), type=int,
                    help="Number of multiprocessers to use for each multitask")
parser.add_argument('--chunksize', default=50, type=int,
                    help="Chunks to feed to the multiprocessers")


cargs = vars(parser.parse_args())
N = cargs["N"]

# Start the logger
logging.root.setLevel(logging.INFO)

logging.info("Starting laplacian build {}".format(N))

# Connect to the database
conn = load_graph_database(N)
sconn = load_graph_database(N, special=True)

f_lconn = "meta_db/lap_{}.db".format(N)
lconn = sqlite3.connect(f_lconn)#, check_same_thread=False)

# Connect to the meta database and template it if needed
f_meta_conn = "meta_db/simple_connected_{}.db".format(N)
meta_conn = sqlite3.connect(f_meta_conn)

def is_connected(g):
    ''' Returns 1 if a graph-tool graph is connected '''
    component_size = graph_tool.topology.label_components(g)[1]
    return len(component_size) == 1


def iso_difference_in_set(g, iso_set):
    ''' Find an isomorphism in a list and return it, otherwise return True'''
    for h in iso_set:
        if graph_tool.topology.isomorphism(g, h):
            return False, h
    return True, None


def connected_cut_iterator(g):
    ''' Iterates over all graphs with one less edge that are still connected '''
    for edge in g.edges():

        gx = g.copy()
        gx.remove_edge(edge)

        # Check if connected
        if is_connected(gx):
            yield gx


def find_iso_set_from_cut(g):
    ''' For every edge in the current graph,
        find isomorphicly distinct set of edge removals
        and their cardinality '''

    iso_set = collections.Counter()

    for h in connected_cut_iterator(g):

        is_unique, h_iso = iso_difference_in_set(h, iso_set)
        if is_unique:
            iso_set[h] += 1
        else:
            iso_set[h_iso] += 1

    return iso_set


def identify_match_from_adj(g, match_set):
    '''
    Given a graph g, and a set of isomorphically distinct graphs
    in match_set (saved as list of two tuples, idx+adj), find
    the match in the set. Assume that the match is possible and check one less.
    '''

    for idx, adj in match_set[:-1]:
        h = graph_tool_representation(adj, N=N)
        if graph_tool.topology.isomorphism(g, h):
            return idx

    return match_set[-1][0]
    # raise ValueError("should find match already")


search_col_names = ['coeff_{}'.format(k) for k in range(1,N+1)]
search_cond = ' AND '.join(["{}=?".format(name) for name in search_col_names])
cmd_search_laplacian = '''
SELECT graph_id,adj FROM laplacian WHERE L_id IN 
(SELECT L_id FROM ref_L WHERE {})'''.format(search_cond)

def compute_meta_edge(item, **kwargs):
    '''
    Build a graph from adj and determine the unique graphs that can
    be made by removing a single edge. For each isomorphically distinct
    graph, compute the Laplacian. Use the Laplacian to quickly identify the connected
    meta edges and compute the forward and backwards weights.'''

    e0, target_adj = item

    # Determine the valid iso_set and compute the invariant (Laplacian)
    g = graph_tool_representation(target_adj, N=N)
    iso_set = find_iso_set_from_cut(g)

    E1_weights = {}
    E0_weights = collections.defaultdict(int)
    lconn = LCONN_CONNECTIONS[kwargs["_internal_id"]]

    for h in iso_set:
        A = graph_tool.spectral.adjacency(h)
        L = invar.special_laplacian_polynomial(A, N=N)
        degree,coeff = zip(*L)    
        match_set = grab_all(lconn, cmd_search_laplacian, coeff)

        # Figure out the forward weights
        e1 = identify_match_from_adj(h, match_set)
        E1_weights[e1] = iso_set[h]

        # Now determine the reverse weight
        for v1, v2 in itertools.combinations(h.vertices(), 2):
            h2 = h.copy()
            h2.add_edge(v1, v2)
            E0_weights[e1] += graph_tool.topology.isomorphism(g, h2)

    return (e0, E1_weights, E0_weights)

def record_edge_set(item,**kw):
    ''' Save the weights into the meta_conn database. '''

    e0, E1_weights, E0_weights = item

    cmd_insert = '''
    INSERT INTO metagraph
    (meta_n,e0,e1,weight,direction)
    VALUES (?,?,?,?,?)
    '''

    def edge_insert_itr():
        for e1 in E1_weights:
            yield (N, e0, e1, E1_weights[e1], 0)
        for e1 in E0_weights:
            yield (N, e1, e0, E0_weights[e1], 1)

    logging.info("Computed meta_n {} e0 ({})".format(N,e0))
    meta_conn.executemany(cmd_insert, edge_insert_itr())

# ######################################################################

# Build the source generator, tuple of (graph_id, adj)
cmd_grab = '''SELECT graph_id,adj FROM graph'''
source = select_itr(conn, cmd_grab)

# Build a list of database connections
LCONN_CONNECTIONS = [sqlite3.connect(f_lconn) for _ in range(cargs["procs"])]

# Build the multichain process early while nothing is allocated
from multi_chain import multi_Manager
MULTI_TASKS  = [compute_meta_edge,]
SERIAL_TASKS = [record_edge_set,]
M = multi_Manager(source, MULTI_TASKS, SERIAL_TASKS,
                  chunksize=cargs["chunksize"],
                  procs=cargs["procs"])

with open("templates/meta.sql") as FIN:
    script = FIN.read()
    meta_conn.executescript(script)
    meta_conn.commit()

# Process the clear and force command line arguments
if cargs["clear"] or cargs["force"]:
    cmd_clear = '''DELETE FROM metagraph WHERE meta_n = (?)'''
    logging.warning("Clearing metagraph values {}".format(N))
    meta_conn.execute(cmd_clear, (N,))
    meta_conn.execute("VACUUM")

    cmd_clear_complete = '''DELETE FROM computed WHERE meta_n = (?)'''
    meta_conn.execute(cmd_clear_complete, (N,))

    meta_conn.commit()

    if not cargs["force"]:
        M.shutdown()
        exit()

# Check if values have been computed, if so, exit early
cmd_check_complete = '''SELECT meta_n FROM computed'''
complete_n = grab_vector(meta_conn, cmd_check_complete)
if N in complete_n:
    msg = "meta {} has already been computed, exiting".format(N)
    print(msg)
    M.shutdown()
    exit()

# Copy the graph_id, adj information from conn -> meta_conn
logging.info("Populating the adj database")
cmd_select = '''SELECT graph_id,adj FROM graph'''
input_information = select_itr(conn, cmd_select)
cmd_copy_over = '''INSERT OR IGNORE INTO 
graph (n,graph_id, adj) VALUES ({},?,?)'''

cmd_copy_over = cmd_copy_over.format(N)
meta_conn.executemany(cmd_copy_over, input_information)

logging.info("Building the adj index")
cmd_create_idx = '''
CREATE INDEX IF NOT EXISTS idx_grap ON graph(n);'''
meta_conn.execute(cmd_create_idx)

logging.info("Started building the meta graph")
M.run()

if cargs["test"]:
    exit()

cmd_create_idx = '''
CREATE INDEX IF NOT EXISTS idx_metagraph ON metagraph(meta_n);'''
meta_conn.execute(cmd_create_idx)

cmd_mark_complete = "INSERT INTO computed VALUES (?)"
meta_conn.execute(cmd_mark_complete, (N,))
meta_conn.commit()

