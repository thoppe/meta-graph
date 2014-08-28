import sqlite3
import logging
import argparse
import itertools

from src.helper_functions import load_graph_database, grab_vector
from src.helper_functions import grab_all, select_itr, grab_scalar

desc = "Creates the laplacian database of order N"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('N', type=int, default=4,
                    help="graph order")
parser.add_argument('--clear', default=False, action='store_true',
                    help="Clears this value from the database [unused]")
parser.add_argument('--force', default=False, action='store_true',
                    help="Clears and starts computation [unused]")
parser.add_argument('--test', default=False, action='store_true',
                    help="Does not save the computation. [unused]")

cargs = vars(parser.parse_args())
N = cargs["N"]

# Start the logger
logging.root.setLevel(logging.INFO)

logging.info("Starting laplacian build {}".format(N))

# Connect to the database
conn = load_graph_database(N)
sconn = load_graph_database(N, special=True)
f_lconn = "meta_db/lap_{}.db".format(N)
lconn = sqlite3.connect(f_lconn, check_same_thread=False)

# Template the lconn database
cmd_template = '''
CREATE TABLE IF NOT EXISTS ref_L(
    L_id INTEGER PRIMARY KEY AUTOINCREMENT,
    {rows},
    {unique_constraint}
);

CREATE TABLE IF NOT EXISTS laplacian(
    graph_id UNSIGNED INTEGER  PRIMARY KEY,
    L_id INTEGER,
    adj UNSIGNED BIG INT
);

CREATE TABLE IF NOT EXISTS computed(
    name STRING PRIMARY KEY
);
'''

col_names = ['coeff_{}'.format(k) for k in range(1,N+1)]
rows = ',\n'.join(['\t{} INTEGER'.format(name) for name in col_names])
unique_con = 'UNIQUE ({})'.format(','.join(col_names))
#print cmd_template.format(rows=rows,
#                          unique_constraint=unique_con)

cmd_template = cmd_template.format(rows=rows,
                                   unique_constraint=unique_con)

lconn.executescript(cmd_template)

# Determine the computed values
computed_names = grab_vector(lconn, "SELECT name FROM computed")

cmd_grab_adj = '''SELECT graph_id,adj FROM graph'''

cmd_single_grab = '''
SELECT x_degree,coeff FROM laplacian_polynomial
WHERE graph_id = (?) ORDER BY x_degree,coeff'''

if "ref_L" not in computed_names:
    logging.info("Inserting L values")
    cmd_insert = '''INSERT OR IGNORE INTO ref_L ({}) VALUES ({})'''
    qmarks = ['?']*len(col_names)
    cmd_insert = cmd_insert.format(','.join(col_names), ','.join(qmarks))

    def L_iter(source):
        for g_id, adj in source:
            L_result = grab_all(sconn, cmd_single_grab,(g_id,))
            L = zip(*L_result)[1]
            yield L

    # Build the Laplacian database
    source = select_itr(conn, cmd_grab_adj)
    lconn.executemany(cmd_insert, L_iter(source))

    lconn.execute('INSERT INTO computed VALUES ("ref_L")')
    cmd_index = '''CREATE INDEX IF NOT EXISTS idx_ref_L ON ref_L ({})'''
    lconn.execute(cmd_index.format(','.join(col_names)))

    lconn.commit()

def L_indexed_value(source):

    logging.info("Inserting graph match to L values")

    cmd_find = '''SELECT L_id FROM ref_L WHERE {cond}'''

    for g_id, adj in source:
        L_result = grab_all(sconn, cmd_single_grab,(g_id,))
        L = zip(*L_result)[1]
        cond = ' AND '.join(["{}={}".format(name,val) for name,val in 
                             zip(col_names, L)])
        L_id = grab_scalar(lconn, cmd_find.format(cond=cond))
        yield L_id,g_id,adj

if "laplacian" not in computed_names:

    cmd_insert = '''INSERT INTO laplacian 
    (L_id, graph_id, adj) VALUES (?,?,?)'''

    source = select_itr(conn, cmd_grab_adj)
    lconn.executemany(cmd_insert, L_indexed_value(source))

    cmd_index = '''CREATE INDEX IF NOT EXISTS 
                   idx_laplacian ON laplacian (adj)'''
    lconn.execute(cmd_index)

    lconn.execute('INSERT INTO computed VALUES ("laplacian")')
   

    lconn.commit()
