-- Meta-graph constructed by removing edges

CREATE TABLE IF NOT EXISTS metagraph(
    meta_n INTEGER,
    e0     INTEGER,
    e1     INTEGER,
    UNIQUE (meta_n, e0, e1)
);

CREATE TABLE IF NOT EXISTS ref_invariant_integer(
    invariant_id INTEGER PRIMARY KEY,  
    function_name TEXT,
    computed TINYINY DEFAULT false,
    UNIQUE (function_name)
);

-- Invariant table for integers

CREATE TABLE IF NOT EXISTS invariant_integer(
    meta_n INTEGER,
    invariant_id INTEGER,
    value INTEGER
);

-- List of invariants

INSERT OR IGNORE INTO ref_invariant_integer (function_name) VALUES 

    ("automorphism_group_n"),

    ("chromatic_number"),

    ("n_vertex"),

    ("diameter"), 
    ("n_cycle_basis"),
    ("circumference"),
    ("girth"),

    ("n_edge"),
    ("n_endpoints"),

    ("is_k_regular"),
    ("is_strongly_regular"),

    ("radius"),
    ("is_eulerian"),
    ("is_distance_regular"),    
    ("is_planar"),
    ("is_bipartite"),

    ("n_articulation_points"),
    ("is_subgraph_free_K3"),
    ("is_subgraph_free_K4"),
    ("is_subgraph_free_K5"),
    ("is_subgraph_free_C4"),
    ("is_subgraph_free_C5"),
    ("is_subgraph_free_C6"),
    ("is_subgraph_free_C7"),
    ("is_subgraph_free_C8"),
    ("is_subgraph_free_C9"),
    ("is_subgraph_free_C10"),

    ("is_integral"),

    ("vertex_connectivity"),
    ("edge_connectivity"),

    ("is_tree"),
    ("is_chordal"),

    ("k_max_clique");
