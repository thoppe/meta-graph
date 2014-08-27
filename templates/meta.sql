-- Meta-graph 

CREATE TABLE IF NOT EXISTS metagraph(
    meta_n INTEGER,
    e0     INTEGER,
    e1     INTEGER,
    weight INTEGER,
    direction TINYINT,
    UNIQUE (meta_n, e0, e1, weight)
);

CREATE TABLE IF NOT EXISTS computed(
    meta_n INTEGER DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS graph(
    n UNSIGNED INTEGER,
    graph_id   INTEGER,
    adj    UNSIGNED BIG INT NOT NULL,
    UNIQUE (n,graph_id)
);
