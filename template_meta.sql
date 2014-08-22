-- Meta-graph constructed by removing edges

CREATE TABLE IF NOT EXISTS metagraph(
    meta_n INTEGER,
    e0     INTEGER,
    e1     INTEGER,
    UNIQUE (meta_n, e0, e1)
);

CREATE INDEX IF NOT EXISTS idx_metagraph ON metagraph(meta_n, e0);

CREATE TABLE IF NOT EXISTS computed(
    meta_n INTEGER DEFAULT FALSE
);
