all:
	python generate_meta.py 4

view:
	sqlitebrowser simple_meta.db

max_n = 10
possible_N_values = $(shell seq 1 ${max_n})

build:
	$(foreach n,$(possible_N_values),python generate_meta.py $(n) --chunksize 1000;)

max_n_draw = 8
possible_N_draw = $(shell seq 1 ${max_n_draw})

draw:
	$(foreach n,$(possible_N_draw),python draw_meta.py $(n) --draw;)

full_clean:
	rm -vf simple_meta.db
	rm -vf reps/*.gml figures/meta_simple_*.png

