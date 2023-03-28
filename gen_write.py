import numpy as np
import struct
import os
from itertools import cycle


def open_with_mkdir(path, mode):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Made directory {directory}")
    return open(path, mode)


def generate_write_workload(
    dataset,
    data_dir,
    write_dir,
    init_size=100_000_000,
    num_workloads=40,
    workload_length=1_000_000,
    seed=14637928,  # R: seed + 1, RW: seed + 2, W-heavy: seed + 3, W-only: seed + 4
):
    print(f"Generating write workloads for {dataset} into {write_dir} (seed= {seed})")
    rng = np.random.default_rng(seed)

    # Read data blob.
    blob_path = os.path.join(data_dir, dataset)
    assert os.path.exists(blob_path)
    data = np.fromfile(blob_path, dtype=np.uint64)[1:]  # Ignore first number (array length)
    print(f"Loaded data of shape {data.shape}")

    # Write prefix initial data.
    init_data_idxs = rng.choice(np.arange(data.shape[0]), size=init_size, replace=False)
    init_data = np.sort(data[init_data_idxs])
    remaining_data = np.delete(data, init_data_idxs)
    init_data_path = os.path.join(write_dir, dataset)
    with open_with_mkdir(init_data_path, "wb") as f:
        f.write(struct.pack("Q", len(init_data)))
        init_data.tofile(f)
    print(f"Written initial data to {init_data_path}")

    # Generate read read-write, write-heavy, and write-only workloads. READ = r, WRITE = w.
    named_mode_cycles = [
        ("ronly", ['r'], workload_length, seed + 1),
        ("rw", ['r'] * 19 + ['w'], workload_length, seed + 1),
        ("wheavy", ['r', 'w'], workload_length, seed + 1),
        ("wonly", ['w'], workload_length, seed + 1),
        ("ronlymini", ['r'], workload_length // 100, seed + 1),
        ("rwmini", ['r'] * 19 + ['w'], workload_length // 100, seed + 1),
        ("wheavymini", ['r', 'w'], workload_length // 100, seed + 1),
        ("wonlymini", ['w'], workload_length // 100, seed + 1),
    ]
    for mode_name, mode_cycle, mode_num_queries, mode_seed in named_mode_cycles:
        rng = np.random.default_rng(mode_seed)
        for workload_idx in range(num_workloads):
            query_ts = np.array(
                [mode for _, mode in zip(range(mode_num_queries), cycle(mode_cycle))]
            )
            read_queries = rng.choice(init_data, size=mode_num_queries)
            write_queries = rng.choice(
                remaining_data,
                size=mode_num_queries,
                replace=False,
            )
            queries = np.where(query_ts == 'r', read_queries, write_queries)
            print(f"Samples of generated queries: {query_ts[:5]}, {queries[:5]}")

            # Compile into string before batch write.
            query_str = "\n".join([
                f"{query_t} {query}" for query_t, query in zip(cycle(mode_cycle), queries)
            ])

            # Write to file.
            query_path = os.path.join(write_dir, mode_name, f"{dataset}_ks_{workload_idx}")
            with open_with_mkdir(query_path, "w") as f:
                f.write(query_str)
                f.write("\n")
            print(f"Written {query_path}")

    
DATA_DIR = "data/"
WRITE_DIR = "datarw/"
# generate_write_workload("books_800M_uint64", DATA_DIR, WRITE_DIR)
# generate_write_workload("fb_200M_uint64", DATA_DIR, WRITE_DIR)
generate_write_workload("osm_cellids_800M_uint64", DATA_DIR, WRITE_DIR)
# generate_write_workload("wiki_ts_200M_uint64", DATA_DIR, WRITE_DIR)
# generate_write_workload("gmm_k100_800M_uint64", DATA_DIR, WRITE_DIR)
