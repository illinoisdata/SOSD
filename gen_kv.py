import numpy as np

BATCH_SIZE = 10000000

def to_key_value(dataset, key_sample_sets):
  dataset_name = dataset[0]
  dataset_type = np.uint64 if dataset[1] == "uint64" else np.uint32

  target_name = dataset_name + "_kv"
  print(f"Generate key-value dataset from {dataset_name} to {target_name}")
  d = np.fromfile(dataset_name, dtype=dataset_type)
  # assert d[0] == len(d) - 1
  print(d[0], len(d))
  d = d[1:]
  print(d[:10], "...", d[-10:])

  print(f"writing {d.shape[0]} key-values")
  with open(target_name, "w") as target:
    for batch_idx in range(d.shape[0] // BATCH_SIZE + int(d.shape[0] % BATCH_SIZE == 0)):
      print(f"Batch {batch_idx}")
      d_batch = d[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
      target.write("".join(f"{di} {idx}\n" for idx, di in enumerate(d_batch)))

  target_name = dataset_name + "_keyset"
  rng = np.random.default_rng(abs(hash(dataset_name)))
  sample_idx = rng.integers(d.shape[0], size=70000)
  print(f"Writing {sample_idx.shape[0]} sample keys to {target_name}")
  with open(target_name, "w") as target:
    for batch_idx in range(sample_idx.shape[0] // BATCH_SIZE + int(sample_idx.shape[0] % BATCH_SIZE != 0)):
      sample_idx_batch = sample_idx[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
      d_batch = d[sample_idx_batch]
      target.write("".join(f"{di}\n" for di in d_batch))

  for idx in key_sample_sets:
    sample_keys(d, idx)

def sample_keys(d, idx):
  target_name = f"{dataset_name}_keyset_{idx}"
  rng = np.random.default_rng(abs(hash(dataset_name) + idx))
  sample_idx = rng.integers(d.shape[0], size=70000)
  print(f"Writing {sample_idx.shape[0]} sample keys to {target_name}")
  with open(target_name, "w") as target:
    for batch_idx in range(sample_idx.shape[0] // BATCH_SIZE + int(sample_idx.shape[0] % BATCH_SIZE != 0)):
      sample_idx_batch = sample_idx[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
      d_batch = d[sample_idx_batch]
      target.write("".join(f"{di}\n" for di in d_batch))

if __name__ == "__main__":
  DATASETS = [
    ("data/books_200M_uint32", "uint32"),
    ("data/books_200M_uint64", "uint64"),
    ("data/books_400M_uint64", "uint64"),
    ("data/books_600M_uint64", "uint64"),
    ("data/books_800M_uint64", "uint64"),  # ***
    ("data/fb_200M_uint64", "uint64"),  # ***
    ("data/lognormal_200M_uint32", "uint32"),
    ("data/lognormal_200M_uint64", "uint64"),  # ***
    ("data/normal_200M_uint32", "uint32"),
    ("data/normal_200M_uint64", "uint64"),  # ***
    ("data/osm_cellids_200M_uint64", "uint64"),
    ("data/osm_cellids_400M_uint64", "uint64"),
    ("data/osm_cellids_600M_uint64", "uint64"),
    ("data/osm_cellids_800M_uint64", "uint64"),  # ***
    ("data/uniform_dense_200M_uint32", "uint32"),
    ("data/uniform_dense_200M_uint64", "uint64"),  # ***
    ("data/uniform_sparse_200M_uint32", "uint32"),
    ("data/uniform_sparse_200M_uint64", "uint64"),  # ***
    ("data/wiki_ts_200M_uint64", "uint64"),  # ***
  ]
  for dataset in DATASETS:
    to_key_value(dataset, list(range(40)))
