import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm
import os
import argparse

# any arbitrary seed value will do, but this one is clearly the best.
np.random.seed(seed=42) 

parser = argparse.ArgumentParser(prog='gen_gmm')
parser.add_argument('--n', type=int, default=800,
                    help='number of keys to generate (in millions)')
parser.add_argument('--k', type=int, default=100,
                    help='number of clusters')
args = parser.parse_args()

MAX_U64 = 2**63 - 1
num_keys = args.n * 1_000_000
num_clusters = args.k
num_keys_per_cluster = num_keys // num_clusters
range_size = 10 * num_keys_per_cluster * np.log(num_keys_per_cluster)
blob_name = f"data/gmm_k{args.k}_{args.n}M_uint64"
assert range_size + MAX_U64 * (num_clusters - 1) / num_clusters < 2 ** 64
assert num_keys_per_cluster < MAX_U64 / num_clusters
print(f"MAX_U64= {MAX_U64}")
print(f"num_keys= {num_keys}")
print(f"num_clusters= {num_clusters}")
print(f"num_keys_per_cluster= {num_keys_per_cluster}")
print(f"range_size= {range_size}")
print(f"blob_name= {blob_name}")


def generate_bounded_normal(nkeys):
    # mostly from gen_norm.py
    keys = np.linspace(0, 1, nkeys + 2)[1:-1]
    keys = np.array_split(keys, 1000)
    keys = [norm.ppf(x) for x in keys]
    keys = np.array(keys).flatten()
    keys = (keys - np.min(keys)) / (np.max(keys) - np.min(keys))
    return keys



keys = []
for kdx in range(num_clusters):
    start_pos = MAX_U64 * kdx // num_clusters
    cluster_keys = generate_bounded_normal(num_keys_per_cluster) * range_size
    cluster_keys = cluster_keys.astype(np.uint64)
    cluster_keys += start_pos
    keys.append(cluster_keys)
    print(f"kdx= {kdx}: generated {len(set(cluster_keys))} unique keys")
keys = np.array(keys).flatten()

with open(blob_name, "wb") as f:
    f.write(struct.pack("Q", len(keys)))
    keys.tofile(f)

DIV = 100
plt.plot(keys[::DIV], np.arange(keys.shape[0])[::DIV] * 8)
plt.show()

