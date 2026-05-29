"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import argparse
import time
from functools import partial
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

base_dir = get_base_dir()
DATASET_ENV_VAR = "NANOCHAT_PRETRAIN_DATASET"
DATASET_SPECS = {
    "gigaverbo-v2": {
        "base_url": "https://huggingface.co/datasets/Polygl0t/gigaverbo-v2/resolve/main/default",
        "data_dir": os.path.join(base_dir, "base_data_gigaverbo_v2"),
        "shard_count": 224,
    },
    "gigaverbo-v2-synth": {
        "base_url": "https://huggingface.co/datasets/Polygl0t/gigaverbo-v2-synth/resolve/main/default",
        "data_dir": os.path.join(base_dir, "base_data_gigaverbo_v2_synth"),
        "shard_count": 224,
    },
}
DEFAULT_DATASET_TAG = "gigaverbo-v2"
DATASET_ALIASES = {
    "v2": "gigaverbo-v2",
    "gigaverbo2": "gigaverbo-v2",
    "synth": "gigaverbo-v2-synth",
    "gigaverbo2-synth": "gigaverbo-v2-synth",
}


def _resolve_dataset_tag(dataset_tag=None):
    if dataset_tag is None:
        dataset_tag = os.environ.get(DATASET_ENV_VAR, DEFAULT_DATASET_TAG)
    tag = DATASET_ALIASES.get(dataset_tag.strip().lower(), dataset_tag.strip().lower())
    if tag not in DATASET_SPECS:
        options = ", ".join(sorted(DATASET_SPECS.keys()))
        raise ValueError(f"Unknown dataset '{dataset_tag}'. Expected one of: {options}")
    return tag


def _get_dataset_spec(dataset_tag=None):
    tag = _resolve_dataset_tag(dataset_tag)
    return tag, DATASET_SPECS[tag]


def index_to_filename(index, dataset_tag=None):
    _, spec = _get_dataset_spec(dataset_tag)
    shard_count = spec["shard_count"]
    return f"train-{index:05d}-of-{shard_count:05d}.parquet"


# Backward-compatible module-level constants, resolved to the default dataset.
BASE_URL = DATASET_SPECS[DEFAULT_DATASET_TAG]["base_url"]
DATA_DIR = DATASET_SPECS[DEFAULT_DATASET_TAG]["data_dir"]
MAX_SHARD = DATASET_SPECS[DEFAULT_DATASET_TAG]["shard_count"] - 1

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(data_dir=None, warn_on_legacy=False, dataset_tag=None):
    """ Looks into a data dir and returns full paths to all parquet files. """
    resolved_tag, spec = _get_dataset_spec(dataset_tag)
    default_data_dir = spec["data_dir"]
    data_dir = default_data_dir if data_dir is None else data_dir

    # Legacy-supporting code due to dataset upgrades over time.
    # This code will eventually be deleted.
    can_fallback_to_legacy = (
        dataset_tag is None
        and resolved_tag == DEFAULT_DATASET_TAG
        and data_dir == default_data_dir
    )
    if not os.path.exists(data_dir) and can_fallback_to_legacy:
        climbmix_data_dir = os.path.join(base_dir, "base_data_climbmix")
        fineweb_data_dir = os.path.join(base_dir, "base_data")
        if os.path.exists(climbmix_data_dir):
            data_dir = climbmix_data_dir
        elif os.path.exists(fineweb_data_dir):
            data_dir = fineweb_data_dir
        else:
            data_dir = default_data_dir
        if warn_on_legacy:
            print()
            print("=" * 80)
            print("  WARNING: DATASET UPGRADE REQUIRED")
            print("=" * 80)
            print()
            print(f"  Could not find: {default_data_dir}")
            print()
            print(f"  nanochat is configured to use Polygl0t/{resolved_tag} for pretraining.")
            print("  To prepare the selected dataset and tokenizer, run these two commands:")
            print()
            print(f"    python -m nanochat.dataset --dataset {resolved_tag} -n 64")
            print("    python -m scripts.tok_train           # re-train tokenizer on pretraining data")
            print()
            print(f"  For now, falling back to available legacy dataset at: {data_dir}")
            print("=" * 80)
            print()

    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Could not find dataset directory '{data_dir}' for dataset '{resolved_tag}'. "
            f"Run: python -m nanochat.dataset --dataset {resolved_tag}"
        )

    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

def parquets_iter_batched(split, start=0, step=1, dataset_tag=None):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files(dataset_tag=dataset_tag)
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

# -----------------------------------------------------------------------------
def download_single_file(index, dataset_tag=None):
    """ Downloads a single file index, with some backoff """
    resolved_tag, spec = _get_dataset_spec(dataset_tag)
    data_dir = spec["data_dir"]
    base_url = spec["base_url"]

    # Construct the local filepath for this file and skip if it already exists
    filename = index_to_filename(index, dataset_tag=resolved_tag)
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = f"{base_url}/{filename}"
    print(f"Downloading {filename}...")

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pretraining dataset shards")
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            f"Dataset name ({', '.join(sorted(DATASET_SPECS.keys()))}). "
            f"Default is {DATASET_ENV_VAR} or '{DEFAULT_DATASET_TAG}'."
        ),
    )
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of train shards to download (default: all), -1 = all")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    args = parser.parse_args()
    resolved_tag, spec = _get_dataset_spec(args.dataset)
    data_dir = spec["data_dir"]
    max_shard = spec["shard_count"] - 1

    # Prepare the output directory
    os.makedirs(data_dir, exist_ok=True)

    # The way this works is that the user specifies the number of train shards to download via the -n flag.
    # In addition to that, the validation shard is *always* downloaded and is pinned to be the last shard.
    num_train_shards = max_shard if args.num_files == -1 else min(args.num_files, max_shard)
    ids_to_download = list(range(num_train_shards))
    ids_to_download.append(max_shard) # always download the validation shard

    # Download the shards
    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    print(f"Dataset: {resolved_tag}")
    print(f"Target directory: {data_dir}")
    print()
    download_fn = partial(download_single_file, dataset_tag=resolved_tag)
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_fn, ids_to_download)

    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {data_dir}")
