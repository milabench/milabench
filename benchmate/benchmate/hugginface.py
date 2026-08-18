

def download_hf_model(model_id, token=None, cache_dir=None, ignore_patterns=None):
    from huggingface_hub import snapshot_download

    # No default ignore_patterns: at server start, vllm's own get_model_path()
    # independently calls snapshot_download() with no filter and, running
    # offline (HF_HUB_OFFLINE=1) on compute nodes with no internet, rejects
    # the cache with IncompleteSnapshotError for any file missing from the
    # full upstream tree -- including ones we'd have deliberately skipped
    # (.gitattributes, LICENSE.md, README.md, merges.txt, ...). Since that
    # check can't be told about a partial download, the local snapshot must
    # be complete; only skip files when the caller explicitly passes patterns.
    snapshot_kwargs = {
        "repo_id": model_id,
    }
    if ignore_patterns:
        snapshot_kwargs["ignore_patterns"] = ignore_patterns

    if cache_dir:
        snapshot_kwargs["cache_dir"] = cache_dir

    if token:
        snapshot_kwargs["token"] = token

    print(f"Downloading {model_id}...")

    # snapshot_download() respects HF_HOME as the primary cache location
    # Falls back to HF_HOME/hub for storing models
    local_path = snapshot_download(**snapshot_kwargs)

    print(f"Model is available at {local_path}")


def download_hf_dataset(dataset_name, split=None, name=None, token=None):
    from datasets import load_dataset

    print(f"Downloading dataset: {dataset_name}" + (f" (split: {split})" if split else ""))

    load_kwargs = {}
    if token:
        load_kwargs["token"] = token

    if split:
        load_kwargs["split"] = split
    
    dataset = load_dataset(dataset_name, name=name, **load_kwargs)

    _print_hf_dataset_cache(dataset)

    return True


def _print_hf_dataset_cache(dataset):
    """Attempts to print the cache location(s) of a loaded HuggingFace dataset."""
    try:
        # Datasets with .cache_files attribute (common in datasets>=2.4.0)
        if hasattr(dataset, "cache_files"):
            files = dataset.cache_files
            if isinstance(files, list):
                for entry in files:
                    print(f"Cached at: {entry.get('filename', entry)}")
            elif isinstance(files, dict):
                for split_name, split_files in files.items():
                    for entry in split_files:
                        print(f"Cached at (split {split_name}): {entry.get('filename', entry)}")
        # Sometimes a split dataset is returned as a DatasetDict, print recursively
        elif isinstance(dataset, dict):
            for split_name, split_dataset in dataset.items():
                print(f"Checking cache for split: {split_name}")
                _print_hf_dataset_cache(split_dataset)
        elif hasattr(dataset, "data") and hasattr(dataset.data, "dataset") and hasattr(dataset.data.dataset, "cache_files"):
            files = dataset.data.dataset.cache_files
            for entry in files:
                print(f"Cached at: {entry.get('filename', entry)}")
        else:
            print("Could not determine local cache location for this dataset object.")
    except Exception as e:
        print(f"Error getting dataset cache location: {e}")