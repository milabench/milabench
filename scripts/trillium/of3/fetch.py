"""Download the S3 keys listed in KEYS (one per line) from the public bucket openfold3-data to DEST/<key>, 16 at a time.
A file already in DEST is skipped, so a rerun resumes. Prints the count and bytes fetched and the keys that failed.
Usage: python fetch.py KEYS DEST"""
import concurrent.futures as cf
import os
import sys
import urllib.request
from pathlib import Path

B = "https://openfold3-data.s3.amazonaws.com/"
keys = [k.strip() for k in open(sys.argv[1]) if k.strip()]
dest = Path(sys.argv[2])


def fetch(key):
    path = dest / key
    if path.exists():
        return key, path.stat().st_size, None
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(B + urllib.request.quote(key), timeout=120) as r:
            data = r.read()
    except Exception as e:
        return key, 0, e
    part = path.with_name(path.name + ".part")
    part.write_bytes(data)
    os.replace(part, path)
    return key, len(data), None


done = size = 0
failed = []
with cf.ThreadPoolExecutor(16) as ex:
    for key, n, err in ex.map(fetch, keys):
        if err:
            failed.append(f"{key}: {err}")
        else:
            done += 1
            size += n
        if (done + len(failed)) % 5000 == 0:
            print(f"{done + len(failed)} / {len(keys)}", flush=True)
print(f"fetched {done} files, {size / 1e9:.2f} GB; failed {len(failed)}")
print("\n".join(failed[:50]))
