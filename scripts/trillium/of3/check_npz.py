"""Integrity of every .npz under DIR (each is a zip archive): opens it and checks every member's CRC (zipfile's
testzip), 8 processes. Prints the count checked and each bad file with its error.
Usage: python check_npz.py DIR"""
import concurrent.futures as cf
import sys
import zipfile
from pathlib import Path


def check(path):
    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
        return path, None if bad is None else f"bad member {bad}"
    except Exception as e:
        return path, f"{type(e).__name__}: {e}"


files = sorted(str(p) for p in Path(sys.argv[1]).rglob("*.npz"))
with cf.ProcessPoolExecutor(8) as ex:
    bad = [(p, err) for p, err in ex.map(check, files, chunksize=64) if err]
print(f"{len(files)} .npz checked, {len(bad)} bad")
for p, err in bad:
    print(f"{p}: {err}")
