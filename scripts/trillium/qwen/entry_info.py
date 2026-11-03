"""Print the launch settings of one entry of a milabench config, for tt_trillium.sh.

    python entry_info.py <config.yaml> <entry>

Output (shell assignments): NUM_MACHINES (0 when the entry uses every node of
the allocation, num_machines >= 999), MAX_DURATION (seconds), LAYOUT.
"""

import sys

from milabench.config import build_config


def main(config_file: str, name: str) -> int:
    config = build_config(config_file)
    if name not in config:
        print(f"no entry {name!r} in {config_file}", file=sys.stderr)
        return 2
    entry = config[name]
    machines = int(entry.get("num_machines", 1))
    print(f"NUM_MACHINES={0 if machines >= 999 else machines}")
    print(f"MAX_DURATION={int(entry.get('max_duration', 600))}")
    print(f"LAYOUT={entry.get('argv', {}).get('--layout') or 'fsdp'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
