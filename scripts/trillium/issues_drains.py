#!/usr/bin/env python3
"""Rebuild the node event table of ISSUES.md from Slurm. Read-only on the cluster; one ssh call, five small queries.

    python3 issues_drains.py                 print the table
    python3 issues_drains.py --update FILE   replace the two generated blocks in FILE

Sources: sacctmgr's event log and reservations, sinfo -R, scontrol's BootTime, $SCRATCH/trillium/ops/node_states.log,
sacct for our jobs. NOT_IN_SLURM and EXPLAIN below are hand-written: what only a journal holds, and our reading.
"""
import argparse
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta

START = "2026-11-03T00:00"
NODES = "trig[0101-0152]"
REMOTE = f"""
echo @@now; date +%FT%T
echo @@events; sacctmgr -n -P show event where start={START} nodes="{NODES}" format=NodeName,TimeStart,TimeEnd,State,Reason,User
echo @@open; sinfo -R -h -n "{NODES}" -o "%N|%T|%E|%u|%H"
echo @@res; sacctmgr -n -P show reservation start={START} format=Name,NodeNames,TimeStart,TimeEnd,Flags
echo @@live; scontrol show reservation -o
echo @@nodes; scontrol show nodes -o "{NODES}"
echo @@jobs; sacct -X -n -P -u $USER -S {START} -E now -o JobID,JobName,State,Start,End,NodeList
echo @@log; cat $SCRATCH/trillium/ops/node_states.log
"""
SSH = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "trillium-gpu", REMOTE]
UNRESPONSIVE = "health.chk.nvidia-smi.unresponsive"
STEPD = "health.chk.slurmstepd..high.cpu.root.process"
NO_ROW = "boot, no event row"
FAR = datetime(9999, 1, 1)

# What Slurm's records do not hold or will lose. (nodes, start, end, end is an upper bound, state, reason, set by, source)
NOT_IN_SLURM = [
    ("trig[0122,0123,0133,0138,0140,0143,0145,0152]", "2026-11-05T00:09", "2026-11-06T12:44", True, "no Slurm state", "GPUs capped at 200 W", "not visible", "admin statement for the time, journal 11-06 10:35; end: last node's 1,000 W read, journal 11-06 12:50"),
    ("trig0101", "2026-11-05T06:23:55", None, False, "draining", UNRESPONSIVE, "root", "scontrol, journal 11-05"),
    ("trig[0141-0142]", "2026-11-05T19:07:11", "2026-11-05T19:29", True, "draining", UNRESPONSIVE, "root", "scontrol, journal 11-05 19:15, 19:40"),
    ("trig[0123,0133,0138,0140,0143]", "2026-11-06T10:19:51", "2026-11-06T10:41", True, "no Slurm state", "200 W caps cleared by admins", "admins", "srygaard's checks: hw/srygaard-hc/README.md"),
    ("trig0122", "2026-11-06T11:48:57", "2026-11-06T11:53:09", False, "BOOT", NO_ROW, "not visible", "scontrol BootTime, SlurmdStartTime"),
    ("trig0138", "2026-11-06T19:31:31", None, False, "draining", "health.chk.gpu.ecc.error", "root", "journal 11-06 19:40"),
    ("trig0134", "2026-11-06T21:21:27", None, False, "draining", STEPD, "root", "journal 11-06 22:20"),
    ("trig0122", "2026-11-06T21:56:31", None, False, "draining", STEPD, "root", "journal 11-06 22:20"),
    ("trig0136", "2026-11-06T22:31:31", "2026-11-06T23:33:37", True, "draining", STEPD, "root", "journal 11-06 22:40, gone from node_states.log"),
]

# First match wins: (node regex, reason regex, start regex, text). Say who says it: measured, admin statement, inferred.
EXPLAIN = [
    (".", "Reboot ASAP", "2026-11-03", "Reboot before the reservation's 16:50 start; no reason given to us. Unknown. Per-node times: rerun sacctmgr"),
    (".", "Reboot ASAP", "2026-11-05", "Kernel change, `intel_pstate=passive` (admin statement); [HW-21]. Per-node boots: `analysis/pstate_boots.tsv`"),
    (".", "GPUs capped", "", "Power glitch (admin statement); first read at 00:21 (measured). [HW-16]"),
    (".", "caps cleared", "", "Between the last 200 W read and the first 1,000 W read, on srygaard's jobs (measured). No reboot. [HW-16]"),
    ("trig0101", "unresponsive", "", "Probably one slow `nvidia-smi` as the job ended (inferred). [HW-07]"),
    ("trig014[12]", "unresponsive", "", "Same reason as `trig0101`, mid-job, no reboot; cause not verified. [HW-07]"),
    ("trig0146", "unresponsive|gres/gpu|trig0146", "", "7 of 8 GPUs (Slurm reason); which GPU and what fixed it not stated. [HW-14]"),
    ("trig0152", "gpu.not.equal.8|gres/gpu|trig0152-", "", "7 of 8 GPUs (Slurm reason); what fixed it not stated. [HW-12]"),
    ("trig0152", "Not responding", "2026-11-03", "Just before its 7 of 8 GPU state; unknown. [HW-12]"),
    ("trig0152", "Not responding", "2026-11-05", "Unknown; capped and unused by us then. [HW-13]"),
    ("trig0152", "gpu_cap_test", "", "Name only: test of the caps, ends about when they clear. [HW-16]"),
    ("trig014[89]|trig015", "AI_inference", "2026-1", "Inference service until 11-04 (reservation name; journal 11-02)"),
    ("trig014[89]|trig015", "Not responding", "2026-11-0[34]", "Inside `AI_inference-nodes`, same second on 4 nodes; maintenance (inferred). Unknown"),
    ("trig0150", "unresponsive", "", "Inside `AI_inference-nodes`, before the campaign. Unknown"),
    ("trig0122", "img_test", "", "Image test (name only); firmware updated inside it (inferred). [HW-17]"),
    ("trig0122", "Not responding", "2026-11-05", "Inside `img_test`; booted 16:24:55 (journal). Unknown"),
    ("trig0122", "Not responding", "2026-11-06", "Inside `img_test`; a boot about 10:18 (journal 11-06 10:40). Unknown"),
    ("trig0122", "puppet", "", "Root's `puppet` over the site's CPU check, inside `img_test` (reason text). Cleared at a later check"),
    ("trig0122", "mlxfwmanager", "", "Root's firmware tool over the CPU check, so firmware update under way (inferred); `mlx5_1` new by 12:28 (measured). [HW-17]"),
    ("trig0122", "boot, no event", "", "Inside `img_test`, after the firmware tool; `mlx5_1` new by 12:28 (measured). [HW-17]"),
    ("trig0138", "ecc", "", "GPU 7 corrected DRAM errors, 10 then 14 (measured); site's threshold unknown. [HW-02]"),
    ("trig0122|trig0134|trig0136", "slurmstepd", "", "`slurmstepd` accounting thread near 80% of a core (measured); from process count (inferred). [HW-01]"),
    ("trig013[45]|trig0148", "Not responding|power_throttle", "", "PSU change from N+N to N+1, reboot 13:49-13:50 (admin statement). [HW-18]"),
    ("trig0135", "trig0135-", "", "Old firmware (admin statement); no readable firmware differs (measured). [HW-19]"),
    ("trig0140", "Not responding", "", "Reboot 12:04:11 (BootTime), after which the fabric error is gone (measured); who and why not stated. [HW-15]"),
    ("trig0145", "gpu_issue|trig0145-", "", "Held for more cap tests (admin statement). [HW-16]"),
    ("trig0145", "Not responding", "", "Reboot 12:07:07 (BootTime), in `gpu_issue`; back at 1,000 W by 12:44 (measured). [HW-16]"),
    ("trig0145", "nvidia-smi..high", "", "Root's `nvidia-smi` over the CPU check as our job ended (reason text); cleared at a later check (inferred)"),
    ("trig0114", "psu_issue", "", "Rebooted to look at power and meter, no effect; a bad PSU suspected, not sure (admin statement). [HW-03]"),
    ("trig0114", "Not responding", "", "Telemetry ends 17:03:12, inside `psu_issue` (measured). Failed or stopped: not known. [HW-03]"),
]


@dataclass
class Event:
    nodes: list
    start: str
    end: str | None
    end_by: bool
    state: str
    reason: str
    who: str
    source: list = field(default_factory=list)
    start_by: bool = False
    ongoing: bool = False


def expand(hostlist):
    """trig[0101-0103,0105] -> trig0101, trig0102, trig0103, trig0105."""
    m = re.fullmatch(r"(\w+)\[(.*)\]", hostlist)
    if not m:
        return [hostlist]
    out = []
    for part in m.group(2).split(","):
        lo, _, hi = part.partition("-")
        out += [f"{m.group(1)}{i:0{len(lo)}d}" for i in range(int(lo), int(hi or lo) + 1)]
    return out


def compress(nodes):
    nums = sorted(int(n[4:]) for n in nodes)
    if len(nums) == 1:
        return f"trig{nums[0]:04d}"
    runs = []
    for n in nums:
        if runs and runs[-1][1] == n - 1:
            runs[-1][1] = n
        else:
            runs.append([n, n])
    return "trig[" + ",".join(f"{a:04d}" if a == b else f"{a:04d}-{b:04d}" for a, b in runs) + "]"


def hosts(text):
    """A,trig[1,2],B -> [A, trig1, trig2, B]."""
    return [n for h in re.findall(r"[^,\[]+(?:\[[^\]]*\])?", text) for n in expand(h)]


def t(s):
    return datetime.fromisoformat(s)


def fetch():
    out = subprocess.run(SSH, capture_output=True, text=True, check=True).stdout
    sections, name = {}, None
    for line in out.splitlines():
        if line.startswith("@@"):
            name = line[2:]
            sections[name] = []
        elif name and line.strip():
            sections[name].append(line)
    return sections


def touches(a, b):
    """Same node and reason, intervals meeting, one minute of slack. An open end runs on."""
    end_a = t(a.end) if a.end else FAR
    end_b = t(b.end) if b.end else FAR
    slack = timedelta(minutes=1)
    return a.nodes == b.nodes and a.reason == b.reason and t(b.start) <= end_a + slack and t(a.start) <= end_b + slack


def add(events, new):
    for e in events:
        if touches(e, new):
            e.start, e.start_by = (e.start, e.start_by) if e.start <= new.start else (new.start, new.start_by)
            if new.end and (not e.end or e.end_by or not new.end_by):
                e.end, e.end_by = new.end, new.end_by
            e.ongoing = e.ongoing or new.ongoing
            e.who = e.who or new.who
            e.source += [s for s in new.source if s not in e.source]
            return
    events.append(new)


def from_event_log(lines):
    """Rows of the event log, with contiguous rows of one node, state and reason joined. Reboots stay apart, to batch."""
    rows = sorted((line.split("|") for line in lines), key=lambda r: (r[0], r[1]))
    joined = []
    for node, start, end, state, reason, who in rows:
        p = joined[-1] if joined else None
        if p and p[0] == node and p[2] == start and p[3] == state and p[4] == reason:
            p[2] = end
        else:
            joined.append([node, start, end, state, reason, who])
    return joined


def batches(rows):
    """One Event per reboot batch: rows with the same start and reason."""
    groups = {}
    for r in rows:
        groups.setdefault((r[1], r[4]), []).append(r)
    return [
        Event(sorted(x[0] for x in g), start, None if any(x[2] == "Unknown" for x in g) else max(x[2] for x in g), False, g[0][3], f"{reason} ({len(g)} nodes)", g[0][5], ["event log"])
        for (start, reason), g in groups.items()
    ]


def collect(s):
    events = []
    for nodes, start, end, by, state, reason, who, source in NOT_IN_SLURM:
        events.append(Event(expand(nodes), start, end, by, state, reason, who, [source]))

    rows = from_event_log(s["events"])
    reboot = [r for r in rows if r[4].startswith("Reboot ASAP")]
    for node, start, end, state, reason, who in (r for r in rows if not r[4].startswith("Reboot ASAP")):
        add(events, Event([node], start, end if end != "Unknown" else None, False, state, reason, who, ["event log"]))
    events += batches(reboot)

    for line in s["open"]:
        nodes, state, reason, who, stamp = line.split("|")
        for node in hosts(nodes):
            add(events, Event([node], stamp, None, False, state, reason, who, ["sinfo"], ongoing=True))

    live = {m.group(1): m for m in (re.search(r"ReservationName=(\S+).*Users=(\S+)", x) for x in s["live"]) if m}
    for line in s["res"]:
        name, nodes, start, end, flags = line.split("|")
        if "MAINT" not in flags or "trig" not in nodes or end < START:
            continue
        is_open = end > s["now"][0]
        who = live[name].group(2) if name in live else "not recorded"
        for node in hosts(nodes):
            if 101 <= int(node[4:]) <= 152:
                add(events, Event([node], start, None if is_open else end, False, "maint reservation", name, who, ["reservation"], ongoing=is_open))

    # Boots with no row: a BootTime outside every down, drain or reboot row of that node. Slurm notices a reboot up to
    # SlurmdTimeout (300 s) late, so a boot just before a row belongs to it. BootTime moves by a second between reads.
    for line in s["nodes"]:
        node = re.search(r"NodeName=(\S+)", line).group(1)
        boot = re.search(r"BootTime=(\S+)", line).group(1)
        slurmd = re.search(r"SlurmdStartTime=(\S+)", line).group(1)
        covered = any(r[0] == node and t(r[1]) - timedelta(seconds=300) <= t(boot) <= (FAR if r[2] == "Unknown" else t(r[2])) for r in rows) or any(
            node in e.nodes and e.reason == NO_ROW and abs((t(e.start) - t(boot)).total_seconds()) < 5 for e in events)
        if not covered and boot >= START:
            events.append(Event([node], boot, slurmd, False, "BOOT", NO_ROW, "not visible", ["scontrol BootTime, SlurmdStartTime"]))

    # The change log: when something left, or appeared with no other record.
    sections = []
    for line in s["log"]:
        if line.startswith("== "):
            sections.append((line[3:].strip(), set()))
        elif "|" in line:
            state, nodes, reason = line.split("|", 2)
            for node in hosts(nodes):
                sections[-1][1].add((node, reason))
    for stamp, seen in sections:
        for node, reason in seen:
            if not any(e.nodes == [node] and e.reason == reason for e in events):
                events.append(Event([node], stamp, None, False, "see reason", reason, "not visible", ["node_states.log"], start_by=True))
    for e in events:
        if e.state == "BOOT" or len(e.nodes) > 1 or e.end or e.ongoing:
            continue
        gone = next((stamp for stamp, seen in sections if stamp > e.start and (e.nodes[0], e.reason) not in seen), None)
        if gone:
            e.end, e.end_by = gone, True
    return sorted(events, key=lambda e: (e.start, e.nodes)), rows, reboot


def parse_jobs(lines):
    jobs = []
    for line in lines:
        jid, name, state, start, end, nodelist = line.split("|")
        if nodelist.startswith("None") or start in ("None", "Unknown"):
            continue
        jobs.append((jid, name, state.split()[0], start, None if end == "Unknown" else end, set(hosts(nodelist))))
    return jobs


def jobs_at(e, jobs):
    """Our jobs on the event's nodes at its start, or ended in the 2 minutes before: drains are logged when a job ends."""
    at = t(e.start)
    hits = []
    for jid, name, state, start, end, nodes in jobs:
        if not nodes & set(e.nodes) or t(start) > at:
            continue
        if end is None or t(end) >= at:
            hits.append(f"{jid} `{name}`" + ("" if state in ("COMPLETED", "RUNNING") else f" {state}"))
        elif t(end) >= at - timedelta(minutes=2):
            hits.append(f"{jid} `{name}`, ended {end[11:]}")
    return "; ".join(hits) or "none of ours"


def explain(e):
    node = compress(e.nodes) if len(e.nodes) == 1 else e.nodes[0]
    for pattern, reason, prefix, text in EXPLAIN:
        if re.search(pattern, node) and re.search(reason, e.reason) and re.match(prefix, e.start):
            return text
    return "unknown"


def stamp(when):
    return when[5:].replace("T", " ")


def render(s):
    events, rows, reboot = collect(s)
    jobs = parse_jobs(s["jobs"])
    out = [
        f"Generated {s['now'][0]} by `issues_drains.py`. Event log: {len(s['events'])} rows, {len(reboot)} of them reboots in {len({r[1] for r in reboot})} batches. This table: {len(events)} rows.",
        "",
        "| Node | Start | End | State | Reason | Set by | Our job at the time | Our explanation | Source |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for e in events:
        start = ("by " if e.start_by else "") + stamp(e.start)
        end = "ongoing" if e.ongoing else ("by " if e.end_by else "") + stamp(e.end) if e.end else "unknown"
        nodes = compress(e.nodes)
        reason = e.reason.replace("|", "/")
        out.append(f"| `{nodes}` | {start} | {end} | {e.state} | {reason} | {e.who} | {jobs_at(e, jobs)} | {explain(e)} | {', '.join(e.source)} |")
    return "\n".join(out), events


def out_now(s):
    lines = ["| Node | State | Why | Our job on it |", "|---|---|---|---|"]
    jobs = [j for j in parse_jobs(s["jobs"]) if j[4] is None]
    for line in s["nodes"]:
        node = re.search(r"NodeName=(\S+)", line).group(1)
        state = re.search(r"\bState=(\S+)", line).group(1)
        if not re.search("DRAIN|DOWN|MAINT|INVAL|NOT_RESPONDING|FAIL", state):
            continue
        reason = re.search(r"Reason=(.*?\[.*?\])", line)
        res = [m.group(1) for x in s["live"] if "MAINT" in x and (m := re.search(r"ReservationName=(\S+).*Nodes=(\S+)", x)) and node in hosts(m.group(2))]
        why = reason.group(1) if reason else "reservation " + ", ".join(f"`{r}`" for r in res)
        on = "; ".join(f"{j[0]} `{j[1]}`" for j in jobs if node in j[5]) or "none"
        lines.append(f"| `{node}` | {state} | {why} | {on} |")
    return f"Read {s['now'][0]}.\n\n" + "\n".join(lines)


def replace_block(text, name, body):
    a, b = f"<!-- BEGIN {name}: generated by issues_drains.py -->", f"<!-- END {name} -->"
    return re.sub(re.escape(a) + ".*?" + re.escape(b), lambda _: f"{a}\n{body}\n{b}", text, flags=re.S)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--update", help="file to rewrite between its markers")
    args = p.parse_args()
    s = fetch()
    table, _ = render(s)
    if not args.update:
        print(table)
        print()
        print(out_now(s))
        return
    with open(args.update) as f:
        text = f.read()
    text = replace_block(replace_block(text, "out now", out_now(s)), "node events", table)
    with open(args.update, "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
