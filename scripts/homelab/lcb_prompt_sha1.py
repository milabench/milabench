"""LiveCodeBench v6 question_id -> hash of its chat messages, the same hash qb_launch.py's timing log writes
(prompt_sha1), so report_html.py can join each reply's time to its grading. Runs in QualityBench's LCB image,
offline; the HF cache is copied first (datasets locks its files, the shared cache stays untouched):

  cp -a <qb home>/hf/modules <qb home>/hf/datasets /tmp/hf
  podman run --rm --network none -v /tmp/hf:/hf -v <qb home>/hf/hub:/hf/hub:ro -v $PWD:/out -e HOME=/tmp \
      -e HF_HOME=/hf -e HF_DATASETS_OFFLINE=1 -w /lcb localhost/qualitybench-lcb:<tag> python /out/lcb_prompt_sha1.py
"""
import hashlib, json
from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
from lcb_runner.prompts.code_generation import format_prompt_generation
from lcb_runner.lm_styles import LMStyle
out = {}
for p in load_code_generation_dataset("v6"):
    msgs = json.loads(json.dumps(format_prompt_generation(p, LMStyle.OpenAIChat)))
    out[str(p.question_id)] = hashlib.sha1(json.dumps(msgs, sort_keys=True).encode()).hexdigest()[:16]
json.dump(out, open("/out/lcb_prompt_sha1.json", "w"))
print(len(out))
