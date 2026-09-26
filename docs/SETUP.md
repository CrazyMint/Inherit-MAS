# Additional Setup

Run commands below from the repository root after completing the main [installation and configuration](../README.md).

## API Configuration

Use an OpenAI-compatible Chat Completions endpoint with tool calling. Set `OPENAI_BASE_URL` and `OPENAI_API_KEY` in `.env`, and the corresponding API model names in `credentials.json`. For a model with a separate endpoint or key, add `base_url` and `api_key_env` to its entry under `api.models`; put the named key in `.env`. Per-model URLs override the shared URL. Model access depends on your provider and account.

## EvoMAS and TacoMAS

EvoMAS and TacoMAS on WorkBench, and EvoMAS with GPT-4o-mini on HotpotQA, need the separate environments below. All other runs use the main environment.

Fetch the pinned external code and apply the bundled modifications:

```bash
python scripts/setup_baselines.py
```

Downloads stay in Git-ignored `baselines/*/upstream/` directories. See [THIRD_PARTY.md](../THIRD_PARTY.md) for their separate terms. These WorkBench runners execute generated Python. Use an isolated container or VM without unrelated files or credentials; a Conda environment is not a security sandbox.

Create the separate environments. Do not combine these requirements with the main environment:

```bash
conda create -n inherit-evomas python=3.11 -y
conda run -n inherit-evomas python -m pip install -r baselines/evomas/setup-requirements.txt
conda create -n inherit-tacomas python=3.11 -y
conda run -n inherit-tacomas python -m pip install -r baselines/tacomas/requirements.txt
```

Set the executable paths in `.env`:

```dotenv
INHERIT_EVOMAS_PYTHON=/path/to/envs/inherit-evomas/bin/python
INHERIT_TACOMAS_PYTHON=/path/to/envs/inherit-tacomas/bin/python
```

Run the public commands from the main `inherit-mas` environment:

```bash
python run.py run workbench --method evomas --limit 3 --task-concurrency 1 --usd-cap 2 --out runs/evomas-workbench
python run.py run workbench --method tacomas --limit 3 --task-concurrency 1 --usd-cap 2 --out runs/tacomas-workbench
python run.py run hotpotqa --method tacomas --limit 3 --task-concurrency 1 --usd-cap 2 --out runs/tacomas-hotpotqa
```

EvoMAS on WorkBench and GPT-4o-mini HotpotQA requires `--task-concurrency 1`. For the latter, set `HOTPOT_BM25_URL=http://127.0.0.1:8765` in `.env` and start the BM25 service in another terminal, in the main environment:

```bash
python scripts/serve_bm25.py
```

Then run:

```bash
python run.py run hotpotqa --method evomas --limit 3 --task-concurrency 1 --usd-cap 2 --out runs/evomas-hotpotqa
```

EvoMAS with Qwen3-32B on HotpotQA needs only the main environment and does not require the BM25 service above.

## Local Qwen3-32B

Provide local Qwen3-32B weights and a separate environment with vLLM 0.21.0. Start one tensor-parallel replica on two GPUs:

```bash
python scripts/serve_qwen.py --model /path/to/Qwen3-32B --devices 0,1 --port 8000
```

Set the local endpoint and the provenance file written by that command in `.env`. TacoMAS on WorkBench also needs the tokenizer path:

```dotenv
INHERIT_QWEN_BASE_URLS=http://127.0.0.1:8000/v1
INHERIT_QWEN_PROVENANCE=runs/qwen/serve_provenance.json
INHERIT_QWEN_TOKENIZER_PATH=/path/to/Qwen3-32B
```

Run from the main environment:

```bash
python run.py run workbench --method inherit-mas --backbone qwen3-32b --limit 3 --task-concurrency 1 --usd-cap 2 --out runs/inherit-qwen-workbench
python run.py run hotpotqa --method evoagent --backbone qwen3-32b --limit 3 --task-concurrency 1 --usd-cap 2 --out runs/evoagent-qwen-hotpotqa
```

Change `--method` to run the other baselines, completing any required setup above. Controller and judge API calls remain enabled where applicable. EvoAgent uses Qwen for every role and makes no external API calls. Run commands do not download model weights or start or stop GPU servers automatically.

## Score and Resume

Every method and backbone uses the same scoring command:

```bash
python run.py score --out runs/evomas-workbench
```

`SCORE.json` contains scores and usage summaries, `RUN_STATUS.json` tracks progress, and `units/` contains task outputs. Scoring makes no model calls and rejects incomplete runs. Repeating the original run command resumes checkpoints; use a new output directory for a different task selection or configuration.
