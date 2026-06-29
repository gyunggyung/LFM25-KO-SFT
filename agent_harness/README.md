# LFM2.5 KO SFT Agent Harness

Bounded command-following harness for the final
`LLM-OS-Models/LFM2.5-8B-A1B-KO-SFT` model.

It is intentionally narrower than a general agent. It routes prompts into
profiles that match the SFT mix:

- Korean legal QA / bar-style answer formatting
- Korean finance/accounting explanations
- terminal/tool-call status work
- Text2SQL
- bounded code assistant tasks
- general Korean instruction following

Run in mock mode without GPU/model:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft
python agent_harness/lfm2ko_agent.py --mock --smoke
```

Run the agentic task suite in mock mode. This verifies routing, tool-call
parsing, scoring, and JSONL reporting without occupying a GPU:

```bash
cd /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft
bash scripts/run_lfm2ko_agentic_eval.sh
```

Run against a vLLM OpenAI-compatible endpoint after the final SFT model is
available:

```bash
OPENAI_BASE_URL=http://localhost:1053/v1 \
OPENAI_API_KEY=EMPTY \
MODEL_NAME=lfm2-ko-sft \
python agent_harness/lfm2ko_agent.py \
  --workspace /home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft \
  "README.md를 읽고 이 모델의 학습/평가 실행법을 요약해라."
```

Run the same agentic suite against vLLM:

```bash
MODE=real \
AGENT_BACKEND=vllm \
OPENAI_BASE_URL=http://localhost:1053/v1 \
OPENAI_API_KEY=EMPTY \
MODEL_NAME=lfm2-ko-sft \
EXECUTE_TOOLS=1 \
ALLOW_SHELL=1 \
bash scripts/run_lfm2ko_agentic_eval.sh
```

Run against a CPU GGUF served by llama.cpp:

```bash
MODEL_GGUF=/path/to/LFM2.5-8B-A1B-KO-SFT-Q8_0.gguf \
CTX_SIZE=8192 \
PORT=8080 \
bash scripts/start_llamacpp_gguf_server_for_agent.sh
```

Then:

```bash
AGENT_BACKEND=llamacpp \
OPENAI_BASE_URL=http://localhost:8080/v1 \
MODEL_NAME=lfm2-ko-sft-gguf \
python agent_harness/lfm2ko_agent.py \
  --backend llamacpp \
  --context-window 8192 \
  --prompt-budget 20000 \
  "법률 문서를 근거 중심으로 요약해라."
```

For GGUF evaluation, keep the same suite and switch only the backend endpoint:

```bash
MODE=real \
AGENT_BACKEND=llamacpp \
OPENAI_BASE_URL=http://localhost:8080/v1 \
MODEL_NAME=lfm2-ko-sft-gguf \
AGENT_CONTEXT_WINDOW=8192 \
AGENT_PROMPT_BUDGET_CHARS=20000 \
EXECUTE_TOOLS=1 \
ALLOW_SHELL=1 \
bash scripts/run_lfm2ko_agentic_eval.sh
```

Tool execution is dry-run by default. Read-only shell inspection can be enabled
explicitly:

```bash
python agent_harness/lfm2ko_agent.py \
  --allow-shell --execute-tools \
  "현재 git 상태와 최근 커밋을 요약해라."
```

Write/destructive shell commands remain blocked unless the harness is modified
or run with explicit write permissions. This keeps the agent focused on the
tasks this SFT model is expected to do well.

The task suite is intentionally small and operational:

- grounded README/runbook status reading
- train-log diagnosis with numeric fields
- safe shell planning without destructive commands
- small code patch planning and verification commands
- Korean legal/finance answer boundaries
- Text2SQL generation
- missing-evidence refusal behavior
