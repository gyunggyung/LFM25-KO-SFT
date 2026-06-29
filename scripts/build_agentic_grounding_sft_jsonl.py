#!/usr/bin/env python3
"""Build small grounded document/log/tool-use SFT rows for the agentic stage."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable


START = "<|startoftext|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
DEFAULT_SYSTEM = (
    "너는 한국어 LFM 에이전트다. 사용자가 제공한 문서, 로그, 파일 내용만 근거로 "
    "상황을 요약하고, 필요한 조치와 검증 방법을 간결하게 제시한다. 근거가 부족하면 "
    "부족하다고 말한다. 도구가 필요하면 LFM tool-call 형식만 출력한다."
)


def chat_text(messages: list[tuple[str, str]]) -> str:
    parts = [START]
    for role, content in messages:
        parts.append(f"{IM_START}{role}\n{content.strip()}{IM_END}\n")
    return "".join(parts)


def read_excerpt(path: Path, max_chars: int = 5000) -> str:
    if not path.exists():
        return f"[missing: {path}]"
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + "\n\n[...중간 생략...]\n\n" + text[-tail:]


def parse_latest_train_log(path: Path) -> dict[str, float | int | str] | None:
    if not path.exists():
        return None
    latest = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "step" in row and "planned_steps" in row:
            latest = row
    if not latest:
        return None
    step = int(latest["step"])
    planned = int(latest["planned_steps"])
    eps = float(latest.get("examples_per_sec") or 0.0)
    remaining_steps = max(planned - step, 0)
    # The current training setup uses an effective batch of 128 sequences/update.
    eta_hours = (remaining_steps * 128 / eps / 3600) if eps > 0 else math.nan
    return {
        "step": step,
        "planned_steps": planned,
        "progress": round(step / planned * 100, 2) if planned else 0,
        "examples_per_sec": eps,
        "loss": latest.get("loss", ""),
        "eta_hours": round(eta_hours, 2) if not math.isnan(eta_hours) else "unknown",
        "time": latest.get("time", ""),
    }


def emit(handle, source: str, messages: list[tuple[str, str]]) -> None:
    handle.write(
        json.dumps(
            {
                "text": chat_text(messages),
                "source": source,
                "conversion": "local_agentic_grounding_to_lfm_chat",
            },
            ensure_ascii=False,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="/home/work/.projects/LLM-OS-Models/Terminal/lfm2_ko_sft")
    parser.add_argument("--data-root", default="/home/work/.data/lfm2_ko_sft")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    workspace = Path(args.workspace)
    data_root = Path(args.data_root)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    readme = read_excerpt(workspace / "README.md", 6000)
    runbook = read_excerpt(workspace / "docs" / "RUNBOOK_20260628.ko.md", 5000)
    agent_doc = read_excerpt(workspace / "docs" / "AGENT_HARNESS_20260629.ko.md", 5000)
    train_log_path = data_root / "models" / "LFM2.5-8B-A1B-KO-SFT-stage1-8k-legal-terminal-20260628" / "train_log.jsonl"
    train_tail = read_excerpt(train_log_path, 4500)
    latest = parse_latest_train_log(train_log_path)

    count = 0
    with out.open("w", encoding="utf-8") as handle:
        emit(
            handle,
            "lfm2ko_readme_grounded_summary",
            [
                ("system", DEFAULT_SYSTEM),
                ("user", f"[문서: README.md]\n{readme}\n\n질문: 이 프로젝트의 현재 학습 순서와 평가 원칙을 근거 기반으로 요약해라."),
                (
                    "assistant",
                    "요약: 이 프로젝트는 LFM2.5-8B-A1B를 한국어 SFT 모델로 만들기 위해 Stage0/0b/Stage1/Stage2 순서로 full-parameter SFT를 진행한다.\n\n"
                    "근거:\n"
                    "- README의 Current Status와 Training Order 표에 Stage1 8k, Stage2 plus KoTSQA, vLLM 평가 순서가 정리되어 있다.\n"
                    "- Why LFM-Style Preprocessing 섹션은 LFM tokenizer로 다시 토큰화해야 하는 이유를 설명한다.\n\n"
                    "판단: GPU 학습 체인이 우선이고, 평가는 학습 stage가 GPU를 놓은 뒤 vLLM으로 base/CPT/SFT를 같은 조건에서 비교해야 한다.\n\n"
                    "다음 조치: Stage2 final이 생기면 빠른 gate eval을 먼저 돌리고, 이후 agentic/Fable SFT를 별도 stage로 이어간다.",
                ),
            ],
        )
        count += 1

        emit(
            handle,
            "lfm2ko_train_log_diagnosis",
            [
                ("system", DEFAULT_SYSTEM),
                ("user", f"[로그: train_log.jsonl tail]\n{train_tail}\n\n질문: 이 학습이 정상 진행 중인지, 남은 작업과 검증 방법을 설명해라."),
                (
                    "assistant",
                    (
                        "상태: 로그에 step, planned_steps, loss, examples_per_sec가 계속 갱신되므로 학습은 진행 중이다.\n\n"
                        f"숫자: 최신 파싱 기준 {latest['step']} / {latest['planned_steps']} step, 진행률 {latest['progress']}%, "
                        f"속도 {latest['examples_per_sec']} examples/sec, loss {latest['loss']}, ETA 약 {latest['eta_hours']}시간이다.\n\n"
                        "근거: train_log tail의 연속 step 증가와 checkpoint 생성 로그가 같은 output_dir 아래에 존재한다.\n\n"
                        "검증 명령: `tail -n 20 <train_log.jsonl>`, `nvidia-smi`, `find <output_dir> -maxdepth 1 -type d -name 'checkpoint-*'` 순서로 확인한다."
                        if latest
                        else "상태: 제공된 로그에서 최신 step을 구조적으로 파싱하지 못했다.\n\n검증: train_log.jsonl의 JSON 행과 checkpoint 디렉터리 생성을 먼저 확인해야 한다."
                    ),
                ),
            ],
        )
        count += 1

        tool_call = '<|tool_call_start|>{"name":"read_file","arguments":{"path":"README.md","max_bytes":4000}}<|tool_call_end|>'
        emit(
            handle,
            "lfm2ko_tool_call_readme",
            [
                ("system", DEFAULT_SYSTEM),
                ("user", "README.md를 읽고 현재 학습/평가 실행법을 요약해라. 먼저 파일을 확인해라."),
                ("assistant", tool_call),
            ],
        )
        count += 1

        emit(
            handle,
            "lfm2ko_tool_result_grounding",
            [
                ("system", DEFAULT_SYSTEM),
                ("user", "README.md를 읽고 현재 학습/평가 실행법을 요약해라."),
                ("assistant", tool_call),
                ("tool", f"[README.md excerpt]\n{readme[:3500]}"),
                (
                    "assistant",
                    "README 기준 실행 순서는 Stage1/Stage2 SFT를 먼저 끝내고, 이후 vLLM 평가와 agent harness 평가를 실행하는 것이다. "
                    "학습 실행은 `scripts/run_lfm25_ko_sft_torchrun_lfmchat_dataset.sh`가 중심이고, 평가는 `scripts/run_vllm_eval_8gpu_queue.sh`와 "
                    "`agent_harness`를 사용한다. 문서에 없는 결과 점수는 주장하지 않는다.",
                ),
            ],
        )
        count += 1

        emit(
            handle,
            "lfm2ko_error_recovery_git_push_rejected",
            [
                ("system", DEFAULT_SYSTEM),
                (
                    "user",
                    "[로그]\n"
                    "To github.com:org/repo.git\n"
                    " ! [rejected]        main -> main (fetch first)\n"
                    "error: failed to push some refs\n\n"
                    "질문: 로컬 변경을 잃지 않고 push 가능하게 만드는 절차를 말해라.",
                ),
                (
                    "assistant",
                    "원인: 원격 main에 로컬에 없는 커밋이 있어서 non-fast-forward push가 거부됐다.\n\n"
                    "안전 절차:\n"
                    "1. `git status --short`로 로컬 변경을 확인한다.\n"
                    "2. 커밋할 변경은 먼저 `git add ... && git commit -m ...`로 보존한다.\n"
                    "3. `git pull --rebase origin main`으로 원격 변경을 로컬 커밋 아래에 재배치한다.\n"
                    "4. 충돌이 나면 파일을 열어 수동 해결하고 `git rebase --continue`를 실행한다.\n"
                    "5. `git push origin main`을 실행한다.\n\n"
                    "주의: `git reset --hard`, `git checkout -- .`, `git push --force`는 사용자가 명시적으로 허락하기 전에는 쓰지 않는다.",
                ),
            ],
        )
        count += 1

        emit(
            handle,
            "lfm2ko_vllm_eval_plan",
            [
                ("system", DEFAULT_SYSTEM),
                ("user", f"[문서: AGENT_HARNESS]\n{agent_doc}\n\n질문: SFT 완료 후 agentic 능력은 어떻게 평가해야 하나?"),
                (
                    "assistant",
                    "평가 순서: 먼저 vLLM으로 모델 endpoint를 띄우고, 같은 프롬프트를 harness에 넣어 도구 호출 형식과 최종 답변을 확인한다.\n\n"
                    "핵심 항목:\n"
                    "- README/문서 읽기: read_file tool-call을 올바르게 내는지 확인한다.\n"
                    "- 로그 진단: 학습 로그를 보고 step, loss, ETA, 다음 조치를 말하는지 확인한다.\n"
                    "- 코딩 보조: 파일 검색, 수정 계획, 테스트 명령을 안전하게 제안하는지 확인한다.\n"
                    "- 한국어 근거 답변: 문서에 없는 내용을 지어내지 않는지 확인한다.\n\n"
                    "판정: agentic SFT 직후에는 공식 BFCL/tau2 전체보다 smoke와 제한된 task를 먼저 돌려 품질 붕괴 여부를 확인한다.",
                ),
            ],
        )
        count += 1

    print(json.dumps({"output": str(out), "rows": count}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
