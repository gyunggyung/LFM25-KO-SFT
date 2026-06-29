#!/usr/bin/env python3
"""Agentic behavior evaluation harness for LFM2.5 KO SFT models."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import lfm2ko_agent


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = ROOT / "agent_harness" / "agentic_eval_tasks.jsonl"
DEFAULT_OUT = ROOT / "logs" / "agent_eval" / "agentic_eval_report.jsonl"


def load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("id", f"task_{line_no}")
            row.setdefault("required_substrings", [])
            row.setdefault("forbidden_substrings", [])
            row.setdefault("requires_tool", False)
            tasks.append(row)
    return tasks


def contains_all(text: str, needles: list[str]) -> tuple[bool, list[str]]:
    lower = text.lower()
    missing = [x for x in needles if x.lower() not in lower]
    return not missing, missing


def contains_none(text: str, needles: list[str]) -> tuple[bool, list[str]]:
    lower = text.lower()
    found = [x for x in needles if x.lower() in lower]
    return not found, found


def parse_tool_name(content: str) -> str | None:
    call = lfm2ko_agent.parse_tool_call(content)
    if not call:
        return None
    return str(call.get("name", "missing_name"))


def run_agent_trace(
    cfg: lfm2ko_agent.HarnessConfig,
    prompt: str,
    profiles: dict[str, Any],
    profile_name: str,
) -> dict[str, Any]:
    prompt, prompt_clipped = lfm2ko_agent.clamp_text(prompt, cfg.prompt_budget)
    messages = [
        {"role": "system", "content": lfm2ko_agent.build_system_prompt(profile_name, profiles)},
        {"role": "user", "content": prompt},
    ]
    if prompt_clipped:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Harness note: the original user prompt was clipped to fit "
                    f"the configured context budget ({cfg.prompt_budget} chars)."
                ),
            }
        )

    trace: list[dict[str, Any]] = []
    final = ""
    for turn in range(cfg.max_turns):
        content = lfm2ko_agent.openai_chat_completion(cfg, messages)
        final = content
        tool_name = parse_tool_name(content)
        step: dict[str, Any] = {
            "turn": turn,
            "assistant": content,
            "tool_name": tool_name,
        }
        call = lfm2ko_agent.parse_tool_call(content)
        if not call:
            trace.append(step)
            break
        if call.get("name") == "invalid_tool_call":
            tool_result = {"ok": False, "error": call["arguments"]}
        else:
            tool_result = lfm2ko_agent.run_tool(cfg, call)
        step["tool_result"] = tool_result
        trace.append(step)
        messages.append({"role": "assistant", "content": content})
        messages.append(
            {
                "role": "user",
                "content": "<tool_result>\n"
                + lfm2ko_agent.clamp_text(
                    json.dumps(tool_result, ensure_ascii=False, indent=2),
                    cfg.prompt_budget,
                )[0]
                + "\n</tool_result>\n"
                + "Use this tool result to answer the original request. "
                + "Do not call another tool unless strictly necessary.",
            }
        )
    return {"answer": final, "trace": trace}


def score_answer(
    task: dict[str, Any],
    answer: str,
    picked_profile: str,
    trace: list[dict[str, Any]],
) -> dict[str, Any]:
    required_ok, missing = contains_all(answer, task.get("required_substrings", []))
    forbidden_ok, forbidden_found = contains_none(answer, task.get("forbidden_substrings", []))
    tool_names = [step["tool_name"] for step in trace if step.get("tool_name")]
    tool_required = bool(task.get("requires_tool"))
    tool_ok = bool(tool_names) if tool_required else True
    profile_ok = picked_profile == task.get("profile", picked_profile)
    passed = required_ok and forbidden_ok and tool_ok and profile_ok
    return {
        "passed": passed,
        "profile_ok": profile_ok,
        "required_ok": required_ok,
        "missing_required": missing,
        "forbidden_ok": forbidden_ok,
        "forbidden_found": forbidden_found,
        "tool_required": tool_required,
        "tool_ok": tool_ok,
        "tool_names": tool_names,
        "turns": len(trace),
        "answer_chars": len(answer),
    }


def run_eval(args: argparse.Namespace) -> int:
    profiles = lfm2ko_agent.load_profiles(Path(args.profiles))
    workspace = lfm2ko_agent.normalize_workspace(args.workspace)
    cfg = lfm2ko_agent.HarnessConfig(
        backend=args.backend,
        endpoint=args.endpoint,
        api_key=args.api_key,
        model=args.model,
        workspace=workspace,
        profile=args.profile,
        context_window=args.context_window,
        prompt_budget=args.prompt_budget,
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        execute_tools=args.execute_tools,
        allow_shell=args.allow_shell,
        allow_write=args.allow_write,
        mock=args.mock,
    )

    tasks = load_tasks(Path(args.tasks))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    config = asdict(cfg)
    config["workspace"] = str(cfg.workspace)
    summary = {
        "task_count": len(tasks),
        "passed": 0,
        "failed": 0,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config,
    }

    with out_path.open("w", encoding="utf-8") as out:
        for task in tasks:
            requested_profile = task.get("profile", args.profile)
            picked = lfm2ko_agent.pick_profile(task["prompt"], profiles, requested_profile)
            started = time.time()
            try:
                result = run_agent_trace(cfg, task["prompt"], profiles, picked)
                answer = result["answer"]
                trace = result["trace"]
                error = None
            except Exception as exc:  # noqa: BLE001 - keep eval running.
                answer = ""
                trace = []
                error = str(exc)
            elapsed = time.time() - started
            score = score_answer(task, answer, picked, trace)
            if error:
                score["passed"] = False
                score["error"] = error
            summary["passed" if score["passed"] else "failed"] += 1
            row = {
                "id": task["id"],
                "profile": task.get("profile"),
                "picked_profile": picked,
                "elapsed_sec": round(elapsed, 3),
                "score": score,
                "prompt": task["prompt"],
                "answer": answer,
                "trace": trace,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"{task['id']}: {'PASS' if score['passed'] else 'FAIL'} "
                f"profile={picked} tools={score['tool_names']} chars={score['answer_chars']}"
            )
            if args.print_answers:
                print(answer[: args.print_answer_chars])
                print("-" * 80)

        summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        out.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["failed"] and args.allow_failures:
        print("agentic_eval_failures_allowed=true")
        return 0
    return 0 if summary["failed"] == 0 else 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LFM2 KO agentic eval tasks")
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--backend", choices=("vllm", "llamacpp", "openai"), default="vllm")
    parser.add_argument("--endpoint", default="http://localhost:1053/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="lfm2-ko-agentic-sft")
    parser.add_argument("--workspace", default=str(ROOT))
    parser.add_argument("--profiles", default=str(lfm2ko_agent.DEFAULT_PROFILES))
    parser.add_argument("--profile", default="auto")
    parser.add_argument("--context-window", type=int, default=8192)
    parser.add_argument("--prompt-budget", type=int, default=24000)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--execute-tools", action="store_true")
    parser.add_argument("--allow-shell", action="store_true")
    parser.add_argument("--allow-write", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--print-answers", action="store_true")
    parser.add_argument("--print-answer-chars", type=int, default=1200)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    return run_eval(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
