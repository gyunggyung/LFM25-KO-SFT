#!/usr/bin/env python3
"""Small command-following harness for LFM2.5 KO SFT models.

The harness targets the model behavior trained in this workspace: Korean legal
and finance QA, terminal/tool-call style outputs, Text2SQL, and bounded code
assistant tasks. It intentionally keeps execution limited and dry-run by default.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILES = ROOT / "agent_harness" / "profiles.json"
DEFAULT_SMOKE_TASKS = ROOT / "agent_harness" / "smoke_tasks.jsonl"

TOOL_CALL_RE = re.compile(
    r"<\|tool_call_start\|>\s*(.*?)\s*<\|tool_call_end\|>",
    re.DOTALL,
)

DENY_SHELL_PATTERNS = [
    r"\brm\b",
    r"\bsudo\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bmkfs\b",
    r"\bdd\b",
    r"\bmount\b",
    r"\bumount\b",
    r">\s*/",
    r"\bchmod\s+-R\b",
    r"\bchown\s+-R\b",
    r"\bgit\s+reset\b",
    r"\bgit\s+checkout\b",
    r"\bgit\s+clean\b",
    r"\bkill\s+-9\b",
]

ALLOW_SHELL_PREFIXES = (
    "pwd",
    "ls",
    "find",
    "rg",
    "sed",
    "head",
    "tail",
    "wc",
    "cat",
    "git status",
    "git diff",
    "git log",
    "python -m py_compile",
    "python3 -m py_compile",
)


@dataclass(frozen=True)
class HarnessConfig:
    backend: str
    endpoint: str
    api_key: str
    model: str
    workspace: Path
    profile: str
    context_window: int
    prompt_budget: int
    max_turns: int
    max_tokens: int
    temperature: float
    top_p: float
    execute_tools: bool
    allow_shell: bool
    allow_write: bool
    mock: bool


def load_profiles(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_workspace(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def ensure_under_workspace(workspace: Path, target: str | Path) -> Path:
    target_path = Path(target)
    if not target_path.is_absolute():
        target_path = workspace / target_path
    resolved = target_path.expanduser().resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"path outside workspace: {resolved}") from exc
    return resolved


def pick_profile(prompt: str, profiles: dict[str, Any], requested: str) -> str:
    if requested != "auto":
        if requested not in profiles["profiles"]:
            raise ValueError(f"unknown profile: {requested}")
        return requested

    lower = prompt.lower()
    korean = prompt
    if any(word in lower for word in ("sql", "duckdb", "schema", "table", "query")):
        return "text2sql"
    if any(word in korean for word in ("코드", "문법", "테스트", "버그", "수정", "패치")) or any(
        word in lower for word in ("code", "bug", "patch", "test", "pytest", "refactor")
    ):
        return "code_assistant"
    if any(word in lower for word in ("terminal", "shell", "bash", "tool", "command", "tmux")):
        return "terminal_tool"
    if any(word in korean for word in ("README", "파일", "실행법", "상태", "로그", "커밋", "명령", "폴더")):
        return "terminal_tool"
    if any(
        word in korean
        for word in (
            "법률",
            "판례",
            "조문",
            "변시",
            "민법",
            "형법",
            "상법",
            "행정법",
            "민사",
            "형사",
            "소송",
            "계약",
        )
    ):
        return "ko_legal"
    if any(word in korean for word in ("금융", "회계", "재무", "투자", "주식", "공시")):
        return "ko_finance"
    return "general_ko_instruction"


def build_system_prompt(profile_name: str, profiles: dict[str, Any]) -> str:
    profile = profiles["profiles"][profile_name]
    shared = profiles["shared_policy"]
    strengths = "\n".join(f"- {x}" for x in profile["strengths"])
    boundaries = "\n".join(f"- {x}" for x in profile["boundaries"])
    output_rules = "\n".join(f"- {x}" for x in profile["output_rules"])
    tools = "\n".join(f"- {x}" for x in shared["tool_contract"])
    refusal = "\n".join(f"- {x}" for x in shared["refusal_rules"])

    return textwrap.dedent(
        f"""
        You are LFM2.5-KO-SFT Agent, a bounded command-following assistant.
        Follow the user's instruction directly, but stay inside the profile.

        Active profile: {profile_name}

        Strengths:
        {strengths}

        Boundaries:
        {boundaries}

        Output rules:
        {output_rules}

        Tool contract:
        {tools}

        Refusal / redirect rules:
        {refusal}

        If a tool is needed, emit exactly one LFM-style tool call:
        <|tool_call_start|>{{"name":"read_file","arguments":{{"path":"README.md"}}}}<|tool_call_end|>

        Available tool names are: read_file, list_files, shell.
        Do not invent tool names. Do not claim that a tool ran until a tool
        result is provided. Prefer concise Korean unless the user asks otherwise.
        """
    ).strip()


def openai_chat_completion(
    cfg: HarnessConfig,
    messages: list[dict[str, str]],
) -> str:
    if cfg.mock:
        return mock_completion(messages)

    url = cfg.endpoint.rstrip("/") + "/chat/completions"
    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "max_tokens": cfg.max_tokens,
    }
    if cfg.backend == "llamacpp":
        payload["cache_prompt"] = True
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"chat completion failed: HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"chat completion failed: {exc}") from exc

    return body["choices"][0]["message"].get("content", "")


def mock_completion(messages: list[dict[str, str]]) -> str:
    user = messages[-1]["content"]
    if "README" in user or "파일" in user:
        return (
            "<|tool_call_start|>"
            '{"name":"read_file","arguments":{"path":"README.md","max_bytes":1200}}'
            "<|tool_call_end|>"
        )
    return "요청을 확인했습니다. 이 harness는 mock 모드라 실제 모델 호출 없이 형식만 검증합니다."


def clamp_text(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    head = max_chars // 2
    tail = max_chars - head
    return (
        text[:head]
        + "\n\n[... clipped by harness to stay inside context budget ...]\n\n"
        + text[-tail:],
        True,
    )


def parse_tool_call(content: str) -> dict[str, Any] | None:
    match = TOOL_CALL_RE.search(content)
    if not match:
        return None
    raw = match.group(1)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"name": "invalid_tool_call", "arguments": {"error": str(exc), "raw": raw}}
    if not isinstance(parsed, dict):
        return {"name": "invalid_tool_call", "arguments": {"error": "tool call must be object"}}
    return parsed


def run_tool(cfg: HarnessConfig, call: dict[str, Any]) -> dict[str, Any]:
    name = call.get("name")
    args = call.get("arguments") or {}
    if not isinstance(args, dict):
        return {"ok": False, "error": "arguments must be an object"}
    if name == "read_file":
        return tool_read_file(cfg, args)
    if name == "list_files":
        return tool_list_files(cfg, args)
    if name == "shell":
        return tool_shell(cfg, args)
    return {"ok": False, "error": f"unknown or disabled tool: {name}"}


def tool_read_file(cfg: HarnessConfig, args: dict[str, Any]) -> dict[str, Any]:
    try:
        path = ensure_under_workspace(cfg.workspace, args["path"])
        max_bytes = min(int(args.get("max_bytes", 12000)), cfg.prompt_budget)
        data = path.read_bytes()[:max_bytes]
        return {
            "ok": True,
            "path": str(path),
            "truncated": path.stat().st_size > max_bytes,
            "content": data.decode("utf-8", errors="replace"),
        }
    except Exception as exc:  # noqa: BLE001 - report tool errors to model.
        return {"ok": False, "error": str(exc)}


def tool_list_files(cfg: HarnessConfig, args: dict[str, Any]) -> dict[str, Any]:
    try:
        subdir = ensure_under_workspace(cfg.workspace, args.get("path", "."))
        limit = min(int(args.get("limit", 200)), 500)
        pattern = str(args.get("pattern", ""))
        files: list[str] = []
        for path in sorted(subdir.rglob("*")):
            if path.is_file():
                rel = str(path.relative_to(cfg.workspace))
                if not pattern or pattern in rel:
                    files.append(rel)
                if len(files) >= limit:
                    break
        return {"ok": True, "path": str(subdir), "files": files, "limit": limit}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def shell_allowed(command: str, allow_write: bool) -> tuple[bool, str]:
    command = command.strip()
    if not command:
        return False, "empty command"
    for pattern in DENY_SHELL_PATTERNS:
        if re.search(pattern, command):
            return False, f"denied pattern: {pattern}"
    if allow_write:
        return True, "allow_write enabled"
    if any(command == prefix or command.startswith(prefix + " ") for prefix in ALLOW_SHELL_PREFIXES):
        return True, "read-only allowlist"
    return False, "not in read-only shell allowlist"


def tool_shell(cfg: HarnessConfig, args: dict[str, Any]) -> dict[str, Any]:
    command = str(args.get("command", ""))
    if not cfg.allow_shell:
        return {"ok": False, "error": "shell tool disabled; rerun with --allow-shell"}
    allowed, reason = shell_allowed(command, cfg.allow_write)
    if not allowed:
        return {"ok": False, "error": reason, "command": command}
    if not cfg.execute_tools:
        return {"ok": False, "dry_run": True, "command": command, "reason": "rerun with --execute-tools"}
    try:
        cwd = ensure_under_workspace(cfg.workspace, args.get("cwd", "."))
        proc = subprocess.run(
            shlex.split(command),
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=int(args.get("timeout_sec", 30)),
            check=False,
        )
        output = (proc.stdout + proc.stderr)[-12000:]
        output, clipped = clamp_text(output, cfg.prompt_budget)
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "command": command,
            "cwd": str(cwd),
            "output": output,
            "clipped": clipped,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "command": command}


def run_agent(cfg: HarnessConfig, prompt: str, profiles: dict[str, Any]) -> str:
    profile_name = pick_profile(prompt, profiles, cfg.profile)
    prompt, prompt_clipped = clamp_text(prompt, cfg.prompt_budget)
    messages = [
        {"role": "system", "content": build_system_prompt(profile_name, profiles)},
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
    final = ""
    for turn in range(cfg.max_turns):
        content = openai_chat_completion(cfg, messages)
        final = content
        call = parse_tool_call(content)
        if not call:
            break
        if call.get("name") == "invalid_tool_call":
            tool_result = {"ok": False, "error": call["arguments"]}
        else:
            tool_result = run_tool(cfg, call)
        messages.append({"role": "assistant", "content": content})
        messages.append(
            {
                "role": "user",
                "content": "<tool_result>\n"
                + clamp_text(json.dumps(tool_result, ensure_ascii=False, indent=2), cfg.prompt_budget)[0]
                + "\n</tool_result>\n"
                + "Use this tool result to answer the original request. "
                + "Do not call another tool unless strictly necessary.",
            }
        )
        if turn == cfg.max_turns - 1:
            final = content + "\n\n[tool_result]\n" + json.dumps(tool_result, ensure_ascii=False, indent=2)
    return final


def run_smoke_tasks(cfg: HarnessConfig, profiles: dict[str, Any], tasks_path: Path) -> int:
    failures = 0
    with tasks_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            task = json.loads(line)
            prompt = task["prompt"]
            expected_profile = task.get("expected_profile")
            picked = pick_profile(prompt, profiles, cfg.profile)
            print(f"task={idx} id={task.get('id')} profile={picked}")
            if expected_profile and picked != expected_profile:
                print(f"  profile_mismatch expected={expected_profile} actual={picked}")
                failures += 1
            answer = run_agent(cfg, prompt, profiles)
            print(textwrap.indent(answer.strip(), "  ")[:4000])
    return failures


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LFM2.5 KO SFT bounded agent harness")
    parser.add_argument("prompt", nargs="*", help="user instruction")
    parser.add_argument("--backend", choices=("vllm", "llamacpp", "openai"), default=os.getenv("AGENT_BACKEND", "vllm"))
    parser.add_argument("--endpoint", default=os.getenv("OPENAI_BASE_URL", "http://localhost:1053/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "lfm2-ko-sft"))
    parser.add_argument("--workspace", default=str(ROOT))
    parser.add_argument("--profiles", default=str(DEFAULT_PROFILES))
    parser.add_argument("--profile", default="auto")
    parser.add_argument("--context-window", type=int, default=int(os.getenv("AGENT_CONTEXT_WINDOW", "8192")))
    parser.add_argument("--prompt-budget", type=int, default=int(os.getenv("AGENT_PROMPT_BUDGET_CHARS", "24000")))
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--execute-tools", action="store_true")
    parser.add_argument("--allow-shell", action="store_true")
    parser.add_argument("--allow-write", action="store_true")
    parser.add_argument("--mock", action="store_true", help="do not call model; test harness flow only")
    parser.add_argument("--smoke", action="store_true", help="run bundled smoke tasks")
    parser.add_argument("--smoke-tasks", default=str(DEFAULT_SMOKE_TASKS))
    parser.add_argument("--print-system-prompt", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    profiles = load_profiles(Path(args.profiles))
    workspace = normalize_workspace(args.workspace)
    cfg = HarnessConfig(
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

    prompt = " ".join(args.prompt).strip()
    if args.print_system_prompt:
        profile = pick_profile(prompt or "한국어 일반 지시", profiles, args.profile)
        print(build_system_prompt(profile, profiles))
        return 0
    if args.smoke:
        return run_smoke_tasks(cfg, profiles, Path(args.smoke_tasks))
    if not prompt:
        print("prompt is required unless --smoke or --print-system-prompt is used", file=sys.stderr)
        return 2
    start = time.time()
    answer = run_agent(cfg, prompt, profiles)
    print(answer)
    print(f"\n[harness] elapsed_sec={time.time() - start:.2f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
