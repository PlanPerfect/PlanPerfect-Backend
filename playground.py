# this is just to test the agent system. DO NOT use in prod.

import argparse
import asyncio
import json
import mimetypes
import os
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from Services import DatabaseManager as DM
from Services import FileManager as FM
from Services import LLMManager as LLM
from Services import ServiceOrchestra as SO
from Services import AgentSynthesizer as AGS

class LocalUploadFile:
    def __init__(self, path: str):
        resolved_path = Path(path).expanduser().resolve()

        if not resolved_path.exists():
            raise FileNotFoundError(f"File not found: {resolved_path}")
        if not resolved_path.is_file():
            raise ValueError(f"Not a file: {resolved_path}")

        self.path = resolved_path
        self.filename = resolved_path.name
        self.content_type = (
            mimetypes.guess_type(str(resolved_path))[0] or "application/octet-stream"
        )
        self._content = resolved_path.read_bytes()

    async def read(self) -> bytes:
        return self._content


@dataclass
class PlaygroundState:
    user_id: str
    max_iterations: int
    last_session_id: Optional[str] = None
    files: Dict[str, LocalUploadFile] = field(default_factory=dict)


def _compact(value: Any, max_len: int = 200) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if len(text) <= max_len:
        return text
    return f"{text[:max_len].rstrip()}..."


def _tool_step_label(tool_name: Optional[str]) -> Optional[str]:
    mapping = {
        "generate_image": "Generating Image...",
        "classify_style": "Classifying Style...",
        "detect_furniture": "Detecting Furniture...",
        "get_recommendations": "Getting Recommendations...",
        "web_search": "Searching the web...",
        "extract_colors": "Extracting Colors...",
        "generate_floor_plan": "Generating Floor Plan...",
    }
    if not isinstance(tool_name, str):
        return None
    return mapping.get(tool_name)


def _steps_for_latest_query(session: Dict[str, Any], status: str) -> list[str]:
    steps = session.get("steps", [])
    if not isinstance(steps, list) or not steps:
        return []

    last_user_query_index = 0
    for idx, step in enumerate(steps):
        if isinstance(step, dict) and step.get("type") == "user_query":
            last_user_query_index = idx

    query_steps = steps[last_user_query_index:]
    labels: list[str] = []

    def append_unique(label: Optional[str]) -> None:
        if not label:
            return
        if labels and labels[-1] == label:
            return
        labels.append(label)

    for step in query_steps:
        if not isinstance(step, dict):
            continue

        step_type = step.get("type")
        if step_type == "thought":
            append_unique("Thinking...")
        elif step_type == "file_analysis":
            append_unique("Analysing files...")
        elif step_type == "file_confirmation_requested":
            append_unique("Waiting for file confirmation...")
        elif step_type == "tool_call":
            append_unique(_tool_step_label(step.get("tool")))

    if status == "completed":
        append_unique("Done!")

    return labels

def _initialize_services() -> None:
    required_env = [
        "FIREBASE_DATABASE_URL",
        "FIREBASE_CREDENTIALS_PATH",
        "GROQ_API_KEY",
        "GEMINI_API_KEY",
        "CLOUDINARY_URL",
    ]

    missing = [name for name in required_env if not os.getenv(name)]
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables: {missing_text}")

    if not DM._initialized:
        DM.initialize(
            database_url=os.getenv("FIREBASE_DATABASE_URL"),
            credentials_path=os.getenv("FIREBASE_CREDENTIALS_PATH"),
        )

    if not LLM._initialized:
        LLM.initialize()

    if not FM._initialized:
        FM.initialize()

    if not SO._initialized:
        SO.initialize()

    if not AGS._initialized:
        AGS.initialize()


def _register_file(state: PlaygroundState, file_id: str, file_path: str) -> None:
    upload_file = LocalUploadFile(file_path)
    state.files[file_id] = upload_file
    AGS.register_file(
        file_id=file_id,
        file_obj=upload_file,
        user_id=state.user_id,
        source="playground",
    )

    print(f"Registered file_id='{file_id}' -> {upload_file.path}")


def _list_files(state: PlaygroundState) -> None:
    if not state.files:
        print("No files registered.")
        return

    print("Registered files:")
    for file_id, upload_file in state.files.items():
        print(f"  {file_id}: {upload_file.path}")


def _remove_file(state: PlaygroundState, file_id: str) -> None:
    if file_id not in state.files:
        print(f"File ID not found: {file_id}")
        return

    del state.files[file_id]
    AGS.unregister_file(file_id=file_id, user_id=state.user_id)

    print(f"Removed file_id='{file_id}'.")


def _clear_files(state: PlaygroundState) -> None:
    state.files.clear()
    AGS.clear_file_registry(user_id=state.user_id)
    print("Cleared all registered files.")

async def _execute_query(state: PlaygroundState, query: str) -> None:
    print("\nRunning agent...")
    result = await AGS.execute(
        user_id=state.user_id,
        query=query,
        max_iterations=state.max_iterations,
    )

    state.last_session_id = result.get("session_id")
    session = AGS.get_session(
        user_id=state.user_id,
        session_id=state.last_session_id,
    ) if state.last_session_id else None

    print("\nAssistant:")
    print(result.get("response", ""))

    if session:
        step_labels = _steps_for_latest_query(session, result.get("status", ""))
        if step_labels:
            print("\nSteps Taken:")
            for step in step_labels:
                print(f"  - {step}")

    if result.get("status") == "error":
        print(f"Error: {result.get('error', 'Unknown error')}")


async def _handle_command(state: PlaygroundState, raw: str) -> bool:
    try:
        parts = shlex.split(raw)
    except ValueError as exc:
        print(f"Invalid command syntax: {exc}")
        return True

    if not parts:
        return True

    cmd = parts[0].lower()

    if cmd in {"/quit", "/exit"}:
        return False

    if cmd == "/files":
        if len(parts) < 2:
            print("Usage: /files <add|list|remove|clear> ...")
            return True

        sub = parts[1].lower()
        if sub == "add":
            if len(parts) != 4:
                print("Usage: /files add <file_id> <path>")
                return True
            _register_file(state, parts[2], parts[3])
            return True

        if sub == "list":
            _list_files(state)
            return True

        if sub == "remove":
            if len(parts) != 3:
                print("Usage: /files remove <file_id>")
                return True
            _remove_file(state, parts[2])
            return True

        if sub == "clear":
            _clear_files(state)
            return True

        print("Unknown /files command. Use add, list, remove, or clear.")
        return True

    print("Unknown command.")
    return True


def _parse_file_registration(raw_value: str) -> tuple[str, str]:
    if "=" not in raw_value:
        raise ValueError("File registration must be formatted as <file_id>=<path>")

    file_id, file_path = raw_value.split("=", 1)
    file_id = file_id.strip()
    file_path = file_path.strip()

    if not file_id or not file_path:
        raise ValueError("File registration must be formatted as <file_id>=<path>")

    return file_id, file_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Console playground for testing AgentSynthesizer + ServiceOrchestra."
    )
    parser.add_argument(
        "--user-id",
        default=os.getenv("PLAYGROUND_USER_ID", "playground-user"),
        help="User ID used for session tracking in Firebase.",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    _initialize_services()

    state = PlaygroundState(
        user_id=args.user_id,
        max_iterations=args.max_iterations,
    )

    for file_spec in args.file:
        file_id, file_path = _parse_file_registration(file_spec)
        _register_file(state, file_id, file_path)

    if args.query:
        await _execute_query(state, args.query)
        return 0

    print("Agent Playground Ready.")
    print(f"User ID: {state.user_id}")

    while True:
        try:
            raw = input("agent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return 0

        if not raw:
            continue

        if raw.startswith("/"):
            try:
                should_continue = await _handle_command(state, raw)
            except Exception as exc:
                print(f"Command failed: {exc}")
                continue

            if not should_continue:
                print("Exiting.")
                return 0
            continue

        try:
            await _execute_query(state, raw)
        except Exception as exc:
            print(f"Query failed: {exc}")


def main() -> int:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args()

    try:
        return asyncio.run(_run(args))
    except Exception as exc:
        print(f"Playground failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
