
from pathlib import Path, PurePosixPath
from typing import TypedDict
import os

from deepagents.backends import FilesystemBackend

IGNORED_DIRS = {
    ".git", ".venv", "venv", "__pycache__",
    "node_modules", "dist", "build", ".next",
}

ALLOWED_DIR = Path("/workspace").resolve()

backend = FilesystemBackend(
    root_dir=str(ALLOWED_DIR),
    virtual_mode=True,
)

MAX_FILE_BYTES = 300_000

def list_source_files(
    repo_path: str,
) -> list[str]:
    root = PurePosixPath(repo_path)

    ignored_dirs = set(IGNORED_DIRS)
    extensions = {
        ".py", ".js", ".ts", ".tsx", ".jsx",
        ".java", ".go", ".rs", ".cs", ".cpp",
        ".c", ".h", ".php", ".rb", ".sql",
    }

    result: list[str] = []
    stack = [root.as_posix()]

    while stack:
        current = stack.pop()
        listing = backend.ls(current)

        if listing.error:
            if current == root.as_posix():
                raise ValueError(
                    f"Invalid repository directory: {repo_path}"
                )
            continue

        for entry in listing.entries:
            path = PurePosixPath(entry["path"])
            relative_path = path.relative_to(root)

            if entry["is_dir"]:
                if path.name not in ignored_dirs:
                    stack.append(path.as_posix())
                continue

            if path.suffix.lower() not in extensions:
                continue

            size = entry.get("size")

            if size is not None and size <= MAX_FILE_BYTES:
                result.append(relative_path.as_posix())

    return sorted(result)

def read_source_file(repo_path: str, relative_path: str) -> str:
    root = Path(repo_path).resolve()
    target = (root / relative_path).resolve()

    # Ochrona przed wyjściem poza katalog repozytorium.
    if not target.is_relative_to(root):
        raise ValueError("Path escapes repository root")

    if target.is_symlink() or not target.is_file():
        raise ValueError("Not a regular source file")

    if target.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("File exceeds configured size limit")

    return target.read_text(encoding="utf-8", errors="replace")

def read_source_file_new(path: str) -> str:
    result = backend.read(str(path))

    if result.file_data is None:
        return f"Error: test file not exists - {path}"
    else:
        target = Path(backend.cwd) / path.lstrip("/")

    if target.is_symlink() or not target.is_file():
        raise ValueError("Not a regular source file")

    if target.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("File exceeds configured size limit")

    return result.file_data['content']


def extract_lines(
    repo_path: str,
    relative_path: str,
    start_line: int,
    end_line: int,
    filesystem_backend: FilesystemBackend,
) -> dict:
    if start_line < 1 or end_line < start_line:
        raise ValueError("Invalid line range")

    path = f"{repo_path.rstrip('/')}/{relative_path.lstrip('/')}"

    result = filesystem_backend.read(
        path,
        offset=start_line - 1,
        limit=end_line - start_line + 1,
    )

    if result.error:
        raise ValueError(f"Could not read file: {path}: {result.error}")

    lines = result.file_data['content'].splitlines()
    expected_lines = end_line - start_line + 1

    if len(lines) < expected_lines:
        raise ValueError("Line range exceeds file length")

    return {
        "file": relative_path,
        "start_line": start_line,
        "end_line": end_line,
        "code": "\n".join(lines[:expected_lines]),
    }


def extract_lines_new(
    file_path: str,
    start_line: int,
    end_line: int,
) -> dict:

    if start_line < 1 or end_line < start_line:
        raise ValueError("Invalid line range")

    result = backend.read(
        file_path=file_path,
        offset=start_line - 1,
        limit=end_line - start_line + 1,
    )

    if result.error:
        raise ValueError(f"Could not read file: {file_path}: {result.error}")

    lines = result.file_data['content'].splitlines()
    expected_lines = end_line - start_line + 1

    if len(lines) < expected_lines:
        raise ValueError("Line range exceeds file length")

    return {
        "file": file_path,
        "start_line": start_line,
        "end_line": end_line,
        "code": "\n".join(lines[:expected_lines]),
    }


class Finding(TypedDict):
    title: str
    file: str
    start_line: int
    end_line: int
    code: str
    explanation: str
    confidence: float


class AnalysisState(TypedDict, total=False):
    question: str
    repo_path: str
    files: list[str]
    findings: list[Finding]
    summary: str
    limitations: list[str]
    errors: list[str]
