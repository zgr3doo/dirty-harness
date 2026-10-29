import sys, json
import subprocess
from pathlib import Path
from langchain.tools import tool, ToolRuntime
from dataclasses import dataclass
from uuid import uuid4
from deepagents.backends import LocalShellBackend, FilesystemBackend
from langchain_community.tools import ReadFileTool
from code_reader_agent import CodeReadingAgent
from llm_provider import connect_to_llm
import os
import tempfile

ALLOWED_DIR = Path("/workspace").resolve()

backend = FilesystemBackend(
    root_dir=str(ALLOWED_DIR),
    virtual_mode=True,
)

llm = connect_to_llm()

@tool
def read_file(
    file_path: str,
    offset: int = 0,
    limit: int = 100,
) -> str:
    """Odczytaj plik tekstowy.

    Args:
        file_path: File path, eg. /example.txt.
        offset: Line number, from which to start reading (from 0).
        limit: Max line count to be read.
    """
    if offset < 0:
        return "Error: offset must be positive"

    if limit < 1:
        return "Error: limit must be bigger than 0"

    result = backend.read(
        file_path=file_path,
        offset=offset,
        limit=limit,
    )

    if result.error:
        return f"Error: {result.error}"

    if result.file_data is None:
        return "File is empty"

    return result.file_data["content"]

@tool
def write_file(file_path: str, content: str) -> str:
    """
    Save the text to a file
    Args:
        file_path: File path, e.g., /report.txt.
        content: Content to save.
    """
    result = backend.write(
        file_path="/" + file_path,
        content=content,
    )

    if result.error:
        return f"Error: {result.error}"

    return f"File saved: {result.path}"

@tool
def create_file(file_path: str) -> str:
    """
    Create a new, empty file.
    Do not overwrite existing files.
    """

    result = backend.read(str(file_path))

    if result.file_data is not None:
        return f"Error: file exists - {file_path}"

    result = backend.write(
        file_path=str(file_path),
        content="",
    )

    if result.error:
        return f"Error: {result.error}"

    return f"File created: {result.path}"

@tool
def ls(path: str = ".") -> str:
    """List files and directories in the available workspace."""
    try:
        result = backend.ls(path)
    except ValueError as exc:
        if "Path traversal not allowed" in str(exc):
            return f"Invalid path: {path}"
        raise

    if result.error:
        return f"Error: {result.error}"

    return "\n".join(
        f"{'DIR ' if item.get('is_dir', False) else 'FILE'} {item.get('path', '<unknown>')}"
        for item in result.entries
    )

@tool
def glob(pattern: str, path: str = "/") -> str:
    """Find files matching a glob pattern under the allowed directory.

    Args:
        pattern: Glob pattern, e.g. '**/*.py' or '*.txt'.
        path: Base directory to search from. Must be inside ALLOWED_DIR.
    """
    try:
        result = backend.glob(
            pattern=pattern,
            path=str(path),
        )

        if result.error:
            return f"Error: {result.error}"

        if not result.matches:
            return "No files found."

        return "\n".join(
            match if isinstance(match, str)
            else json.dumps(match, ensure_ascii=False)
            for match in result.matches
        )

    except ValueError as exc:
        if "Path traversal not allowed" in str(exc):
            return f"Error: invalid path - {path}"
        raise

@tool
def run_pytest(
    path: str = "tests",
    extra_args: list[str] | None = None,
) -> str:
    """
    Uruchamia testy pytest w określonym katalogu lub dla wskazanego pliku.

    Args:
        path: Ścieżka do katalogu z testami lub pliku testowego.
        extra_args: Opcjonalne argumenty pytest, allowed are ["-v", "-vv", "-q", "-x", "--tb=short", "--tb=long"]. ["-v", "-x"].

    Returns:
        Raport z wykonania testów, zawierający status i output.
    """

    try:
        result = backend.read(str(path))

        if result.file_data is None:
            target = Path(backend.cwd) / path
            # return f"Error: test file not exists - {path}"
        else:
            target = Path(backend.cwd) / path.lstrip("/")

    except ValueError as exc:
        if "Path traversal not allowed" in str(exc):
            return f"Invalid path: {path}"
        raise

    args = extra_args or []

    # Ogranicz argumenty do bezpiecznego zestawu.
    allowed_flags = {"-v", "-vv", "-q", "-x", "--tb=short", "--tb=long"}
    if any(arg not in allowed_flags for arg in args):
        return "Error: not allowed arguments."

    try:
        result = subprocess.run([sys.executable, "-m", "pytest", str(target), *args,],
            cwd=Path(backend.cwd),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

        status = (
            "PASSED"
            if result.returncode == 0
            else "FAILED"
        )

        return (
            f"Status: {status}\n"
            f"Exit code: {result.returncode}\n\n"
            f"STDOUT:\n{result.stdout[-12000:]}\n\n"
            f"STDERR:\n{result.stderr[-6000:]}"
        )

    except subprocess.TimeoutExpired:
        return "TIMEOUT: tests exceeded the 120-second time limit"
    except OSError as exc:
        return f"Error: running pytest - {exc}"

@tool
def validate_python_file(path: str) -> str:
    """Waliduje plik Python za pomocą Ruff i zwraca wykryte problemy.

    Args:
        file_path: Ścieżka względna do pliku, np. /src/main.py.
    """

    try:
        result = backend.read(str(path))

        if result.file_data is None:
            return f"Error: test file not exists - {path}"

        target = Path(backend.cwd) / path.lstrip("/")

        result = subprocess.run(
            [
                "ruff",
                "check",
                "--output-format=concise",
                "--stdin-filename",
                target,
                "-",
            ],
            input=result.file_data["content"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            cwd=Path(backend.cwd),
        )

    except ValueError as exc:
        if "Path traversal not allowed" in str(exc):
            return f"Invalid path: {path}"
        raise

    except FileNotFoundError:
        return "Error: file does not exist or Ruff is not available"
    except subprocess.TimeoutExpired:
        return "Error: Ruff exceeded the 30-second time limit"
    except Exception as exc:
        return f"Error: {exc}"

    if result.returncode == 0:
        return f"Success: no problems detected in file {path}."

    if result.returncode == 1:
        return (
            f"Results for  {path}:\n"
            f"{result.stdout or 'Issues found'}"
        )

    return f"Error:\n{result.stderr or result.stdout}"

@tool
def validate_puml(code: str) -> str:
    """Validate PlantUML code syntax.

    Use this tool to check whether a PlantUML diagram has valid syntax.
    Pass the complete PlantUML source code as the code argument.
    Returns validation status and diagnostic output to help fix errors.
    """
    if not code or not code.strip():
        return "INVALID: PlantUML code is empty."

    jar_path = Path(
        os.getenv("PLANTUML_JAR", "/opt/plantuml.jar")
    )

    if not jar_path.is_file():
        return f"ERROR: PlantUML JAR not found: {jar_path}"

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            puml_file = Path(tmpdir) / "diagram.puml"
            puml_file.write_text(code, encoding="utf-8")

            result = subprocess.run(
                [
                    "java",
                    "-Djava.awt.headless=true",
                    "-jar",
                    str(jar_path),
                    "-checkonly",
                    str(puml_file),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )

            output = "\n".join(
                part.strip()
                for part in (result.stdout, result.stderr)
                if part.strip()
            )

            if result.returncode != 0:
                return (
                    "INVALID: PlantUML validation failed.\n"
                    f"Exit code: {result.returncode}\n"
                    f"Diagnostics:\n{output or 'No diagnostics returned.'}"
                )

            return (
                "VALID: PlantUML syntax check passed.\n"
                "Note: This confirms syntax validation only, "
                "not the semantic correctness of the diagram."
            )

    except subprocess.TimeoutExpired:
        return "ERROR: PlantUML validation timed out after 30 seconds."
    except OSError as exc:
        return f"ERROR: Could not execute PlantUML: {exc}"


@tool
def analyze_repository(question: str) -> dict:
    """
    Analyzes the repository code and returns findings supported by the source code
    """

    code_agent = CodeReadingAgent(llm)

    result = code_agent.graph.invoke({
        "repo_path": "/",
        "question": question,
        "files": [],
        "findings": [],
        "errors": [],
    })

    return {
        "summary": result.get("summary", ""),
        "findings": result.get("findings", []),
        "limitations": result.get("limitations", []),
        "errors": result.get("errors", []),
    }

def resolve_project_path(path: str) -> Path:
    """
    Paths provided by LLM are relative to ALLOWED_DIR
    use this method only for tools which do not use FilestystemBackend
    :param path: path assumed virtual (virtual_mode=True)
    :return: path prepended with ALLOWED_DIR
    """

    relative = Path(path.lstrip("/"))
    target = (ALLOWED_DIR / relative).resolve()

    if not target.is_relative_to(ALLOWED_DIR):
        raise ValueError("Path is outside of project directory")

    return target