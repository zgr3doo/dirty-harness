# tools.py — narzedzia agenta. run_tests to WERYFIKATOR (serce harnessu).
import sys
import subprocess
from pathlib import Path
from langchain.tools import tool, ToolRuntime
from dataclasses import dataclass
from uuid import uuid4
from deepagents.backends import FilesystemBackend

import os
import tempfile

WORK = Path("workspace")

@dataclass
class Ctx:
    user_id: str

workspace = Path("/").resolve()

backend = FilesystemBackend(
    root_dir=str(workspace),
    virtual_mode=True,
)

@tool
def read_file(
    file_path: str,
    offset: int = 0,
    limit: int = 100,
) -> str:
    """Odczytaj plik tekstowy.

    Args:
        file_path: Ścieżka do pliku, np. /example.txt.
        offset: Numer linii, od której zacząć (od 0).
        limit: Maksymalna liczba linii do odczytania.
    """
    if offset < 0:
        return "Błąd: offset nie może być ujemny."

    if limit < 1:
        return "Błąd: limit musi być większy od 0."

    result = backend.read(
        file_path=file_path,
        offset=offset,
        limit=limit,
    )

    if result.error:
        return f"Błąd odczytu pliku: {result.error}"

    if result.file_data is None:
        return "Brak zawartości pliku."

    return result.file_data["content"]

@tool
def write_file(file_path: str, content: str) -> str:
    """Zapisz tekst do pliku.

    Args:
        file_path: Ścieżka pliku, np. /example.txt.
        content: Treść do zapisania.
    """
    result = backend.write(
        file_path=file_path,
        content=content,
    )

    if result.error:
        return f"Błąd zapisu: {result.error}"

    return f"Plik zapisany: {result.path}"


@tool
def create_file(file_path: str) -> str:
    """Utwórz nowy, pusty plik. Nie nadpisuj istniejących plików."""

    # Sprawdź, czy plik już istnieje
    result = backend.read(file_path)

    if result.file_data is not None:
        return f"Plik już istnieje: {file_path}"

    # Utwórz pusty plik
    result = backend.write(
        file_path=file_path,
        content="",
    )

    if result.error:
        return f"Błąd tworzenia pliku: {result.error}"

    return f"Utworzono plik: {file_path}"

@tool
def ls(path: str = ".") -> str:
    """List files and directories in the available workspace."""
    try:
        result = backend.ls(path)
    except ValueError as exc:
        return f"Error listing directory: {exc}"

    if result.error:
        return f"Error: {result.error}"

    return "\n".join(
        f"{'DIR ' if item.get('is_dir') else 'FILE'} {item['path']}"
        for item in result.entries
    )

@tool
def glob(pattern: str, path: str = "/") -> str:
    """Find files matching a glob pattern under the specified path.

    Args:
        pattern: Glob pattern, e.g. '**/*.py' or '*.txt'.
        path: Base directory to search from. Defaults to '/'.
    """
    result = backend.glob(pattern=pattern, path=path)

    if result.error:
        return f"Error: {result.error}"

    if not result.matches:
        return "No files found."

    return "\n".join(
        item["path"] for item in result.matches
    )

@tool
def run_pytest(
    path: str = "tests",
    extra_args: list[str] | None = None,
) -> str:
    """
    Uruchamia testy pytest w określonym katalogu lub dla wskazanego pliku.

    Args:
        path: Ścieżka do katalogu z testami lub pliku testowego.
        extra_args: Opcjonalne argumenty pytest, np. ["-v", "-x"].

    Returns:
        Raport z wykonania testów, zawierający status i output.
    """
    project_root = Path.cwd().resolve()
    target = (project_root / path).resolve()

    # Nie pozwalaj uruchamiać testów poza katalogiem projektu.
    if not target.is_relative_to(project_root):
        return "Błąd: ścieżka do testów musi znajdować się w projekcie."

    if not target.exists():
        return f"Błąd: ścieżka nie istnieje: {path}"

    args = extra_args or []

    # Ogranicz argumenty do bezpiecznego zestawu.
    allowed_flags = {"-v", "-vv", "-q", "-x", "--tb=short", "--tb=long"}
    if any(arg not in allowed_flags for arg in args):
        return "Błąd: niedozwolone argumenty pytest."

    try:
        result = subprocess.run([sys.executable, "-m", "pytest", str(target), *args,],
            cwd=project_root,
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
        return "TIMEOUT: testy przekroczyły limit 120 sekund."
    except OSError as exc:
        return f"Błąd uruchamiania pytest: {exc}"

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
def remember(text: str, runtime: ToolRuntime[Ctx]) -> str:
    """Zapisz TRWAŁY fakt o użytkowniku/projekcie. Nie zapisuj PII ani sekretów."""
    ns = (runtime.context.user_id, "facts")
    runtime.store.put(ns, str(uuid4()), {"text": text})
    return "Zapamiętane."

@tool
def recall(query: str, runtime: ToolRuntime[Ctx]) -> str:
    """Przeszukaj pamięć długoterminową (semantic search po treści)."""
    ns = (runtime.context.user_id, "facts")
    hits = runtime.store.search(ns, query=query, limit=3)
    if not hits:
        return "Brak wspomnień na ten temat."
    return "\n".join(f"- {h.value['text']}" for h in hits)


def safe_path(path: str) -> Path:
    path = Path(path)

    # Jeśli podano już root, nie dokładaj go drugi raz
    if path == WORK or WORK in path.parents:
        return path

    return WORK / path