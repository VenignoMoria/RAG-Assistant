"""Проверка AST без импортов проекта; YAML проверяется только при наличии PyYAML."""
import ast
import json
from pathlib import Path
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    excluded = {".venv", ".git", "work", "outputs", "artifacts"}
    paths = [p for p in root.rglob("*") if p.is_file() and not excluded.intersection(p.relative_to(root).parts)]
    errors = []
    python_count = yaml_count = json_count = 0
    for path in paths:
        try:
            if path.suffix == ".py":
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
                python_count += 1
            elif path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
                json_count += 1
            elif path.suffix == ".jsonl":
                for line in path.read_text(encoding="utf-8").splitlines():
                    json.loads(line)
                    json_count += 1
        except (SyntaxError, ValueError) as exc:
            errors.append(f"{path.relative_to(root)}: {exc}")
    try:
        import yaml
    except ImportError:
        print("YAML: skipped (PyYAML not installed)")
    else:
        for path in paths:
            if path.suffix in {".yaml", ".yml"}:
                try:
                    list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
                    yaml_count += 1
                except yaml.YAMLError as exc:
                    errors.append(f"{path.relative_to(root)}: {exc}")
        print(f"YAML syntax: {yaml_count} files")
    print(f"Python AST: {python_count} files; JSON/JSONL: {json_count} objects")
    if errors:
        print("\n".join(errors))
        sys.exit(1)
    print("Syntax checks only; runtime tests were not executed by this script.")


if __name__ == "__main__":
    main()
