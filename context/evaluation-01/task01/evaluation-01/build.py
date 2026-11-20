"""Build the architecture report evaluation-01.html.

Run from the repository root:

    python3 context/evaluation-01/task01/evaluation-01/build.py

Steps:
1. Read every module under src/digital_twin_engineering/ with `ast`.
   For each module, keep its line count, the first line of its docstring and its imports
   from this package. Write this data to modules.json (next to this script).
2. Make the module table and the "imported by" column from that data.
3. Put the CSS, the JavaScript, the marker file and the module table into template.html.
   Write the result to ../evaluation-01.html (one self-contained file).

The script uses only the Python standard library.
"""

import ast
import datetime
import html
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent              # .../task01/evaluation-01
TASK = HERE.parent                                   # .../task01
REPO = HERE.parents[3]                               # repository root
PACKAGE = REPO / "src" / "digital_twin_engineering"

TEMPLATE = HERE / "template.html"
MARKERS = TASK / "evaluation-01-markers.json"
MODULES_JSON = HERE / "modules.json"
OUTPUT = TASK / "evaluation-01.html"


# ---------------------------------------------------------------- 1. module data
def module_name(path: Path) -> str:
    """src/digital_twin_engineering/experts/pas.py -> experts.pas"""
    rel = path.relative_to(PACKAGE).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) or "(package)"


def internal_imports(tree: ast.Module, name: str, is_package: bool) -> list[str]:
    """The modules of this package that a module imports.
    The package uses only relative imports (from . import x, from ..x import y)."""
    found = set()
    own = name.split(".") if name != "(package)" else []
    here = own if is_package else own[:-1]           # the package that holds the module
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ImportFrom) and node.level):
            continue
        base = here[: len(here) - (node.level - 1)]
        if node.module:                              # from .x import y: the module is x,
            target = base + node.module.split(".")   # or x.y if y is a module (from .live import replayer)
            found.add(".".join(target))
            for alias in node.names:
                found.add(".".join(target + [alias.name]))
        else:                                        # from . import a, b: each name is a module
            for alias in node.names:
                found.add(".".join(base + [alias.name]))
    return sorted(found)


def collect() -> list[dict]:
    known = {module_name(p) for p in PACKAGE.rglob("*.py")}
    rows = []
    for path in sorted(PACKAGE.rglob("*.py")):
        text = path.read_text()
        tree = ast.parse(text)
        name = module_name(path)
        doc = (ast.get_docstring(tree) or "").strip().splitlines()
        imports = internal_imports(tree, name, path.name == "__init__.py")
        rows.append({
            "module": name,
            "path": str(path.relative_to(REPO)),
            "lines": len(text.splitlines()),
            "doc": doc[0] if doc else "",
            "imports": [m for m in imports if m in known],
        })
    for row in rows:
        row["imported_by"] = sorted(r["module"] for r in rows if row["module"] in r["imports"])
    return rows


# ---------------------------------------------------------------- 2. module table
def module_table(rows: list[dict]) -> str:
    def code_list(items):
        return ", ".join(f"<code>{html.escape(i)}</code>" for i in items) or "–"

    out = ['<table class="modtab">',
           "<thead><tr><th>Module</th><th>Lines</th><th>First line of the docstring</th>"
           "<th>Imports (this package)</th><th>Imported by</th></tr></thead>", "<tbody>"]
    for r in rows:
        if r["lines"] == 0:
            continue                                 # empty __init__.py
        out.append(
            f"<tr><td><code>{html.escape(r['module'])}</code></td><td class=\"num\">{r['lines']}</td>"
            f"<td>{html.escape(r['doc'])}</td><td>{code_list(r['imports'])}</td>"
            f"<td>{code_list(r['imported_by'])}</td></tr>"
        )
    out += ["</tbody>", "</table>"]
    return "\n".join(out)


# ---------------------------------------------------------------- 3. inline everything
def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    rows = collect()
    MODULES_JSON.write_text(json.dumps(rows, indent=2) + "\n")

    inline = {name: (HERE / name).read_text().rstrip("\n") for name in ("style.css", "marks.css", "marks.js")}
    inline["markers.json"] = MARKERS.read_text().rstrip("\n").replace("</", "<\\/")

    out = TEMPLATE.read_text()
    out = out.replace("{{MODULE_TABLE}}", module_table(rows))
    out = out.replace("{{SOURCE_COMMIT}}", git_commit())
    out = out.replace("{{BUILD_DATE}}", datetime.date.today().isoformat())
    out = out.replace("{{TOTAL_LINES}}", str(sum(r["lines"] for r in rows)))
    out = re.sub(r"\{\{INLINE:([\w.]+)\}\}", lambda m: inline[m.group(1)], out)
    OUTPUT.write_text(out)
    print(f"{len(rows)} modules -> {MODULES_JSON.relative_to(REPO)}")
    print(f"report -> {OUTPUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
