import xml.etree.ElementTree as ET
from pathlib import Path


def api_link(path: str) -> str:
    """
    src/mcframework/core.py -> :mod:`mcframework.core`
    src/mcframework/__init__.py -> :mod:`mcframework`
    """
    mod = path.removeprefix("src/").removesuffix(".py").replace("/", ".")
    if mod.endswith(".__init__"):
        mod = mod[: -len(".__init__")]
    
    return f":mod:`{mod}`"


root = ET.parse("coverage.xml").getroot()
packages = root.find("packages")

rows = []
tot_lines = tot_missed = 0

for pkg in packages.findall("package"):
    for cls in pkg.find("classes").findall("class"):
        filename = cls.get("filename")                     # e.g. src/mcframework/core.py
        lines = cls.find("lines").findall("line")
        n_lines = len(lines)
        missed_nums = [int(ln.get("number")) for ln in lines if ln.get("hits") == "0"]
        n_missed = len(missed_nums)
        coverage = 0.0 if n_lines == 0 else 100.0 * (n_lines - n_missed) / n_lines

        rows.append({
            "modlink": api_link(filename),
            "path": filename,
            "lines": n_lines,
            "missed": n_missed,
            "missed_nums": missed_nums,
            "coverage": coverage,
        })
        tot_lines += n_lines
        tot_missed += n_missed

rows.sort(key=lambda r: r["coverage"])  # lowest coverage first
total_cov = 0.0 if tot_lines == 0 else 100.0 * (tot_lines - tot_missed) / tot_lines

# --- Build an rst list-table (roles render as links) ---
table_rows = "\n".join(
    [
        f"   * - {r['modlink']}\n"
        f"     - {r['lines']}\n"
        f"     - {r['missed']}\n"
        f"     - {r['coverage']:.1f}%"
        for r in rows
    ]
)

# Optional per-file detail sections showing missed line numbers
details = "\n\n".join(
    [
        (
            f".. rubric:: {r['path']}\n\n"
            + (
                f"Missed lines: ``{', '.join(map(str, r['missed_nums']))}``"
                if r["missed_nums"]
                else "No missed lines."
            )
        )
        for r in rows
    ]
)

rst = f"""\
Test Coverage
=============

**Total:** {total_cov:.1f}%

.. list-table::
   :header-rows: 1
   :widths: 40 10 10 10

   * - File
     - Lines
     - Missed
     - Coverage
{table_rows}

{details}
"""

out = Path("docs/source/coverage_summary.rst")
out.write_text(rst, encoding="utf-8")
print(f"Wrote {out}")
