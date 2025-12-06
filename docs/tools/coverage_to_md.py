import xml.etree.ElementTree as ET
from pathlib import Path


def api_link(path: str) -> str:
    mod = path.removeprefix("src/").removesuffix(".py").replace("/", ".")
    return f":module: `{mod}`"

xml = ET.parse("coverage.xml").getroot()
packages = xml.find("packages")
rows = []
tot_lines = tot_missed = 0

for pkg in packages.findall("package"):
    for cls in pkg.find("classes").findall("class"):
        filename = api_link(cls.get("filename"))
        lines = cls.find("lines").findall("line")
        n_lines = len(lines)
        n_missed = sum(1 for ln in lines if ln.get("hits") == "0")
        coverage = 0.0 if n_lines == 0 else 100.0 * (n_lines - n_missed) / n_lines
        rows.append((filename, n_lines, n_missed, coverage))
        tot_lines += n_lines
        tot_missed += n_missed

rows.sort(key=lambda r: r[3])  # lowest coverage first
total_cov = 0.0 if tot_lines == 0 else 100.0 * (tot_lines - tot_missed) / tot_lines

out = Path("docs/source/coverage_summary.md")
out.write_text(
    "# Test Coverage \n\n"
    f"**Total**: {total_cov:.1f}%  \n\n"
    "| File | Lines | Missed | Coverage |\n"
    "|---|---:|---:|---:|\n" +
    "\n".join(f"| `{f}` | {l} | {m} | {c:.1f}% |" for f, l, m, c in rows)
)
print(f"Wrote {out}")
