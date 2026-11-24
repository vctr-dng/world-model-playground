"""Check the documentation: relative links, heading anchors, and script paths.

Run from anywhere: ``python scripts/check_docs.py``. It exits 1 and lists every
problem if a link or an anchor does not resolve, or if a page names a script that
does not exist. It needs only the standard library, so CI runs it without
installing the project.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = [
    ROOT / "README.md",
    ROOT / "scripts/cartpole/README.md",
    *sorted((ROOT / "docs").glob("*.md")),
]
LINK = re.compile(r"\[[^\]]+\]\(([^)\s]+)\)")
# Only this repository's scripts: pages also quote files of other repositories.
SCRIPT = re.compile(r"scripts/cartpole/[\w/]+\.py")


def slug(heading: str) -> str:
    """The anchor GitHub generates for a heading."""
    heading = re.sub(r"[`*_]", "", heading.strip().lower())
    heading = re.sub(r"[^\w\s-]", "", heading)
    return re.sub(r"\s", "-", heading)


def anchors(path: Path) -> set[str]:
    """Anchors of the headings of a page, ignoring comments in code blocks."""
    found, fenced = set(), False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            fenced = not fenced
        match = None if fenced else re.match(r"#+ (.*)", line)
        if match:
            found.add(slug(match.group(1)))
    return found


def check() -> list[str]:
    problems = []
    for page in PAGES:
        name = page.relative_to(ROOT)
        text = page.read_text(encoding="utf-8")
        for target in LINK.findall(text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            file_part, _, fragment = target.partition("#")
            resolved = (page.parent / file_part).resolve() if file_part else page
            if not resolved.exists():
                problems.append(f"{name}: broken link {target}")
            elif fragment and resolved.suffix == ".md":
                if fragment not in anchors(resolved):
                    problems.append(
                        f"{name}: no heading for #{fragment} in {resolved.name}"
                    )
        for script in sorted(set(SCRIPT.findall(text))):
            if not (ROOT / script).is_file():
                problems.append(f"{name}: no such script {script}")
    return problems


def main() -> int:
    problems = check()
    for problem in problems:
        print(problem)
    print(f"checked {len(PAGES)} pages, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
