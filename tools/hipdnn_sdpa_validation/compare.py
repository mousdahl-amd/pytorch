"""Compare the AOTriton and hipDNN runs of PyTorch's SDPA tests, per arch.

    python compare.py <results dir>   # holds <arch>/<test file>.<lib>.xml

Prints Markdown: outcome counts per arch and file, then every test whose outcome differs
between the two runs, with the first line of the hipDNN failure message.
"""
import collections
import pathlib
import sys
import xml.etree.ElementTree as ET

LIBS = ("aotriton", "hipdnn")


def outcomes(path):
    """{test id: (outcome, message)} from a junit XML file."""
    out = {}
    for case in ET.parse(path).iter("testcase"):
        tid = f"{case.get('classname', '').rsplit('.', 1)[-1]}::{case.get('name')}"
        outcome, msg = "passed", ""
        for kind in ("failure", "error", "skipped"):
            node = case.find(kind)
            if node is not None:
                outcome = {"failure": "failed", "error": "error", "skipped": "skipped"}[kind]
                text = (node.get("message") or node.text or "").strip()
                msg = text.splitlines()[0][:220] if text else ""
                break
        out[tid] = (outcome, msg)
    return out


def main(root):
    root = pathlib.Path(root)
    for arch_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        print(f"## {arch_dir.name}\n")
        files = sorted({p.name.split(".")[0] for p in arch_dir.glob("*.xml")})
        print("| File | Library | passed | failed | error | skipped |")
        print("|---|---|---|---|---|---|")
        diffs = []
        for f in files:
            runs = {}
            for lib in LIBS:
                p = arch_dir / f"{f}.{lib}.xml"
                runs[lib] = outcomes(p) if p.exists() else None
                if runs[lib] is None:
                    print(f"| {f} | {lib} | missing | | | |")
                    continue
                c = collections.Counter(o for o, _ in runs[lib].values())
                print(f"| {f} | {lib} | {c['passed']} | {c['failed']} | {c['error']} | {c['skipped']} |")
            if all(runs.values()):
                for tid in sorted(set(runs["aotriton"]) | set(runs["hipdnn"])):
                    a = runs["aotriton"].get(tid, ("absent", ""))
                    h = runs["hipdnn"].get(tid, ("absent", ""))
                    if a[0] != h[0]:
                        diffs.append((f, tid, a[0], h[0], h[1] or a[1]))
        print()
        trans = collections.Counter((a, h) for _, _, a, h, _ in diffs)
        print(f"Outcome changes AOTriton -> hipDNN: {dict(trans) or 'none'}\n")
        if diffs:
            print("| File | Test | AOTriton | hipDNN | Message |")
            print("|---|---|---|---|---|")
            for f, tid, a, h, m in diffs:
                print(f"| {f} | `{tid}` | {a} | {h} | {m.replace('|', '/')} |")
            print()


if __name__ == "__main__":
    main(sys.argv[1])
