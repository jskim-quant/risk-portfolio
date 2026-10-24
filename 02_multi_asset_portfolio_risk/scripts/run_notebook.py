"""Execute a notebook in-process and store its outputs (text, HTML tables, PNG figures).

Minimal replacement for ``jupyter nbconvert --execute`` that needs only the standard library,
matplotlib and the project's own dependencies (no nbformat / nbclient / ipykernel).

Usage (from the project root):
    python scripts/run_notebook.py notebooks/05b_frictions_sensitivity.ipynb
The working directory is set to the notebook's folder, as in Jupyter.
"""
from __future__ import annotations

import ast
import base64
import contextlib
import io
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def run(path: str) -> int:
    """Run all code cells of ``path`` top to bottom and write the outputs back to the file."""
    path = os.path.abspath(path)
    os.chdir(os.path.dirname(path))
    nb = json.load(open(path, encoding="utf-8"))
    g: dict = {"__name__": "__main__"}
    figs: list[str] = []

    def _show(*_a, **_k) -> None:
        for n in plt.get_fignums():
            b = io.BytesIO()
            plt.figure(n).savefig(b, format="png", dpi=90, bbox_inches="tight")
            figs.append(base64.b64encode(b.getvalue()).decode())
        plt.close("all")

    plt.show = _show
    cnt = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        cnt += 1
        figs.clear()
        out = []
        src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
        tree = ast.parse(src)
        last = ast.Expression(tree.body.pop().value) if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(tree, f"cell{cnt}", "exec"), g)
            res = eval(compile(last, f"cell{cnt}", "eval"), g) if last else None
        if plt.get_fignums():
            _show()
        if buf.getvalue():
            out.append({"output_type": "stream", "name": "stdout", "text": buf.getvalue()})
        for f in figs:
            out.append({"output_type": "display_data", "data": {"image/png": f, "text/plain": "<Figure>"}, "metadata": {}})
        if res is not None and not hasattr(res, "figure") and not hasattr(res, "get_figure"):
            data = {"text/plain": repr(res)}
            html = getattr(res, "_repr_html_", lambda: None)()
            if html:
                data["text/html"] = html
            out.append({"output_type": "execute_result", "execution_count": cnt, "data": data, "metadata": {}})
        c["outputs"], c["execution_count"] = out, cnt
    json.dump(nb, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return cnt


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, "→ executed", run(p), "code cells")
