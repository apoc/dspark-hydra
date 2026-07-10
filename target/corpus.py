"""Calibration/eval corpus streaming (§7.1). Shared by dump + eval."""

from __future__ import annotations


def strip_gutenberg(text: str) -> str | None:
    """Return the book body between the START/END markers, else None (too short)."""
    s = text.find("*** START OF THE PROJECT GUTENBERG")
    if s != -1:
        s = text.find("\n", s)
        e = text.find("*** END OF THE PROJECT GUTENBERG")
        body = text[s:e if e != -1 else None].strip()
    else:
        body = text.strip()
    if len(body) < 2000:
        return None
    return body[500:]  # nudge past chapter headers / TOC


def stream_prompts(hf: dict, n: int, strip_gutenberg_flag: bool = False) -> list[str]:
    """Stream up to `n` non-empty prompt strings from an HF dataset spec."""
    from datasets import load_dataset

    kw = {"streaming": True, "split": hf["split"]}
    ds = load_dataset(hf["path"], hf["name"], **kw) if hf.get("name") else load_dataset(hf["path"], **kw)
    field = hf["field"]
    out: list[str] = []
    for ex in ds:
        txt = ex.get(field)
        if not txt:
            continue
        txt = str(txt)
        if strip_gutenberg_flag:
            txt = strip_gutenberg(txt)
            if txt is None:
                continue
        if txt.strip():
            out.append(txt)
        if len(out) >= n:
            break
    return out
