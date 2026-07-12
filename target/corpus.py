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


def _stream_one(hf: dict, need: int, strip_gutenberg_flag: bool, out: list[str]) -> None:
    """Append non-empty prompt strings from one HF dataset spec until len(out) >= need."""
    from datasets import load_dataset

    kw = {"streaming": True, "split": hf["split"]}
    ds = load_dataset(hf["path"], hf["name"], **kw) if hf.get("name") else load_dataset(hf["path"], **kw)
    field = hf["field"]
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
        if len(out) >= need:
            break


def stream_prompts(hf, n: int, strip_gutenberg_flag: bool = False) -> list[str]:
    """Stream up to `n` non-empty prompts from an HF dataset spec, or a LIST of specs
    (drawn in order until `n` reached — lets a data-limited domain like code augment
    a small dataset with a larger one)."""
    specs = hf if isinstance(hf, list) else [hf]
    out: list[str] = []
    for spec in specs:
        if len(out) >= n:
            break
        _stream_one(spec, n, strip_gutenberg_flag, out)
    return out[:n]
