from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable


def _log(msg: str) -> None:
    print(msg, flush=True)


def _ensure_utf8_console() -> None:
    # Windows consoles may default to a non-UTF8 encoding (e.g., GBK).
    # Reconfigure to UTF-8 to avoid UnicodeEncodeError in logs.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


def _download(urls: Iterable[str], out_path: Path) -> bool:
    try:
        import requests  # type: ignore
    except Exception:
        _log("ERROR: missing dependency 'requests'. Install: pip install requests")
        return False

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
    }

    last_err: Exception | None = None
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code != 200:
                _log(f"WARN: download failed {out_path.name}: HTTP {r.status_code} ({url})")
                continue
            out_path.write_bytes(r.content)
            _log(f"OK: downloaded {out_path.name} <- {url}")
            return True
        except Exception as e:
            last_err = e
            _log(f"WARN: download error {out_path.name}: {e} ({url})")

    _log(f"ERROR: failed {out_path.name} (all sources tried)")
    if last_err:
        _log(f"  last_error: {last_err}")
    return False


def main() -> int:
    _ensure_utf8_console()

    js_dir = Path("app/static/js")
    js_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Target dir: {js_dir.resolve()}")

    ok = True

    ok &= _download(
        urls=[
            "https://raw.githubusercontent.com/bigskysoftware/htmx/v1.9.10/dist/htmx.min.js",
            "https://cdn.staticfile.org/htmx/1.9.10/htmx.min.js",
        ],
        out_path=js_dir / "htmx.min.js",
    )

    # HTMX extensions used by this project (preload + sse)
    ok &= _download(
        urls=[
            "https://raw.githubusercontent.com/bigskysoftware/htmx/v1.9.10/dist/ext/preload.js",
        ],
        out_path=js_dir / "htmx-ext-preload.js",
    )
    ok &= _download(
        urls=[
            "https://raw.githubusercontent.com/bigskysoftware/htmx/v1.9.10/dist/ext/sse.js",
        ],
        out_path=js_dir / "htmx-ext-sse.js",
    )

    ok &= _download(
        urls=[
            "https://cdn.staticfile.org/marked/11.1.1/marked.min.js",
        ],
        out_path=js_dir / "marked.min.js",
    )

    # Tailwind CDN runtime (JIT engine). Download as-is and rename for local use.
    ok &= _download(
        urls=[
            "https://cdn.tailwindcss.com",
        ],
        out_path=js_dir / "tailwindcss.js",
    )

    _log("All downloads completed" if ok else "Some downloads failed; check logs and retry")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
