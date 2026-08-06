"""Shared MAIN-folder mirror helper for all figure-build scripts.

Usage from any build script:

    from _mirror import mirror_to_main
    fig.savefig(out_path)
    mirror_to_main(out_path)

Will copy the file (preserving mtime) to the same relative path inside the
MAIN Dropbox folder so that figures are visible in the user's Dropbox UI
without manual rsync.
"""
import shutil
from pathlib import Path

# Source / destination roots. The CONFLICT folder is where data + scripts live;
# the MAIN folder is what the user browses in Dropbox.
CONFLICT_ROOT = Path(
    "/Users/danielliang/Library/CloudStorage/Dropbox/"
    "multi_journal_trend_analysis (Yilun Liang's conflicted copy 2026-04-27)"
)
MAIN_ROOT = Path(
    "/Users/danielliang/Library/CloudStorage/Dropbox/multi_journal_trend_analysis"
)


def mirror_to_main(out_path) -> Path | None:
    """Copy `out_path` from CONFLICT to the same relative path in MAIN.

    Returns the destination path, or None if MAIN folder doesn't exist.
    """
    out_path = Path(out_path).resolve()
    try:
        rel = out_path.relative_to(CONFLICT_ROOT)
    except ValueError:
        # out_path isn't under CONFLICT — nothing to mirror.
        print(f"[mirror] skipping non-CONFLICT path: {out_path}")
        return None
    if not MAIN_ROOT.exists():
        print(f"[mirror] MAIN folder missing: {MAIN_ROOT}")
        return None
    dst = MAIN_ROOT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_path, dst)
    print(f"[mirror] {rel}  →  MAIN")
    return dst
