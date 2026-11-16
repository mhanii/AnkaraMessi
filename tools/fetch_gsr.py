"""Fetch SoccerNet GSR ground truth without downloading the whole 9-11 GB archives.

The dataset (SoccerNet/SN-GSR-2024 on Hugging Face, GPL-3.0) ships as large zips. This reads
only the parts asked for, using HTTP range requests.

    python tools/fetch_gsr.py labels test  data/gsr/test_labels     # ~1 GB of JSON, 49 clips
    python tools/fetch_gsr.py labels valid data/gsr/valid_labels    # 58 clips
    python tools/fetch_gsr.py frames test SNGS-116 data/gsr/frames  # one clip's 750 frames

Nothing fetched here belongs in the repository: `data/` is git-ignored.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import requests

BASE = "https://huggingface.co/datasets/SoccerNet/SN-GSR-2024/resolve/main/{split}.zip"


class HttpFile(io.RawIOBase):
    """A read-only, seekable file backed by HTTP range requests, with a small block cache."""

    def __init__(self, url: str, block: int = 4 << 20):
        self.block, self.pos, self.cache = block, 0, {}
        self.session = requests.Session()
        head = self.session.head(url, allow_redirects=True, timeout=60)
        head.raise_for_status()
        self.url = head.url
        self.size = int(head.headers["content-length"])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = (offset if whence == 0 else self.pos + offset if whence == 1
                    else self.size + offset)
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self.cache:
            if len(self.cache) > 8:
                self.cache.pop(next(iter(self.cache)))
            first = i * self.block
            last = min(self.size, first + self.block) - 1
            for attempt in range(4):
                try:
                    r = self.session.get(self.url, headers={"Range": f"bytes={first}-{last}"},
                                         timeout=120)
                    r.raise_for_status()
                    self.cache[i] = r.content
                    break
                except requests.RequestException:
                    if attempt == 3:
                        raise
        return self.cache[i]

    def readinto(self, buffer) -> int:
        want = len(buffer)
        out = bytearray()
        while want > 0 and self.pos < self.size:
            i, offset = divmod(self.pos, self.block)
            chunk = self._block(i)[offset:offset + want]
            out += chunk
            self.pos += len(chunk)
            want -= len(chunk)
        buffer[:len(out)] = out
        return len(out)


def open_zip(split: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(HttpFile(BASE.format(split=split)), 1 << 20))


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    what, split = sys.argv[1], sys.argv[2]
    archive = open_zip(split)
    if what == "labels":
        out = Path(sys.argv[3])
        out.mkdir(parents=True, exist_ok=True)
        for name in sorted(archive.namelist()):
            if name.endswith("Labels-GameState.json"):
                (out / (name.split("/")[0] + ".json")).write_bytes(archive.read(name))
                print("saved", name)
    elif what == "frames":
        sequence, out = sys.argv[3], Path(sys.argv[4])
        names = [n for n in archive.namelist()
                 if n.startswith(f"{sequence}/img1/") and not n.endswith("/")]
        for name in names:
            target = out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
        print(f"saved {len(names)} frames to {out / sequence / 'img1'}")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
