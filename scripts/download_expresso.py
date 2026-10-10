"""Download and extract the official Expresso release (read-speech subset) into <data root>/raw/.

Usage: .venv/bin/python scripts/download_expresso.py [--status] [--no-extract] [--allow-bad-audio]
                                                      [--config PATH]

What it does, in order (each step is skipped when its result is already verified, so re-running
never downloads or extracts twice):
  1. Download expresso.tar from the official URL in the config (the one quoted by the dataset README
     in facebookresearch/textlesslib), resuming a partial download with HTTP range requests.
  2. Verify size and the md5 published in that README (a sha256 is recorded alongside).
  3. Extract only the members listed in [source].extract (read speech, transcripts, splits), safely:
     regular files and directories only, no absolute paths or '..', written to a temp name first.
  4. Check every extracted WAV header (RIFF/WAV, 48 kHz mono, nonzero length).
  5. Record URL, retrieval date, license, attribution and checksums in <raw>/SOURCES.json.

Runs from any directory. Needs ~38 GB for the archive plus ~7 GB for the extracted audio.
Ctrl-C is safe: the partial file is kept and the next run resumes it.
"""
import argparse
import fnmatch
import hashlib
import http.client
import os
import posixpath
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from _common import ROOT  # first: loads .env
from expresso_common import Config, is_junk, log, read_json, sha256_file, utc_now, write_json

ARCHIVE = "expresso.tar"
EXTRACT_ESTIMATE_BYTES = 7 * 10**9   # read-speech audio is ~6 GB (11.5 h of 48 kHz / 24 bit mono)
MARGIN_BYTES = 2 * 10**9
MAX_ATTEMPTS = 20
CHUNK = 1 << 20


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1000


def head(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return {
            "size": int(r.headers["Content-Length"]),
            "etag": (r.headers.get("ETag") or "").strip(),
            "last_modified": r.headers.get("Last-Modified", ""),
            "accept_ranges": r.headers.get("Accept-Ranges", ""),
        }


class Progress:
    def __init__(self, label, total, start=0, every=10.0):
        self.label, self.total, self.start, self.every = label, total, start, every
        self.t0 = self.last = time.time()

    def update(self, done, force=False):
        now = time.time()
        if not force and now - self.last < self.every:
            return
        self.last = now
        rate = (done - self.start) / max(now - self.t0, 1e-9)
        eta = (self.total - done) / rate if rate > 0 else float("inf")
        log(f"[{self.label}] {human(done)} / {human(self.total)} ({100 * done / self.total:.1f}%) "
            f"{rate / 1e6:.1f} MB/s, ETA {eta / 60:.1f} min")


def download(url, part, meta_path, expected_size, remote):
    """Fetch url into `part`, resuming from its current size. Returns when part has expected_size bytes."""
    meta = read_json(meta_path) if meta_path.is_file() else {}
    if meta.get("etag") and remote["etag"] and meta["etag"] != remote["etag"]:
        log(f"[download] remote object changed ({meta['etag']} -> {remote['etag']}); restarting from zero")
        part.unlink(missing_ok=True)
    meta.update(etag=remote["etag"], last_modified=remote["last_modified"], url=url)
    meta.setdefault("download_started_at", utc_now())
    write_json(meta_path, meta)

    attempts = 0
    while True:
        have = part.stat().st_size if part.exists() else 0
        if have > expected_size:
            raise SystemExit(f"{part} is larger ({have}) than the expected archive ({expected_size}); move it aside and re-run")
        if have == expected_size:
            return
        if attempts >= MAX_ATTEMPTS:
            raise SystemExit(f"giving up after {MAX_ATTEMPTS} failed attempts; re-run to resume from {human(have)}")
        attempts += 1
        headers = {"Range": f"bytes={have}-"}
        if have and remote["etag"]:
            headers["If-Range"] = remote["etag"]
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status == 206:
                    cr = r.headers.get("Content-Range", "")
                    if not cr.startswith(f"bytes {have}-"):
                        raise SystemExit(f"unexpected Content-Range {cr!r} when resuming at {have}")
                    mode = "ab"
                elif r.status == 200:
                    if have:
                        log("[download] server ignored the range request; restarting from zero")
                    mode, have = "wb", 0
                else:
                    raise SystemExit(f"unexpected HTTP status {r.status}")
                log(f"[download] {'resuming at ' + human(have) if have else 'starting'} (attempt {attempts})")
                prog = Progress("download", expected_size, start=have)
                with open(part, mode) as f:
                    got = have
                    while block := r.read(CHUNK):
                        f.write(block)
                        got += len(block)
                        prog.update(got)
                    f.flush()
                    os.fsync(f.fileno())
                prog.update(got, force=True)
                attempts = 0 if got > have else attempts   # progress was made: reset the failure counter
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException, OSError) as e:
            wait = min(60, 2 ** min(attempts, 6))
            log(f"[download] interrupted ({type(e).__name__}: {e}); retrying in {wait}s")
            time.sleep(wait)


def hash_archive(path, expected_size):
    """md5 (published) and sha256 (ours) in one pass."""
    md5, sha = hashlib.md5(), hashlib.sha256()
    prog = Progress("checksum", expected_size)
    done = 0
    with open(path, "rb") as f:
        while block := f.read(CHUNK * 4):
            md5.update(block)
            sha.update(block)
            done += len(block)
            prog.update(done)
    return md5.hexdigest(), sha.hexdigest()


def safe_member_path(name, dest):
    """Destination path for a tar member name, or None if it could escape `dest`."""
    if name.startswith("./"):
        name = name[2:]
    norm = posixpath.normpath(name)
    if posixpath.isabs(norm) or norm == ".." or norm.startswith("../") or "\\" in norm:
        return None
    target = (dest / norm).resolve()
    if dest.resolve() not in target.parents and target != dest.resolve():
        return None
    return target


def extract(tar_path, dest, patterns):
    """Extract matching regular files from the tar. Returns (files, bytes, skipped, rejected)."""
    files = size = skipped = 0
    rejected = []
    t0 = time.time()
    with tarfile.open(tar_path, "r:") as tf:
        for m in tf:
            if not any(fnmatch.fnmatch(m.name, p) for p in patterns):
                continue
            target = safe_member_path(m.name, dest)
            if target is None or not (m.isreg() or m.isdir()):
                rejected.append((m.name, m.type))
                continue
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            files += 1
            size += m.size
            if target.is_file() and target.stat().st_size == m.size:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".partial")
            src = tf.extractfile(m)
            with open(tmp, "wb") as out:
                shutil.copyfileobj(src, out, CHUNK)
            if tmp.stat().st_size != m.size:
                raise SystemExit(f"short write for {m.name}")
            tmp.chmod(0o644)
            tmp.replace(target)
            if (files - skipped) % 500 == 0:
                log(f"[extract] {files - skipped} files written ({human(size)} seen, {time.time() - t0:.0f}s)")
    return files, size, skipped, rejected


def validate_audio(raw_root):
    """Header-level check of every extracted WAV. Returns (n_ok, problems)."""
    import soundfile as sf

    ok, problems = 0, []
    wavs = sorted(p for p in (raw_root / "expresso").rglob("*.wav") if not is_junk(p))
    for i, p in enumerate(wavs):
        rel = str(p.relative_to(raw_root))
        try:
            info = sf.info(str(p))
            issues = []
            if info.format != "WAV":
                issues.append(f"format {info.format}")
            if info.frames <= 0:
                issues.append("no frames")
            if info.samplerate != 48000:
                issues.append(f"sample rate {info.samplerate}")
            if info.channels != 1:
                issues.append(f"{info.channels} channels")
            if issues:
                problems.append({"file": rel, "problem": "; ".join(issues)})
            else:
                ok += 1
        except Exception as e:  # soundfile raises LibsndfileError / RuntimeError on unreadable files
            problems.append({"file": rel, "problem": f"unreadable: {e}"})
        if (i + 1) % 2000 == 0:
            log(f"[validate] {i + 1}/{len(wavs)} headers checked")
    return ok, problems, len(wavs)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", help="path to the experiment config (default configs/expresso_emotion.toml)")
    ap.add_argument("--status", action="store_true", help="report what is present and verified, change nothing")
    ap.add_argument("--no-extract", action="store_true", help="download and verify the archive only")
    ap.add_argument("--allow-bad-audio", action="store_true", help="exit 0 even if some WAV headers fail validation")
    args = ap.parse_args()

    cfg = Config(args.config)
    src = cfg["source"]
    cfg.require_data_root()
    raw = cfg.path("raw")
    raw.mkdir(parents=True, exist_ok=True)
    archive, part = raw / ARCHIVE, raw / (ARCHIVE + ".part")
    part_meta = raw / (ARCHIVE + ".part.json")
    sources_path = raw / "SOURCES.json"
    expected = src["size_bytes"]
    record = read_json(sources_path) if sources_path.is_file() else {}

    def archive_verified():
        a = record.get("archive", {})
        if not (archive.is_file() and a.get("md5_verified")):
            return False
        st = archive.stat()
        return st.st_size == a.get("size_bytes") == expected and st.st_mtime_ns == a.get("mtime_ns")

    if args.status:
        log(f"data root : {cfg.data_root}")
        log(f"archive   : {'verified' if archive_verified() else 'present, unverified' if archive.is_file() else 'partial (' + human(part.stat().st_size) + ')' if part.is_file() else 'absent'}")
        log(f"extracted : {'yes' if record.get('extraction', {}).get('complete') else 'no'}")
        return

    # ---- 1-2. download + verify -------------------------------------------------------------
    already_extracted = record.get("extraction", {}).get("complete") and not args.no_extract
    if archive_verified():
        log(f"[download] {archive.name} already downloaded and verified (md5 {src['md5']}); skipping")
    elif not archive.is_file() and already_extracted and not src["keep_archive"]:
        log("[download] archive was removed after extraction (keep_archive=false); skipping")
    else:
        if not archive.is_file():
            remote = head(src["url"])
            if remote["size"] != expected:
                raise SystemExit(f"server reports {remote['size']} bytes, config expects {expected}: the release changed; "
                                 f"check {src['readme_url']} before updating [source] in the config")
            if "bytes" not in remote["accept_ranges"].lower():
                log("[download] warning: server does not advertise range support; an interrupted download restarts")
            remaining = expected - (part.stat().st_size if part.exists() else 0)
            need = remaining + EXTRACT_ESTIMATE_BYTES + MARGIN_BYTES
            free = shutil.disk_usage(raw).free
            if free < need:
                raise SystemExit(f"not enough disk space at {raw}: need ~{human(need)}, have {human(free)}. "
                                 f"Point data_root (or VIBE_TTS_DATA_DIR) at a bigger volume.")
            started = time.time()
            download(src["url"], part, part_meta, expected, remote)
            if part.stat().st_size != expected:
                raise SystemExit(f"size mismatch: {part.stat().st_size} != {expected}")
            meta = read_json(part_meta)
            md5, sha = hash_archive(part, expected)
            if md5 != src["md5"]:
                bad = part.with_name(ARCHIVE + ".corrupt")
                part.replace(bad)
                raise SystemExit(f"md5 mismatch: got {md5}, official {src['md5']}. Moved to {bad.name}; delete it and re-run to download again.")
            part.replace(archive)
            part_meta.unlink(missing_ok=True)
            record = {
                "dataset": "Expresso",
                "source_url": src["url"],
                "project_page": src["project_page"],
                "readme_url": src["readme_url"],
                "license": src["license"],
                "license_url": src["license_url"],
                "attribution": src["attribution"],
                "retrieved_at": utc_now(),
                "download_started_at": meta.get("download_started_at"),
                "archive": {
                    "file": ARCHIVE, "size_bytes": expected, "md5": md5, "sha256": sha,
                    "md5_verified": True, "md5_source": "official dataset README (textlesslib)",
                    "etag": meta.get("etag"), "last_modified": meta.get("last_modified"),
                    "mtime_ns": archive.stat().st_mtime_ns,
                },
            }
            write_json(sources_path, record)
            log(f"[download] verified: md5 {md5} matches the official value ({(time.time() - started) / 60:.1f} min)")
        else:
            log(f"[checksum] {archive.name} exists without a verification record; hashing it once")
            st = archive.stat()
            if st.st_size != expected:
                raise SystemExit(f"{archive} is {st.st_size} bytes, expected {expected}; move it aside and re-run")
            md5, sha = hash_archive(archive, expected)
            if md5 != src["md5"]:
                raise SystemExit(f"md5 mismatch for {archive}: got {md5}, official {src['md5']}")
            record = {
                "dataset": "Expresso", "source_url": src["url"], "project_page": src["project_page"],
                "readme_url": src["readme_url"], "license": src["license"], "license_url": src["license_url"],
                "attribution": src["attribution"], "retrieved_at": utc_now(),
                "note": "archive was already on disk; hashed and verified, retrieval date is the verification date",
                "archive": {"file": ARCHIVE, "size_bytes": expected, "md5": md5, "sha256": sha, "md5_verified": True,
                            "md5_source": "official dataset README (textlesslib)", "mtime_ns": st.st_mtime_ns},
            }
            write_json(sources_path, record)

    if args.no_extract:
        return

    # ---- 3. extract -----------------------------------------------------------------------
    if archive.is_file():
        patterns = src["extract"]
        log(f"[extract] members matching {patterns}")
        files, size, skipped, rejected = extract(archive, raw, patterns)
        if rejected:
            raise SystemExit(f"refused unsafe or unsupported tar members: {rejected[:5]}")
        if files == 0:
            raise SystemExit("no archive members matched [source].extract; the archive layout differs from the README")
        log(f"[extract] {files} files ({human(size)}); {files - skipped} written, {skipped} already present")
        text_sums = {
            str(p.relative_to(raw)): sha256_file(p)
            for p in sorted((raw / "expresso").glob("*.txt")) + sorted((raw / "expresso" / "splits").glob("*"))
            if p.is_file() and not is_junk(p)
        }
        record["extraction"] = {
            "complete": True, "patterns": patterns, "files": files, "bytes": size,
            "completed_at": utc_now(), "text_file_sha256": text_sums,
        }
        write_json(sources_path, record)
        if not src["keep_archive"]:
            archive.unlink()
            log("[extract] keep_archive=false: removed the archive (record kept; re-run would download again)")

    # ---- 4. validate audio ------------------------------------------------------------------
    ok, problems, total = validate_audio(raw)
    record["audio_validation"] = {"checked": total, "ok": ok, "problems": problems, "at": utc_now()}
    write_json(sources_path, record)
    log(f"[validate] {ok}/{total} WAV headers valid (48 kHz mono, non-empty)")
    for p in problems[:20]:
        log(f"  PROBLEM {p['file']}: {p['problem']}", err=True)
    if problems and not args.allow_bad_audio:
        raise SystemExit(f"{len(problems)} audio file(s) failed validation (details in {sources_path}); "
                         f"re-run with --allow-bad-audio to continue and let the inventory log them")
    log(f"[done] provenance record: {sources_path}")


if __name__ == "__main__":
    main()
