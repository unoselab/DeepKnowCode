#!/usr/bin/env python3
"""J-04a: copy C-01 batch snapshots to Python before-preproc and archive, unchanged."""
import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

PROJECT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
RUN_RE = re.compile(r"^\d{8}-\d{6}-\d+$")
ROOT_SUBDIRS = ("1before-preproc", "2after-preproc", "3pos-clones", "4neg-clones", "archive")
REQUIRED_ARTIFACTS = (
    "project_run.json", "nicad.log", "jscpd.log", "merge.log",
    "file_parse_audit.csv", "filter_summary.json", "jscpd/jscpd-report.json",
    "hybrid/summary.json", "hybrid/nicad_primary_pairs.csv",
    "hybrid/jscpd_fallback_pairs.csv", "hybrid/hybrid_function_clone_pairs.csv",
)

class MigrationError(Exception):
    pass


def require(condition, explanation):
    if not condition:
        raise MigrationError(explanation)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise MigrationError(f"Invalid JSON {path}: {exc}") from exc


def safe_regular_file(path):
    require(path.is_file() and not path.is_symlink(), f"Missing/unsafe regular file: {path}")


def walk_files(root):
    require(root.is_dir() and not root.is_symlink(), f"Missing/unsafe directory: {root}")
    files = {}
    for entry in root.rglob("*"):
        require(not entry.is_symlink(), f"Symlink forbidden: {entry}")
        require(entry.is_file() or entry.is_dir(), f"Nonregular entry forbidden: {entry}")
        if entry.is_file():
            files[entry.relative_to(root).as_posix()] = entry
    return files


def require_same_files(a, b):
    af, bf = walk_files(a), walk_files(b)
    require(set(af) == set(bf), f"File-set mismatch: {a} vs {b}; missing={sorted(set(af)-set(bf))[:5]}, extra={sorted(set(bf)-set(af))[:5]}")
    for rel in sorted(af):
        require(af[rel].stat().st_size == bf[rel].stat().st_size, f"Size mismatch: {rel}")
        require(sha256(af[rel]) == sha256(bf[rel]), f"SHA256 mismatch: {rel}")
    return len(af)


def safe_relative(rel):
    require(isinstance(rel, str) and rel and "\\" not in rel, f"Unsafe relative path: {rel!r}")
    p = Path(rel)
    require(not p.is_absolute() and all(s not in ("", ".", "..") for s in p.parts), f"Unsafe relative path: {rel!r}")
    return p


def slug_to_name(owner, name):
    require(PROJECT_RE.fullmatch(owner or "") and PROJECT_RE.fullmatch(name or ""), "Unsafe github slug")
    return f"{owner.lower()}__{name.lower()}"


def remote_identity(remote):
    if remote.startswith("git@github.com:"):
        full = remote[len("git@github.com:"):]
    else:
        u = urlparse(remote)
        require(u.hostname == "github.com" and u.scheme in ("https", "http", "ssh", "git"), f"Unsupported remote: {remote}")
        full = u.path.strip("/")
    if full.endswith(".git"):
        full = full[:-4]
    pair = full.strip("/").split("/")
    require(len(pair) == 2, f"Unexpected remote slug: {remote}")
    return pair[0], pair[1]


def verify_snapshot(snapshot, entry, source_root):
    require(snapshot.parent.name == "runs", f"Unexpected run parent: {snapshot}")
    require(snapshot.name == entry["c01_run_id"] and RUN_RE.fullmatch(snapshot.name), "Run ID mismatch")
    manifest_path = snapshot / "artifact_manifest.json"
    safe_regular_file(manifest_path)
    manifest = read_json(manifest_path)
    require(manifest.get("project") == entry.get("project"), "Manifest project mismatch")
    for k in ("github_slug", "commit", "remote", "c01_run_id", "files"):
        require(manifest.get(k) == entry.get(k), f"Manifest/organizer conflict: {k} at {snapshot}")
    # C-01 v9 writes artifact_manifest.json BEFORE setting entry["action"] = "COPIED".
    # The organizer summary records COPIED, while source snapshot manifests normally
    # omit this field.  Preserve exact original bytes; never backfill action.
    require(manifest.get("action") in (None, "COPIED"),
            f"Invalid snapshot manifest action: {manifest.get('action')!r}")
    require(manifest.get("snapshot") == str(source_root / snapshot.relative_to(source_root)), "Original snapshot identity mismatch")
    files = manifest["files"]
    require(isinstance(files, dict) and len(files) == entry.get("file_count", -1), "Manifest file count mismatch")
    require(set(REQUIRED_ARTIFACTS).issubset(files), "Missing required artifacts in manifest")
    actual = walk_files(snapshot)
    require(set(actual) == set(files) | {"artifact_manifest.json"}, f"Unexpected/missing files at {snapshot}")
    for rel, info in files.items():
        safe_relative(rel)
        path = snapshot / rel
        safe_regular_file(path)
        require(isinstance(info, dict) and SHA256.fullmatch(str(info.get("sha256", ""))), f"Invalid hash in {rel}")
        require(path.stat().st_size == info.get("bytes") and sha256(path) == info["sha256"], f"Manifest SHA256/size mismatch: {path}")
    run = read_json(snapshot / "project_run.json")
    for k in ("project", "commit", "remote"):
        require(run.get(k) == entry.get(k), f"project_run {k} mismatch at {snapshot}")
    require(run.get("status") == "COMPLETED", "Run is not COMPLETED")
    return manifest


def inspect_source(src, expected_subjects, organizer_name, batch_name):
    require(src.is_dir() and not src.is_symlink(), f"Missing source: {src}")
    require(organizer_name.startswith("organizer-") and organizer_name.endswith(".json"), "Invalid organizer filename")
    require(RUN_RE.fullmatch(batch_name), "Invalid batch run name")
    safe_regular_file(src / organizer_name)
    organizer = read_json(src / organizer_name)
    require(organizer.get("status") == "COMPLETED" and organizer.get("selected") == expected_subjects, "Organizer incomplete or selected count mismatch")
    entries = organizer.get("projects")
    require(isinstance(entries, list) and len(entries) == expected_subjects, "Organizer entry count mismatch")
    root_names = {p.name for p in src.iterdir()}
    subject_dirs = {p.name for p in src.iterdir() if p.is_dir() and p.name != batch_name}
    require(len(subject_dirs) == expected_subjects, f"Expected {expected_subjects} subject directories, observed {len(subject_dirs)}")
    require(root_names == subject_dirs | {batch_name, organizer_name}, f"Unexpected/missing source root entries: {sorted(root_names ^ (subject_dirs | {batch_name, organizer_name}))}")
    require((src / batch_name).is_dir() and not (src / batch_name).is_symlink(), "Batch root missing/unsafe")
    index = []
    seen_original, seen_canonical = set(), set()
    for e in entries:
        require(isinstance(e, dict) and e.get("action") == "COPIED", "Organizer includes non-COPIED entry")
        owner, repo = remote_identity(e.get("remote", ""))
        require(e.get("github_slug") == owner + "/" + repo, "Organizer GitHub slug mismatch")
        require(e.get("github_owner") == owner and e.get("github_repository") == repo, "Organizer owner/repo mismatch")
        require(SHA40.fullmatch(str(e.get("commit", ""))), "Invalid Git SHA")
        name = f"{owner}--{repo}"
        require(name in subject_dirs, f"Missing subject dir: {name}")
        canonical = slug_to_name(owner, repo)
        require(name not in seen_original and canonical not in seen_canonical, f"Duplicate Subject: {canonical}")
        seen_original.add(name)
        seen_canonical.add(canonical)
        runid = e.get("c01_run_id", "")
        require(isinstance(runid, str) and RUN_RE.fullmatch(runid), "Invalid run ID")
        parent = src / name
        require(set(p.name for p in parent.iterdir()) == {"runs"}, f"Unexpected entries in {parent}")
        run_root = parent / "runs"
        require(run_root.is_dir() and not run_root.is_symlink(), "Invalid runs directory")
        require(set(p.name for p in run_root.iterdir()) == {runid}, f"Unexpected runs in {run_root}")
        snapshot = run_root / runid
        verify_snapshot(snapshot, e, src)
        index.append((canonical, name, runid, e))
        print(f"PASS SOURCE subject={canonical} run={runid} files={e['file_count'] + 1}", flush=True)
    require(seen_original == subject_dirs, "Unmapped subject dirs")
    return sorted(index)


def publication_record(canonical, name, runid, entry, source_root, publication_version):
    snapshot = source_root / name / "runs" / runid
    orig_manifest = snapshot / "artifact_manifest.json"
    return {
        "experiment": "J-04a", "version": publication_version, "status": "DETECTOR_SNAPSHOT_COPY_UNMODIFIED",
        "dataset_role": "BEFORE_PREPROC_C01_DETECTOR_ARTIFACTS_NOT_SOURCE_CODE",
        "positive_labels_verified": False, "negative_pairs_present": False,
        "subject": canonical, "source_subject": name, "c01_run_id": runid,
        "source_snapshot": str(snapshot), "remote": entry["remote"], "git_commit": entry["commit"],
        "source_artifact_manifest_sha256": sha256(orig_manifest),
        "source_files": entry["files"],
        "published_artifact_manifest_sha256": sha256(orig_manifest),
    }


def verify_target(src, dst, index, batch_name, publication_version):
    before = dst / "1before-preproc"
    archive = dst / "archive" / "c-01-batch"
    require(before.is_dir() and not before.is_symlink(), "Missing before-preproc")
    require(archive.is_dir() and not archive.is_symlink(), "Missing archive")
    for dirname in ROOT_SUBDIRS:
        require((dst / dirname).is_dir() and not (dst / dirname).is_symlink(), f"Missing stage: {dirname}")
    for dirname in ("2after-preproc", "3pos-clones", "4neg-clones"):
        require(not any((dst / dirname).iterdir()), f"Expected empty reserved stage: {dirname}")
    require(set(p.name for p in before.iterdir()) == {row[0] for row in index}, "Published subject directory set mismatch")
    require(set(p.name for p in (dst / "archive").iterdir()) == {"c-01-batch"}, "Archive root has unexpected entries")
    nfiles = require_same_files(src, archive)
    for canonical, name, runid, entry in index:
        original = src / name / "runs" / runid
        published = before / canonical
        published_files = walk_files(published)
        require("j-04a-publication.json" in published_files, f"Missing publication manifest: {canonical}")
        original_files = walk_files(original)
        require(set(published_files) == set(original_files) | {"j-04a-publication.json"}, "Published files differ")
        for rel, orig in original_files.items():
            require(sha256(orig) == sha256(published / rel), f"Published file changed: {canonical}/{rel}")
        require(read_json(published / "j-04a-publication.json") == publication_record(canonical, name, runid, entry, src, publication_version), f"Publication manifest mismatch: {canonical}")
        print(f"PASS VERIFY subject={canonical}", flush=True)
    print(f"PASS VERIFY_ALL subjects={len(index)} archive_files={nfiles} output={dst}", flush=True)


def do_copy(src, dst, index, batch_name, publication_version):
    require(dst.is_dir() and not dst.is_symlink(), f"Target Python root must already exist: {dst}")
    for dirname in ROOT_SUBDIRS:
        p = dst / dirname
        require(not p.exists() and not p.is_symlink() or (p.is_dir() and not p.is_symlink() and not any(p.iterdir())), f"Destination must be empty: {p}")
    # Never overwrite a previously published J-04a result.
    stage = Path(tempfile.mkdtemp(prefix=".j-04a-stage-", dir=dst))
    installed = []
    try:
        for dirname in ROOT_SUBDIRS:
            (stage / dirname).mkdir()
        shutil.copytree(src, stage / "archive" / "c-01-batch", symlinks=False)
        require_same_files(src, stage / "archive" / "c-01-batch")
        print("PASS COPY_ARCHIVE", flush=True)
        for canonical, name, runid, entry in index:
            source_snapshot = stage / "archive" / "c-01-batch" / name / "runs" / runid
            target = stage / "1before-preproc" / canonical
            shutil.copytree(source_snapshot, target, symlinks=False)
            record = publication_record(canonical, name, runid, entry, src, publication_version)
            (target / "j-04a-publication.json").write_text(json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
            print(f"PASS PREPARED subject={canonical}", flush=True)
        verify_target(src, stage, index, batch_name, publication_version)
        # Install only after complete staging verification. Empty user-created directories remain untouched.
        for dirname in ROOT_SUBDIRS:
            dest = dst / dirname
            work = stage / dirname
            if dest.exists():
                require(dest.is_dir() and not any(dest.iterdir()), f"Changed destination during staging: {dest}")
                # For pre-existing empty directories, move only new children in; never delete dirs.
                for child in list(work.iterdir()):
                    require(not (dest / child.name).exists(), f"Collision: {dest / child.name}")
                    child.rename(dest / child.name)
                    installed.append(dest / child.name)
            else:
                work.rename(dest)
                installed.append(dest)
        verify_target(src, dst, index, batch_name, publication_version)
        print(f"PASS COPY subjects={len(index)} destination={dst}", flush=True)
    except BaseException:
        # Remove only artifacts installed in this invocation, not any original/user data.
        for path in reversed(installed):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--python-root", type=Path, required=True)
    parser.add_argument("--expected-subjects", type=int, required=True)
    parser.add_argument("--organizer-name", required=True)
    parser.add_argument("--batch-name", required=True)
    parser.add_argument("--publication-version", required=True)
    parser.add_argument("--mode", choices=("inspect", "dry-run", "copy", "verify"), required=True)
    args = parser.parse_args()
    try:
        src = args.source_root.absolute()
        dst = args.python_root.absolute()
        require(args.expected_subjects > 0, "Invalid expected subjects")
        require(re.fullmatch(r"v[1-9][0-9]*", args.publication_version), "Invalid publication version")
        require(src != dst and src not in dst.parents and dst not in src.parents, "Source and destination must not overlap")
        require(not src.is_symlink() and not dst.is_symlink(), "Top-level symlinks disallowed")
        index = inspect_source(src, args.expected_subjects, args.organizer_name, args.batch_name)
        if args.mode == "inspect":
            print(f"PASS INSPECT subjects={len(index)} source={src}")
        elif args.mode == "dry-run":
            require(dst.is_dir(), f"Missing Python root: {dst}")
            for dirname in ROOT_SUBDIRS:
                p = dst / dirname
                require(not p.exists() or (p.is_dir() and not any(p.iterdir()) and not p.is_symlink()), f"Nonempty/unsafe target: {p}")
            print(f"PASS DRY_RUN subjects={len(index)} destination={dst}")
        elif args.mode == "copy":
            do_copy(src, dst, index, args.batch_name, args.publication_version)
        else:
            verify_target(src, dst, index, args.batch_name, args.publication_version)
        return 0
    except (MigrationError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"FAIL J-04a: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
