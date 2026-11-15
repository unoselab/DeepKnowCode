#!/usr/bin/env python3
"""H-10 v3: retire legacy Python code map after archive verification.

The D-05bj v2 Python Ground Truth is self-contained: code lives in left/right
source_code and is integrity-protected by side-level SHA256. Do not regenerate it.
All experiment-specific values and paths arrive as required CLI arguments.
"""
import argparse
import collections
import copy
import hashlib
import json
import os
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


REMOVED_PYTHON_KEYS = ("code_map_path", "code_map_sha256", "code_ids")


def require(test, why):
    if not test:
        raise ValueError(why)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as inp:
        for block in iter(lambda: inp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_sha(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(obj):
    return (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def load_json(path):
    require(path.is_file(), "Missing JSON: " + str(path))
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "Expected JSON object: " + str(path))
    return value


def jsonl(path):
    require(path.is_file(), "Missing JSONL: " + str(path))
    with path.open(encoding="utf-8-sig") as inp:
        for lineno, line in enumerate(inp, 1):
            require(bool(line.strip()), f"Blank JSONL line in {path}:{lineno}")
            obj = json.loads(line)
            require(isinstance(obj, dict), f"Invalid object in {path}:{lineno}")
            yield obj


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def validate_ground_truth(args):
    gt = args.ground_truth
    require(gt.is_file() and not gt.is_symlink(), "Missing or symlinked Ground Truth")
    require(sha256(gt) == args.expected_gt_sha256,
            "Ground Truth SHA256 disagrees with D-05bj v2 frozen identity")
    root = gt.parent
    manifest = load_json(root / "manifest.json")
    require(manifest.get("status") == args.expected_gt_status,
            "Unexpected D-05bj Ground Truth status")
    files = manifest.get("files")
    require(isinstance(files, dict), "Missing Ground Truth manifest file inventory")
    names = (gt.name, args.positive_filename, args.negative_filename)
    for name in names:
        require(name in files, "Missing Ground Truth manifest inventory entry: " + name)
    for name, meta in files.items():
        require(name == Path(name).name and name not in (".", ".."),
                "Unsafe Ground Truth inventory member: " + name)
        file = root / name
        require(file.is_file() and not file.is_symlink(), "Missing Ground Truth artifact: " + name)
        require(sha256(file) == meta.get("sha256"), "Ground Truth artifact SHA mismatch: " + name)
        require(file.stat().st_size == meta.get("bytes"), "Ground Truth artifact byte mismatch: " + name)
    stats = collections.Counter()
    projects = set()
    pair_ids = set()
    code_pairs = set()
    all_counter = collections.Counter()
    for item in jsonl(gt):
        pid, project, label = item.get("pair_id"), item.get("project"), item.get("gold_label")
        require(isinstance(pid, str) and pid and pid not in pair_ids,
                "Invalid/duplicate Ground Truth pair_id: " + str(pid))
        require(isinstance(project, str) and project, "Missing Ground Truth project")
        require(type(label) is int and label in (0, 1), "Invalid Ground Truth label: " + pid)
        require(item.get("evaluation_eligible") is True, "Not eligible: " + pid)
        hashes = []
        for side_name in ("left", "right"):
            side = item.get(side_name)
            require(isinstance(side, dict), "Missing side " + pid + "/" + side_name)
            code = side.get("source_code")
            require(isinstance(code, str) and bool(code.strip()), "Missing embedded code " + pid)
            digest = data_sha(code.encode("utf-8"))
            require(digest == side.get("code_sha256"),
                    "Embedded code SHA mismatch: " + pid + "/" + side_name)
            hashes.append(digest)
        sorted_hashes = sorted(hashes)
        require(item.get("pair_code_hashes") == sorted_hashes,
                "Pair code SHA mismatch: " + pid)
        pair_key = tuple(sorted_hashes)
        require(pair_key not in code_pairs, "Duplicate code-hash pair: " + pid)
        code_pairs.add(pair_key)
        pair_ids.add(pid)
        projects.add(project)
        stats[label] += 1
        all_counter[canonical(item)] += 1
    total = sum(stats.values())
    require(total == args.expected_pairs, "Unexpected Ground Truth count")
    require(stats[1] == args.expected_positive and stats[0] == args.expected_negative,
            "Unexpected Ground Truth label split")
    require(len(projects) == args.expected_projects, "Unexpected Ground Truth project coverage")
    require(manifest.get("record_counts", {}).get("binary") == total,
            "Ground Truth manifest binary count mismatch")
    require(manifest.get("record_counts", {}).get("positive") == stats[1] and
            manifest.get("record_counts", {}).get("negative") == stats[0],
            "Ground Truth manifest class count mismatch")
    combined = collections.Counter()
    for name, target_label, expected in ((args.positive_filename, 1, args.expected_positive),
                                          (args.negative_filename, 0, args.expected_negative)):
        count = 0
        for obj in jsonl(root / name):
            require(obj.get("gold_label") == target_label, "Split label mismatch: " + name)
            combined[canonical(obj)] += 1
            count += 1
        require(count == expected, "Split row count mismatch: " + name)
    require(combined == all_counter, "Ground Truth combined/split records disagree")
    return {"pairs": total, "positive_pairs": stats[1], "negative_pairs": stats[0],
            "projects": len(projects), "dataset_sha256": sha256(gt),
            "embedded_code_sides_verified": 2 * total, "split_records_equal": True}


def validate_legacy_map(args):
    path = args.legacy_map
    require(path.is_file() and not path.is_symlink(), "Missing or symlinked legacy code map")
    require(sha256(path) == args.expected_legacy_sha256,
            "Legacy map SHA256 mismatch: expected pinned October map")
    ids = set()
    for row in jsonl(path):
        ident = row.get("id")
        require(isinstance(ident, str) and bool(ident.strip()) and ident not in ids,
                "Invalid or duplicate legacy Code ID")
        ids.add(ident)
    require(len(ids) == args.expected_legacy_code_ids, "Legacy Code Map ID count mismatch")
    return {"code_ids": len(ids), "code_map_sha256": sha256(path),
            "bytes": path.stat().st_size}


def validate_manifests(args):
    paths = [args.manifest, args.manifest_details]
    require(paths[0].resolve() != paths[1].resolve(), "Manifest paths must differ")
    docs = [load_json(p) for p in paths]
    result = []
    for path, doc in zip(paths, docs):
        langs = doc.get("languages")
        require(isinstance(langs, dict) and isinstance(langs.get("java"), dict),
                "Missing Java metadata in " + str(path))
        py = langs.get("python")
        require(isinstance(py, dict), "Missing Python metadata in " + str(path))
        require(py.get("dataset_path") == args.gt_relative_path,
                "Unexpected Python dataset_path in " + str(path))
        for field, expected in (("projects", args.expected_projects),
                                ("pairs", args.expected_pairs),
                                ("positive_pairs", args.expected_positive),
                                ("negative_pairs", args.expected_negative)):
            require(py.get(field) == expected, f"Python {field} differs from D-05bj in {path}")
        # Fail closed if a different/unknown Code Map is named: do not hide it.
        if "code_map_path" in py:
            require(py["code_map_path"] == args.legacy_relative_path,
                    "Manifest points to a different code map: " + str(path))
        if "code_map_sha256" in py:
            require(py["code_map_sha256"] == args.expected_legacy_sha256,
                    "Manifest points to a different code map hash: " + str(path))
        if "code_ids" in py:
            require(py["code_ids"] == args.expected_legacy_code_ids,
                    "Manifest code_ids does not describe expected legacy map: " + str(path))
        new = copy.deepcopy(doc)
        for key in REMOVED_PYTHON_KEYS:
            new["languages"]["python"].pop(key, None)
        require(new["languages"]["java"] == doc["languages"]["java"],
                "Java metadata unexpectedly changed")
        require(args.legacy_relative_path not in canonical(new),
                "Legacy map path remains in active manifest")
        result.append(new)
    return paths, docs, result


def atomic_replace(path, raw):
    fd, tmp = tempfile.mkstemp(prefix=".h10v3-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(raw)
            out.flush()
            os.fsync(out.fileno())
        os.chmod(tmp, path.stat().st_mode & 0o777)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def exclusive_write(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def validate_archive(archive, arcname, source_sha, source_size):
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        require(len(members) == 1 and members[0].isfile() and members[0].name == arcname,
                "Archive must contain exactly the expected safe JSONL member")
        member = members[0]
        require(member.size == source_size, "Archive uncompressed byte count differs")
        src = tar.extractfile(member)
        require(src is not None, "Cannot read archive member")
        digest = hashlib.sha256()
        total = 0
        with src:
            for block in iter(lambda: src.read(1024 * 1024), b""):
                total += len(block)
                digest.update(block)
        require(total == source_size and digest.hexdigest() == source_sha,
                "Archive member does not equal original legacy JSONL")
    return sha256(archive)


def archive_map(args, stamp, source_sha, source_size):
    directory = args.archive_dir
    directory.mkdir(parents=True, exist_ok=True)
    name = f"python-code-map-legacy-{args.version}-{stamp}.tar.gz"
    final = directory / name
    partial = directory / ("." + name + ".partial")
    require(not final.exists() and not partial.exists(), "Archive name collision")
    # Keep archive in same filesystem for exclusive hard-link publication.
    with partial.open("xb") as handle:
        with tarfile.open(fileobj=handle, mode="w:gz", format=tarfile.PAX_FORMAT) as tar:
            tar.add(args.legacy_map, arcname=args.legacy_relative_path, recursive=False)
        handle.flush()
        os.fsync(handle.fileno())
    archive_sha = validate_archive(partial, args.legacy_relative_path, source_sha, source_size)
    os.link(partial, final)  # FileExistsError rather than replacing any existing artifact.
    partial.unlink()  # Remove only the temporary unverified-name staging file.
    return final, archive_sha


def restore_legacy_from_archive(args, archive, source_sha, source_size):
    if args.legacy_map.exists():
        require(sha256(args.legacy_map) == source_sha, "Rollback legacy map path was modified")
        return
    with tarfile.open(archive, "r:gz") as tar:
        member = tar.getmembers()[0]
        src = tar.extractfile(member)
        require(src is not None, "Unable to recover archived legacy file")
        with src, args.legacy_map.open("xb") as dest:
            for block in iter(lambda: src.read(1024 * 1024), b""):
                dest.write(block)
            dest.flush()
            os.fsync(dest.fileno())
        os.chmod(args.legacy_map, member.mode & 0o777)
    require(args.legacy_map.stat().st_size == source_size and
            sha256(args.legacy_map) == source_sha, "Legacy rollback hash verification failed")


def run(args):
    relative = Path(args.legacy_relative_path)
    require(not relative.is_absolute() and ".." not in relative.parts and "." not in relative.parts,
            "Invalid legacy Code Map relative path")
    require(not args.legacy_map.is_symlink(), "Legacy map must not be a symlink")
    require(args.legacy_map.resolve() == (args.root / args.legacy_relative_path).resolve(),
            "Legacy map absolute and relative paths disagree")
    stats = validate_ground_truth(args)
    legacy = validate_legacy_map(args)
    paths, docs, targets = validate_manifests(args)
    originals = [p.read_bytes() for p in paths]
    new_raw = [originals[i] if targets[i] == docs[i] else json_bytes(targets[i])
               for i in range(len(paths))]
    changed = [a != b for a, b in zip(originals, new_raw)]
    timestamp = datetime.now(ZoneInfo(args.timezone))
    info = {"version": args.version, "mode": args.mode,
            "chicago_time": timestamp.isoformat(),
            "ground_truth": str(args.ground_truth), "ground_truth_validation": stats,
            "legacy_code_map": str(args.legacy_map), "legacy_validation": legacy,
            "manifests": [str(p) for p in paths],
            "before_manifest_sha256": [data_sha(raw) for raw in originals],
            "would_change_manifests": changed,
            "java_unchanged": True, "embedded_code_is_authoritative": True,
            "remove_python_keys": list(REMOVED_PYTHON_KEYS)}
    if args.mode == "preflight":
        info["status"] = "H10_V3_PREFLIGHT_PASS_NO_CHANGES"
        print(json.dumps(info, indent=2, ensure_ascii=False))
        print(info["status"])
        return info
    require(args.mode == "apply", "Invalid execution mode")
    stamp = timestamp.strftime("%Y%m%d-%H%M%S-%f") + "-" + str(os.getpid())
    archive, archive_sha = archive_map(args, stamp, legacy["code_map_sha256"], legacy["bytes"])
    info["archive"] = str(archive)
    info["archive_sha256"] = archive_sha
    info["archive_member"] = args.legacy_relative_path
    # Before any deletion, two exclusive durable backups are created.
    args.backup_dir.mkdir(parents=True, exist_ok=True)
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    backups = [args.backup_dir / f"{p.name}.before-python-legacy-retirement-{stamp}"
               for p in paths]
    audit = args.audit_dir / f"python-legacy-code-map-retired-{stamp}.json"
    require(all(not f.exists() for f in backups + [audit]), "Backup/audit collision")
    for dst, original in zip(backups, originals):
        exclusive_write(dst, original)
    info["backups"] = [str(p) for p in backups]
    info["audit_file"] = str(audit)
    try:
        require(sha256(args.legacy_map) == legacy["code_map_sha256"],
                "Legacy source changed after verified archive creation")
        for path, raw in zip(paths, new_raw):
            atomic_replace(path, raw)
        check = [load_json(path) for path in paths]
        for target, seen in zip(targets, check):
            require(target == seen, "Post-write manifest mismatch")
            require(all(k not in seen["languages"]["python"] for k in REMOVED_PYTHON_KEYS),
                    "Legacy Code Map metadata still present")
        require(validate_archive(archive, args.legacy_relative_path,
                                 legacy["code_map_sha256"], legacy["bytes"]) == archive_sha,
                "Archive changed before unlink")
        # Destructive operation is LAST, after backups, archive verification, and manifest verification.
        args.legacy_map.unlink()
        require(not args.legacy_map.exists(), "Legacy JSONL still exists after unlink")
        info["after_manifest_sha256"] = [sha256(path) for path in paths]
        info["legacy_active_path_removed"] = True
        info["status"] = "H10_V3_APPLY_PASS_ARCHIVED_AND_RETIRED"
        exclusive_write(audit, json_bytes(info))
    except BaseException:
        # Recover both live manifests and the legacy code map on any incomplete apply.
        for path, raw in zip(paths, originals):
            if path.read_bytes() != raw:
                atomic_replace(path, raw)
        restore_legacy_from_archive(args, archive, legacy["code_map_sha256"], legacy["bytes"])
        raise
    print(json.dumps(info, indent=2, ensure_ascii=False))
    print(info["status"])
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("preflight", "apply"), required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--root", type=Path, required=True)
    for name in ("ground-truth", "legacy-map", "manifest", "manifest-details",
                 "archive-dir", "backup-dir", "audit-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-gt-sha256", "expected-legacy-sha256",
                 "expected-gt-status", "positive-filename", "negative-filename",
                 "gt-relative-path", "legacy-relative-path", "timezone"):
        p.add_argument("--" + name, required=True)
    for name in ("expected-pairs", "expected-positive", "expected-negative",
                 "expected-projects", "expected-legacy-code-ids"):
        p.add_argument("--" + name, type=int, required=True)
    args = p.parse_args()
    require(args.expected_positive + args.expected_negative == args.expected_pairs,
            "Class counts do not sum to total")
    require(args.expected_pairs > 0 and args.expected_legacy_code_ids > 0,
            "Invalid required count")
    require(args.ground_truth.resolve() == (args.root / args.gt_relative_path).resolve(),
            "Ground Truth absolute and relative paths disagree")
    run(args)


if __name__ == "__main__":
    main()
