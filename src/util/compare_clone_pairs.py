#!/usr/bin/env python3
# =============================================================================
# DeepKnowCode - J-03c Clone Pair Independent Verification
# =============================================================================
#
# Script       : src/util/compare_clone_pairs.py
# Version      : v1
# Experiment   : J-03c - NiCad Clone Detection Before/After Preprocessing
# Verification : Independent Location-based Clone Pair Set Comparison
# Verified     : 2026-11-17 (America/Chicago)
#
# PURPOSE
# -------
# Verify that J-03b Java preprocessing preserves the clone pairs detected
# by NiCad7 across all 47 Java subject repositories.
#
# INPUTS
# ------
# Before : data/clone-detection/1before-preproc/{subject}/clone_pairs.csv
# After  : data/clone-detection/2after-preproc/{subject}/clone_pairs.csv
#
# COMPARISON METHOD
# -----------------
# Each clone fragment is identified by:
#   (repository, repository-relative file path, start_line, end_line)
#
# Each clone pair is represented as an unordered pair of two fragments.
# The Before and After clone pair sets are compared independently
# for each subject repository.
#
# Excluded from comparison:
#   - pair_id and fragment_id
#   - CSV row ordering
#   - source_sha256 and xml_sha256
#   - Other metadata unrelated to clone fragment locations
#
# PASS CRITERIA
# -------------
#   1. Before and After subject directory sets are identical.
#   2. Added clone location pairs = 0 for every subject.
#   3. Removed clone location pairs = 0 for every subject.
#   4. No comparison errors occur.
#
# EXECUTION
# ---------
# Run from the DeepKnowCode project root:
#
#   python src/util/compare_clone_pairs.py
#
# VERIFIED RESULTS (2026-11-17, America/Chicago)
# ------------------------------------------------
# Subject repositories : 47 / 47 PASS
# Before clone pairs   : 3,790,008 (J-03c reported total)
# After clone pairs    : 3,790,008 (J-03c reported total)
# Added location pairs : 0
# Removed location pairs: 0
# Comparison failures  : 0
#
# Final execution output:
#
#   FINAL: PASS subjects=47 failures=0
#
# CONCLUSION
# ----------
# Independent CSV-based verification confirmed that J-03b preprocessing
# preserved the detected clone pair location sets across all 47 subject
# repositories under the evaluated NiCad7 configuration.
#
# NOTE
# ----
# This verifies clone pair location identity, not byte-level equality
# of the source code, clone pair CSV files, or NiCad XML files.
#
# The verification uses Python sets, so duplicate rows are not checked
# for multiplicity. Run from the project root for correct relative paths.
#
# =============================================================================

import csv
from pathlib import Path

ROOT = Path("data/clone-detection")
BEFORE = ROOT / "1before-preproc"
AFTER = ROOT / "2after-preproc"

def load_pairs(path):
    pairs = set()
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            a = (r["repository_a"], r["file_a"],
                 r["start_line_a"], r["end_line_a"])
            b = (r["repository_b"], r["file_b"],
                 r["start_line_b"], r["end_line_b"])
            pairs.add(tuple(sorted((a, b))))
    return pairs

subjects = sorted(p.name for p in BEFORE.iterdir() if p.is_dir())
failures = 0

for subject in subjects:
    try:
        b = load_pairs(BEFORE / subject / "clone_pairs.csv")
        a = load_pairs(AFTER / subject / "clone_pairs.csv")
        added, removed = len(a - b), len(b - a)
        status = "PASS" if added == removed == 0 else "FAIL"
    except (OSError, KeyError, csv.Error) as e:
        print(f"FAIL {subject}: {e}")
        failures += 1
        continue

    failures += (status == "FAIL")
    print(f"{status} {subject}: before={len(b)} "
          f"after={len(a)} added={added} removed={removed}")

if set(subjects) != {p.name for p in AFTER.iterdir() if p.is_dir()}:
    print("FAIL: Subject directory sets differ")
    failures += 1

print(f"\nFINAL: {'PASS' if failures == 0 else 'FAIL'} "
      f"subjects={len(subjects)} failures={failures}")
raise SystemExit(1 if failures else 0)

'''
python src/util/compare_clone_pairs.py

PASS apache__activemq: before=8579 after=8579 added=0 removed=0
PASS apache__ant: before=440 after=440 added=0 removed=0
PASS apache__ant-ivy: before=194 after=194 added=0 removed=0
PASS apache__camel: before=285106 after=285106 added=0 removed=0
PASS apache__cassandra: before=1351 after=1351 added=0 removed=0
PASS apache__cayenne: before=1166 after=1166 added=0 removed=0
PASS apache__commons-lang: before=768 after=768 added=0 removed=0
PASS apache__commons-math: before=538 after=538 added=0 removed=0
PASS apache__cxf: before=5524 after=5524 added=0 removed=0
PASS apache__derby: before=5915 after=5915 added=0 removed=0
PASS apache__dubbo: before=1878 after=1878 added=0 removed=0
PASS apache__flink: before=139967 after=139967 added=0 removed=0
PASS apache__groovy: before=1999 after=1999 added=0 removed=0
PASS apache__hadoop: before=31854 after=31854 added=0 removed=0
PASS apache__hbase: before=109109 after=109109 added=0 removed=0
PASS apache__hive: before=358975 after=358975 added=0 removed=0
PASS apache__ignite: before=1720853 after=1720853 added=0 removed=0
PASS apache__jackrabbit: before=3839 after=3839 added=0 removed=0
PASS apache__jmeter: before=617 after=617 added=0 removed=0
PASS apache__kafka: before=4544 after=4544 added=0 removed=0
PASS apache__logging-log4j2: before=895 after=895 added=0 removed=0
PASS apache__lucene: before=6367 after=6367 added=0 removed=0
PASS apache__maven: before=1655 after=1655 added=0 removed=0
PASS apache__openjpa: before=8697 after=8697 added=0 removed=0
PASS apache__pdfbox: before=355 after=355 added=0 removed=0
PASS apache__pig: before=2103 after=2103 added=0 removed=0
PASS apache__poi: before=8539 after=8539 added=0 removed=0
PASS apache__storm: before=76089 after=76089 added=0 removed=0
PASS apache__synapse: before=624 after=624 added=0 removed=0
PASS apache__tika: before=434 after=434 added=0 removed=0
PASS apache__tomcat: before=2008 after=2008 added=0 removed=0
PASS apache__wicket: before=289 after=289 added=0 removed=0
PASS apache__xerces-j: before=1510 after=1510 added=0 removed=0
PASS apache__zookeeper: before=432 after=432 added=0 removed=0
PASS argouml-tigris-org__argouml: before=11531 after=11531 added=0 removed=0
PASS elastic__elasticsearch: before=492487 after=492487 added=0 removed=0
PASS google__gson: before=67 after=67 added=0 removed=0
PASS jedit-editor__jedit: before=814 after=814 added=0 removed=0
PASS jfree__jfreechart: before=6444 after=6444 added=0 removed=0
PASS jodaorg__joda-time: before=1328 after=1328 added=0 removed=0
PASS jruby__jruby: before=1137 after=1137 added=0 removed=0
PASS libgdx__libgdx: before=160689 after=160689 added=0 removed=0
PASS mockito__mockito: before=73 after=73 added=0 removed=0
PASS netty__netty: before=1946 after=1946 added=0 removed=0
PASS prestodb__presto: before=314892 after=314892 added=0 removed=0
PASS reactivex__rxjava: before=3070 after=3070 added=0 removed=0
PASS wrandelshofer__jhotdraw: before=2317 after=2317 added=0 removed=0

FINAL: PASS subjects=47 failures=0
(deepknowcode) oisit-r158r4l01:deepknowcode$ 
'''