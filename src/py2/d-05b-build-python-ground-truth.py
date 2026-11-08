#!/usr/bin/env python3
"""D-05b v4: assemble unverified 24-subject Python clone candidates.

Safeguards: never assign gold labels; no writes to training, repositories, or legacy;
exclusive output creation; C-01 is detection evidence, not independent annotation.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

STATUS = 'D05B_V4_CANDIDATES_ONLY_GOLD_LABELS_NOT_VERIFIED'
EXCLUDED = 'azure-sdk-for-python'
REPLACEMENT = 'scikit-learn'
CSVNAME = 'hybrid_function_clone_pairs.csv'
C01_COLUMNS = ('file_a','start_a','end_a','function_a','file_b','start_b','end_b','function_b')
LEGACY_COLUMNS = ('pair_left_func','pair_right_func')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

def jlines(path):
    with path.open(encoding='utf-8-sig') as f:
        for line_no,line in enumerate(f,1):
            if line.strip():
                try: obj=json.loads(line)
                except json.JSONDecodeError as e: raise ValueError(f'{path}:{line_no}: {e}') from e
                if not isinstance(obj,dict): raise ValueError(f'{path}:{line_no}: JSON object expected')
                yield obj

def write_json(path,obj):
    with path.open('x',encoding='utf-8') as f: json.dump(obj,f,indent=2,sort_keys=True,ensure_ascii=False); f.write('\n')

def git_commit(repository):
    p=subprocess.run(['git','-C',str(repository),'rev-parse','HEAD'], capture_output=True,text=True,timeout=20,check=False)
    if p.returncode!=0 or not re.fullmatch(r'[0-9a-f]{40}',p.stdout.strip()):
        raise ValueError('Cannot determine git HEAD for '+str(repository))
    return p.stdout.strip()

def discover_c01(root,projects):
    """Mirror C-01 result organizer: join <run-id>/project_run.json to hybrid CSV.

    Neither run directory names nor CSV basenames identify the project.
    Reject conflicting duplicate runs at the same timestamp and stale commits.
    """
    source=root/'results/c-01'
    if not source.is_dir(): raise ValueError('Missing original C-01 run directory: '+str(source))
    expected={p:git_commit(root/'data/repositories'/p) for p in sorted(projects)}
    completed=defaultdict(list)
    rejected=[]
    for meta_file in sorted(source.glob('*/project_run.json')):
        try: meta=json.loads(meta_file.read_text(encoding='utf-8'))
        except (OSError,json.JSONDecodeError) as exc: raise ValueError(f'Invalid C-01 metadata {meta_file}: {exc}') from exc
        project=meta.get('project'); status=meta.get('status'); commit=meta.get('commit')
        if project not in projects:
            rejected.append({'run':meta_file.parent.name,'project':project,'reason':'NOT_ACTIVE_PROJECT'})
            continue
        if status!='COMPLETED':
            rejected.append({'run':meta_file.parent.name,'project':project,'reason':'NOT_COMPLETED','status':status})
            continue
        if commit != expected[project]:
            rejected.append({'run':meta_file.parent.name,'project':project,'reason':'COMMIT_MISMATCH',
                             'recorded_commit':commit,'active_commit':expected[project]})
            continue
        csv_file=meta_file.parent/'hybrid'/CSVNAME
        if not csv_file.is_file() or csv_file.is_symlink():
            rejected.append({'run':meta_file.parent.name,'project':project,'reason':'HYBRID_CSV_MISSING'})
            continue
        completed[project].append((meta_file.parent.name,csv_file,meta_file,meta))
    selected={}; chosen_meta={}; alternatives={}
    for project,runs in sorted(completed.items()):
        runs.sort(key=lambda item:item[0])
        run_id,csv_file,meta_file,meta=runs[-1]
        selected[project]=csv_file
        chosen_meta[project]={'c01_run_id':run_id,'metadata_file':str(meta_file.relative_to(root)),
                              'metadata_sha256':sha(meta_file),'csv_sha256':sha(csv_file),
                              'git_commit':meta['commit'],'status':meta['status'],
                              'remote':meta.get('remote')}
        if len(runs)>1: alternatives[project]=[r[0] for r in runs[:-1]]
    return selected,chosen_meta,expected,rejected,alternatives

def strict_source_path(root,project,raw):
    """Safe, non-invented repository locator. No blind basename-only matches."""
    if not raw or os.path.isabs(raw) or '\\' in raw: return None
    if '\x00' in raw or re.match(r'^[A-Za-z]:',raw): return None
    relative=Path(raw)
    if '..' in relative.parts: return None
    base=(root/'data/repositories'/project).resolve()
    candidates=[base/relative]
    # Common staging records relative to the project directory.
    if relative.parts and relative.parts[0]==project:
        candidates.append(base.joinpath(*relative.parts[1:]))
    hits=[]
    for p in candidates:
        resolved=p.resolve()
        if resolved.is_relative_to(base) and resolved.is_file(): hits.append(resolved)
    return sorted(set(hits),key=str)[0] if len(set(hits))==1 else None

def load_code(root):
    path=root/'data/evaluation/code/python.jsonl'
    if not path.is_file(): raise ValueError(f'Missing {path}')
    result={}; normalized={}
    for obj in jlines(path):
        ident=obj.get('id'); code=obj.get('code')
        if not isinstance(ident,str) or not isinstance(code,str): raise ValueError('Invalid python code corpus id/code')
        if ident in result: raise ValueError('Duplicate python corpus ID: '+ident)
        result[ident]=hashlib.sha256(code.encode()).hexdigest()
        normalized[ident]=hashlib.sha256(''.join(code.split()).encode()).hexdigest()
    return path,result,normalized

def training_ids(root):
    paths=[]; ids=set(); pairs=set(); counts={}
    for split in ('train','validation'):
        file=root/'data/training/prepared/balanced_multilingual'/f'{split}.jsonl'
        if not file.is_file(): raise ValueError(f'Missing active training split {file}')
        n=0
        for item in jlines(file):
            if item.get('corpus')!='python_mixed': continue
            a,b=item.get('id1'),item.get('id2')
            if a is None or b is None: raise ValueError(f'Missing id1/id2 in {file}')
            a,b=str(a),str(b)
            ids.update((a,b)); pairs.add(tuple(sorted((a,b)))); n+=1
        counts[split]={'python_pairs':n,'path':str(file.relative_to(root)),'sha256':sha(file)}
    return ids,pairs,counts

def digest_key(kind,project,*parts):
    return hashlib.sha256(('python\0'+kind+'\0'+project+'\0'+'\0'.join(map(str,parts))).encode()).hexdigest()[:24]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path.cwd())
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--candidate-output',type=Path,required=True)
    args=ap.parse_args(); root=args.root.resolve(); out=args.out.resolve(); target=args.candidate_output.resolve()
    gt=root/'data/evaluation/ground_truth'
    legacy=gt/'python_legacy'
    if not legacy.is_dir(): raise ValueError(f'Missing legacy source {legacy}')
    if not target.is_dir() or any(target.iterdir()): raise ValueError('New output must be an existing EMPTY directory: '+str(target))
    if out.exists(): raise ValueError('Result path exists: '+str(out))
    if target==out or target in out.parents or out in target.parents: raise ValueError('Overlapping output paths')
    repo_root=root/'data/repositories'
    if not repo_root.is_dir(): raise ValueError('Missing data/repositories')
    active={p.name for p in repo_root.iterdir() if p.is_dir() and not p.name.startswith('.')}
    if EXCLUDED in active or REPLACEMENT not in active or len(active)!=24:
        raise ValueError(f'Expected exactly 24 active repositories including scikit-learn, excluding azure-sdk-for-python; found {len(active)}')
    if not (root/'data/repositories'/REPLACEMENT).is_dir(): raise ValueError('Missing scikit-learn repository')
    code_path,code,normal=load_code(root)
    tr_ids,tr_pairs,tr_files=training_ids(root)
    chosen,chosen_meta,repo_commits,rejected_runs,alternative_runs=discover_c01(root,active)
    missing=sorted(active-set(chosen))
    if missing:
        raise ValueError('C-01 COMPLETE 24/24 COVERAGE GATE FAILED, missing '+str(missing)+
                         '; inspect results/c-01/*/project_run.json and active repository HEAD. No output written.')
    ambiguous={}
    duplicate_same_hash=alternative_runs
    orphan=[x['run'] for x in rejected_runs]
    candidates=[]; project_stats=defaultdict(Counter); inputs={}; legacy_projects=set(); excluded=[]; reasons=Counter()
    for file in sorted(legacy.glob('pred_*_with_refactorability.csv')):
        proj=file.name[len('pred_'):-len('_with_refactorability.csv')]
        if proj not in active:
            with file.open(encoding='utf-8-sig',newline='') as f: count=sum(1 for _ in csv.DictReader(f))
            excluded.append({'project':proj,'rows':count,'path':str(file.relative_to(root)),'reason':'NOT_ACTIVE_SUBJECT'})
            continue
        legacy_projects.add(proj); inputs[str(file.relative_to(root))]=sha(file)
        dedupe=set()
        with file.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            if not set(LEGACY_COLUMNS).issubset(reader.fieldnames or []): raise ValueError(f'Invalid legacy schema: {file}')
            for lineno,row in enumerate(reader,2):
                a=(row.get('pair_left_func') or '').strip(); b=(row.get('pair_right_func') or '').strip()
                if not a or not b or a==b: reasons['invalid_legacy_function_pair']+=1; continue
                a,b=sorted((a,b))
                if (a,b) in dedupe: reasons['duplicate_legacy_pair']+=1; continue
                dedupe.add((a,b))
                blocking=[]
                if a not in code or b not in code: blocking.append('LEGACY_CODE_ID_UNRESOLVED')
                if a in tr_ids or b in tr_ids: blocking.append('ACTIVE_TRAIN_VALIDATION_FUNCTION_ID_OVERLAP')
                if (a,b) in tr_pairs: blocking.append('ACTIVE_TRAIN_VALIDATION_PAIR_ID_OVERLAP')
                if a in normal and b in normal and normal[a]==normal[b]: blocking.append('IDENTICAL_NORMALIZED_CODE_REVIEW')
                pred=(row.get('clone_predict') or '').strip()
                candidates.append({'pair_id':digest_key('legacy',proj,a,b),'project':proj,'language':'python','origin':'legacy_prediction_csv',
                                   'left_id':a,'right_id':b,'left_code_sha256':code.get(a),'right_code_sha256':code.get(b),
                                   'legacy_prediction':pred if pred in ('0','1') else None,
                                   'detection_source':None,'source_reference':{'file':str(file.relative_to(root)),'line':lineno},
                                   'gold_label':None,'label_source':None,'annotation_status':'pending','evaluation_eligible':False,
                                   'review_flags':blocking,'code_hash_train_overlap_status':'NOT_ESTABLISHED','near_duplicate_leakage_status':'NOT_CHECKED'})
                project_stats[proj]['legacy_pairs']+=1
    c01_unresolved=Counter(); c01_source_counts=Counter()
    for proj,file in sorted(chosen.items()):
        inputs[str(file.relative_to(root)) if file.is_relative_to(root) else str(file)]=sha(file)
        with file.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            if not set(C01_COLUMNS).issubset(reader.fieldnames or []): raise ValueError(f'C-01 CSV unexpected schema: {file} {reader.fieldnames}')
            seen=set()
            for lineno,row in enumerate(reader,2):
                left=tuple((row.get(k) or '').strip() for k in C01_COLUMNS[:4]); right=tuple((row.get(k) or '').strip() for k in C01_COLUMNS[4:])
                if not all(left[:3]) or not all(right[:3]): reasons['c01_invalid_locator']+=1; continue
                ends=sorted((left,right))
                if ends[0]==ends[1]: reasons['c01_self_pair']+=1; continue
                if tuple(ends) in seen: reasons['c01_duplicate_pair']+=1; continue
                seen.add(tuple(ends))
                # C-01 paths and line ranges are source locators. They are NOT code corpus IDs.
                # Only verify exact repository file existence, not functional/semantic equivalence.
                pa=strict_source_path(root,proj,ends[0][0]); pb=strict_source_path(root,proj,ends[1][0])
                flags=['C01_FUNCTION_CODE_LINK_UNVERIFIED','TRAIN_CODE_HASH_LEAKAGE_NOT_ESTABLISHED']
                if pa is None or pb is None:
                    flags.append('C01_REPOSITORY_SOURCE_PATH_UNRESOLVED'); c01_unresolved[proj]+=1
                if proj in ambiguous: flags.append('AMBIGUOUS_C01_SOURCE')
                source=(row.get('source') or '').strip()
                c01_source_counts[source]+=1
                candidates.append({'pair_id':digest_key('c01',proj,*ends[0],*ends[1]),'project':proj,'language':'python',
                                   'origin':'c01_hybrid_detector','left_id':None,'right_id':None,
                                   'left_locator':dict(zip(('file','start','end','function'),ends[0])),
                                   'right_locator':dict(zip(('file','start','end','function'),ends[1])),
                                   'left_repository_file':str(pa.relative_to(root)) if pa else None,
                                   'right_repository_file':str(pb.relative_to(root)) if pb else None,
                                   'detection_source':source,'legacy_prediction':None,
                                   'source_reference':{'file':str(file.relative_to(root)) if file.is_relative_to(root) else str(file),'line':lineno},
                                   'gold_label':None,'label_source':None,'annotation_status':'pending','evaluation_eligible':False,
                                   'review_flags':flags,'code_hash_train_overlap_status':'NOT_ESTABLISHED','near_duplicate_leakage_status':'NOT_CHECKED'})
                project_stats[proj]['c01_pairs']+=1
    if project_stats[REPLACEMENT]['c01_pairs']!=373: raise ValueError('scikit-learn expected 373 canonical pairs; got '+str(project_stats[REPLACEMENT]['c01_pairs']))
    git_sha=repo_commits[REPLACEMENT]
    now=datetime.now(ZoneInfo('America/Chicago')).isoformat()
    counts=Counter({'legacy_candidates':sum(1 for x in candidates if x['origin']=='legacy_prediction_csv'),
                    'c01_candidates':sum(1 for x in candidates if x['origin']=='c01_hybrid_detector'),
                    'total_candidate_records':len(candidates),'c01_projects_with_results':len(chosen),
                    'active_projects':len(active),'verified_gold_labels':0})
    reports={'status':STATUS,'created_at_chicago':now,'active_repositories':sorted(active),'excluded_subject':EXCLUDED,
             'replacement_subject':REPLACEMENT,'replacement_git_commit_expected':'4dc45d9a0afa48d741fe5924f167a236d8605210',
             'replacement_git_commit_actual':git_sha,
             'replacement_git_commit_match':(git_sha=='4dc45d9a0afa48d741fe5924f167a236d8605210') if git_sha else None,
             'scikit_learn_c01_expected_unique_hybrid_pairs_from_readme':373,
             'scikit_learn_c01_pair_count_matches_readme':(project_stats[REPLACEMENT]['c01_pairs']==373),
             'active_training_splits':tr_files,'source_code_corpus':{'path':str(code_path.relative_to(root)),'sha256':sha(code_path),'functions':len(code)},
             'source_file_sha256':inputs,'excluded_legacy_projects':excluded,'legacy_projects_present':sorted(legacy_projects),
             'c01_source_csv_by_project':{p:(str(v.relative_to(root)) if v.is_relative_to(root) else str(v)) for p,v in chosen.items()},
             'c01_completed_run_metadata':chosen_meta,'active_repository_commits':repo_commits,
             'c01_rejected_runs':rejected_runs,'c01_other_completed_runs':alternative_runs,
             'c01_missing_projects':sorted(active-set(chosen)),'c01_ambiguous_projects':ambiguous,
             'c01_duplicate_identical_sources':duplicate_same_hash,'c01_unmapped_source_files':orphan[:100],
             'c01_repository_path_unresolved_by_project':dict(c01_unresolved),'c01_detector_source_counts':dict(c01_source_counts),
             'project_pair_counts':{p:dict(project_stats[p]) for p in sorted(active)},'counts':dict(counts),
             'invalid_skipped':dict(reasons),'methodology':{
                 'candidate_sources':'Legacy prediction CSVs (active projects only) AND C-01 hybrid detector output; these are separate evidence sources and may overlap physically.',
                 'legacy_not_gold':'clone_predict is not a verified gold annotation',
                 'detector_not_gold':'C-01/B-07/C-02 detection and reproducibility do not establish binary ground truth',
                 'c01_locator':'C-01 file/start/end/function is NOT asserted to match evaluation/code/python.jsonl id',
                 'code_leakage':'ID overlap checks only for legacy rows against active M2/M3 splits; code-hash and near-duplicate leakage remain NOT_ESTABLISHED',
                 'no_negative_invention':'No negatives constructed from absent detection pairs',
                 'coverage':'All 24 C-01 project_run.json records must be COMPLETED, matching current repository HEAD, with hybrid CSV, before writing outputs',
                 'next_gate':'Resolve C-01-to-source-code mapping, full M1-M3 code-level leakage, independent binary annotation and project-stratified evaluation sample.'}}
    out.mkdir(parents=True,exist_ok=False)
    artifact=[]
    cand=out/'candidates.jsonl'
    with cand.open('x',encoding='utf-8') as f:
        for row in sorted(candidates,key=lambda r:(r['project'],r['origin'],r['pair_id'])):
            f.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+'\n')
    artifact.append(cand)
    queue=out/'annotation_queue.csv'
    with queue.open('x',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(('pair_id','project','origin','left_id','right_id','left_file','left_start','right_file','right_start','review_flags','gold_label','annotator','evidence','adjudication_status'))
        for row in sorted(candidates,key=lambda r:(r['project'],r['origin'],r['pair_id'])):
            la=row.get('left_locator') or {}; lb=row.get('right_locator') or {}
            writer.writerow((row['pair_id'],row['project'],row['origin'],row.get('left_id') or '',row.get('right_id') or '',la.get('file',''),la.get('start',''),lb.get('file',''),lb.get('start',''),';'.join(row['review_flags']),'','','','pending'))
    artifact.append(queue)
    audit=out/'provenance_leakage_audit.json'; write_json(audit,reports); artifact.append(audit)
    manifest=out/'manifest.json'; write_json(manifest,{'status':STATUS,'created_at_chicago':now,'source_files_sha256':inputs,'artifacts_sha256':{p.name:sha(p) for p in artifact}}); artifact.append(manifest)
    for file in artifact:
        with file.open('rb') as src,(target/file.name).open('xb') as dst:
            for data in iter(lambda:src.read(1<<20),b''): dst.write(data)
    print(json.dumps({'status':STATUS,'counts':dict(counts),'scikit_learn_c01_pairs':project_stats[REPLACEMENT]['c01_pairs'],
                      'c01_missing_projects':sorted(active-set(chosen)),'c01_ambiguous_projects':sorted(ambiguous),
                      'excluded_legacy_project_rows':sum(x['rows'] for x in excluded),
                      'out':str(out),'ground_truth_target':str(target)},ensure_ascii=False))

if __name__=='__main__':
    try: main()
    except (ValueError,OSError,KeyError) as e:
        print('D05B_V4_BLOCKED: '+str(e),file=sys.stderr); sys.exit(2)
