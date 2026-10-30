#!/usr/bin/env python3
"""Audit project identifiers and cross-split project overlap without modifying datasets."""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ID = re.compile(r'^(.+?)_(\d+)_(\d+)$')
FIELDS = ['dataset','split','project','pair_count','same_project_pairs','cross_project_pairs','test_project_name_match','id_parse_method']


def read_codes(path):
    d = {}
    with path.open(encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():
                o = json.loads(line)
                d[str(o['idx'])] = o
    return d


def read_test(path):
    with path.open(encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():
                o = json.loads(line)
                yield str(o['id'])


def parse_id(key):
    m = PROJECT_ID.fullmatch(key)
    return (m.group(1), 'prefix_two_numeric_suffixes') if m else (None, 'unresolved')


def write_csv(path, fieldnames, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    parser.add_argument('--training', type=Path, default=root/'data/training')
    parser.add_argument('--evaluation', type=Path, default=root/'data/evaluation')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--near-csv', type=Path, default=None, help='Prior A-09 near_duplicate_candidates.csv (optional)')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not (args.training/'manifest.json').is_file():
        raise FileNotFoundError(args.training/'manifest.json')
    test_project_names = defaultdict(set)
    test_id_projects = defaultdict(Counter)
    test_unresolved = defaultdict(list)
    for lang in ('java', 'python'):
        for fn in sorted((args.evaluation/'ground_truth'/lang).glob('pred_*_with_refactorability.csv')):
            project = fn.name[len('pred_'):-len('_with_refactorability.csv')]
            test_project_names[lang].add(project.casefold())
        for key in read_test(args.evaluation/'code'/f'{lang}.jsonl'):
            project, method = parse_id(key)
            if project:
                test_id_projects[lang][project] += 1
            elif len(test_unresolved[lang]) < 15:
                test_unresolved[lang].append(key)
    project_rows = []
    per_corpus = {}
    corpus_id_projects = {}
    corpus_counters = {}
    for ds in ('java_source', 'java_mixed', 'python_mixed'):
        base = args.training/'clean'/ds
        codes = read_codes(base/'data.jsonl')
        project_by_id = {k:parse_id(k)[0] for k in codes}
        corpus_id_projects[ds] = project_by_id
        language = 'python' if ds == 'python_mixed' else 'java'
        # The Python mixed corpus intentionally includes Java BCB source IDs.
        # Compare all names against BOTH held-out project sets, not just corpus name.
        all_heldout = set().union(*test_project_names.values())
        partitions = {}
        for split, filename in (('train','train.txt'), ('validation','valid.txt')):
            counts = Counter(); same = Counter(); cross = Counter(); label = Counter()
            n = missing = unresolved = 0
            ids = set(); examples = []; project_pairs = Counter()
            with (base/filename).open(encoding='utf-8-sig') as f:
                for line_no, line in enumerate(f,1):
                    if not line.strip(): continue
                    p = line.split()
                    if len(p) != 3 or p[2] not in ('0','1'):
                        raise ValueError(f'Invalid pair at {base/filename}:{line_no}')
                    a,b,y = p; n += 1; label[y] += 1
                    if a not in codes or b not in codes:
                        missing += 1;continue
                    ids.update((a,b))
                    pa,pb = project_by_id[a],project_by_id[b]
                    if not pa or not pb: unresolved += 1
                    for proj in set((pa,pb)) - {None}:
                        counts[proj] += 1
                        if proj.casefold() in all_heldout and len(examples) < 15:
                            examples.append({'pair_line':line_no,'id_a':a,'id_b':b,'project_a':pa,'project_b':pb,'label':y})
                    if pa and pb:
                        if pa == pb: same[pa] += 1
                        else:
                            cross[pa] += 1; cross[pb] += 1
                        project_pairs[(pa,pb) if pa <= pb else (pb,pa)] += 1
            splitprojects = set(counts)
            partitions[split] = {'pairs':n,'labels':dict(label),'missing_ids':missing,'pairs_with_unresolved_project_endpoint':unresolved,'referenced_functions':len(ids),'unresolved_referenced_functions':sum(project_by_id[k] is None for k in ids), 'parsed_project_names':len(splitprojects),'projects_matching_heldout_names':sorted(p for p in splitprojects if p.casefold() in all_heldout), 'matching_project_pair_rows':sum(v for p,v in counts.items() if p.casefold() in all_heldout),'example_matches':examples}
            for project, total in counts.most_common():
                project_rows.append({'dataset':ds,'split':split,'project':project,'pair_count':total,'same_project_pairs':same[project],'cross_project_pairs':cross[project],'test_project_name_match':int(project.casefold() in all_heldout),'id_parse_method':'prefix_two_numeric_suffixes'})
            print(f'[A-09] {ds}/{split}: pairs={n} unresolved_endpoint_pairs={unresolved} heldout_name_match_pair_incidences={partitions[split]["matching_project_pair_rows"]}',flush=True)
        train_projects = {x['project'] for x in project_rows if x['dataset']==ds and x['split']=='train'}
        valid_projects = {x['project'] for x in project_rows if x['dataset']==ds and x['split']=='validation'}
        per_corpus[ds] = {'splits':partitions,'shared_project_names_train_validation':sorted(train_projects & valid_projects), 'project_names_train':len(train_projects),'project_names_validation':len(valid_projects),'function_ids_parsed':sum(x is not None for x in project_by_id.values()),'function_ids_total':len(project_by_id)}
    write_csv(args.out/'project_pair_counts.csv',FIELDS,project_rows)
    evidence = []
    near_file = args.near_csv
    if near_file is None:
        near_files = sorted((root/'results/a-09').glob('*/near_duplicate_candidates.csv'),key=lambda p:p.stat().st_mtime,reverse=True)
        near_file = near_files[0] if near_files else None
    near_stats = Counter()
    if near_file and near_file.exists():
        with near_file.open(newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                testproj,method = parse_id(row['test_id'])
                examples = row.get('training_id_examples','').split(';')
                parsed = []
                for entry in examples:
                    if ':' not in entry: continue
                    ds,tid = entry.split(':',1)
                    proj,_ = parse_id(tid)
                    parsed.append((ds,tid,proj))
                matches = [(ds,tid,proj) for ds,tid,proj in parsed if testproj and proj and proj.casefold() == testproj.casefold()]
                near_stats['candidate_rows'] += 1
                if matches: near_stats['candidate_rows_matching_project_prefix'] += 1
                if not parsed or any(x[2] is None for x in parsed): near_stats['rows_with_unresolved_training_id_examples'] += 1
                for ds,tid,proj in matches:
                    near_stats['matched_examples'] += 1
                    evidence.append({'test_language':row['test_language'],'test_id':row['test_id'],'test_project':testproj,'training_dataset':ds,'training_id':tid,'training_project':proj,'token_set_jaccard':row['token_set_jaccard'],'test_hash':row['test_code_hash'],'training_hash':row['training_code_hash']})
        write_csv(args.out/'near_duplicate_same_project.csv',['test_language','test_id','test_project','training_dataset','training_id','training_project','token_set_jaccard','test_hash','training_hash'],evidence)
    else:
        write_csv(args.out/'near_duplicate_same_project.csv',['test_language','test_id','test_project','training_dataset','training_id','training_project','token_set_jaccard','test_hash','training_hash'],[])
    out = {'status':'REVIEW_REQUIRED','training_approval':'NOT_GRANTED_BY_A09','scope':'identifier-based project attribution diagnostic only; NO removal or staging','parse_rule':'^(.+?)_(\\d+)_(\\d+)$; project = first group','limits':['Parsed prefixes are HYPOTHESES, not verified GitHub repository provenance.','Unparsed numeric IDs (often BigCloneBench) have no defensible project inference.','Same-prefix matches do not by themselves prove extraction leakage; verify original ID-generation logic and split policy.','Near-duplicate input is approximate and incomplete; examples contain at most several training IDs per code hash.','Same-project pairs versus cross-project pairs are inferred from ID prefixes, not repository manifests.','No code, pair, or model data modified.'],'evaluation':{'ground_truth_project_names':{k:sorted(v) for k,v in test_project_names.items()},'test_ids_parsed':{k:sum(test_id_projects[k].values()) for k in ('java','python')},'test_id_project_counts':{k:dict(v) for k,v in test_id_projects.items()},'test_id_unresolved_examples':dict(test_unresolved)},'corpora':per_corpus,'near_duplicate_evidence':{'source':str(near_file) if near_file and near_file.exists() else None,**dict(near_stats)},'next_actions':['Verify naming convention by inspecting original data construction scripts or extraction manifests.','Establish whether unseen means function, clone pair, or repository-level holdout.','Use only verified metadata to authorize a project-disjoint split; keep current clean corpus untouched.']}
    (args.out/'audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('[A-09] Project ID diagnostic completed. No data changed; training approval still pending.',flush=True)

if __name__ == '__main__':
    main()
