#!/usr/bin/env python3
"""D-05g v1: read-only diagnostics on D-05c frozen binary predictions.
Input: D-05c output directory or ZIP, with SHA256-verified manifest.
Never reports similarity metrics not present in the source artifacts.
"""
import argparse,collections,csv,hashlib,io,json,math,os,sys,zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION='v1'
MODELS=('M0','M1','M2','M3')
REQUIRED=('aligned_pairs.jsonl', 'preflight.json', 'summary.json')+tuple(f'predictions_{m}.csv' for m in MODELS)
LENGTH_BINS=[(0,256,'0-255'),(256,512,'256-511'),(512,1024,'512-1023'),(1024,2048,'1024-2047'),(2048,4096,'2048-4095'),(4096,float('inf'),'4096+')]

def fail(msg): raise ValueError('D-05g v1 BLOCKED: '+str(msg))
def sha(data): return hashlib.sha256(data).hexdigest()
def stamp(): return datetime.now(ZoneInfo('America/Chicago')).isoformat()
def dumps(o): return (json.dumps(o,ensure_ascii=False,indent=2,sort_keys=True)+'\n').encode('utf-8')
def jsonread(data,label):
    try: return json.loads(data.decode('utf-8'))
    except Exception as e: fail(f'{label}: invalid JSON: {e}')
def source_open(path):
    p=Path(path)
    if p.is_dir():
        def read(name):
            fp=p/name
            if not fp.is_file():fail('missing '+str(fp))
            return fp.read_bytes()
        return read, {'kind':'directory','path':str(p.resolve())}, None
    if p.is_file() and p.suffix.lower()=='.zip':
        z=zipfile.ZipFile(p)
        if z.testzip() is not None: fail('corrupt ZIP CRC')
        names=z.namelist()
        matches=[n for n in names if n.endswith('/manifest.json') and n.count('/')==1]
        if len(matches)!=1:fail('expected one top-level directory with manifest.json')
        prefix=matches[0][:-len('manifest.json')]
        def read(name):
            key=prefix+name
            if key not in names: fail('missing ZIP member '+key)
            return z.read(key)
        return read,{'kind':'zip','path':str(p.resolve()),'prefix':prefix},z
    fail('input must be D-05c directory or ZIP: '+str(path))
def verify(read):
    raw=read('manifest.json'); manifest=jsonread(raw,'manifest.json')
    if not isinstance(manifest.get('files'),dict):fail('manifest missing files map')
    for name,meta in manifest['files'].items():
        if '/' in name or name.startswith('.'):fail('unsafe manifest path')
        b=read(name)
        if sha(b)!=meta.get('sha256') or len(b)!=meta.get('bytes'):fail('manifest hash/bytes mismatch '+name)
    for name in REQUIRED:
        if name not in manifest['files']:fail('unlisted required artifact '+name)
    return manifest,sha(raw)
def csvrows(data,name):
    reader=csv.DictReader(io.StringIO(data.decode('utf-8-sig')))
    if not reader.fieldnames:fail('empty CSV '+name)
    return list(reader)
def labelint(x,name):
    if str(x) not in ('0','1'):fail(f'invalid {name}: {x}')
    return int(x)
def binname(n):
    return next(label for lo,hi,label in LENGTH_BINS if lo<=n<hi)
def rate(num,den):return (num/den if den else None)
def counts(items):
    items=list(items)
    c=collections.Counter((y,p) for y,p in items)
    tp=c[1,1];tn=c[0,0];fp=c[0,1];fn=c[1,0]
    return {'n':len(items),'tp':tp,'tn':tn,'fp':fp,'fn':fn,'accuracy':rate(tp+tn,len(items)),
            'fpr':rate(fp,fp+tn),'fnr':rate(fn,fn+tp),'precision':rate(tp,tp+fp),'recall':rate(tp,tp+fn)}
def prep(read):
    manifest,msha=verify(read)
    pf=jsonread(read('preflight.json'),'preflight.json')
    summary=jsonread(read('summary.json'),'summary.json')
    if summary.get('status')!='D05C_V1_EVALUATION_COMPLETE':fail('D-05c not complete')
    if pf.get('pairs')!=7942:fail('unexpected D-05c pair count')
    pairs={}; project_counter=collections.Counter(); labels=collections.Counter()
    for idx,line in enumerate(read('aligned_pairs.jsonl').decode('utf-8').splitlines(),1):
        if not line.strip():continue
        r=json.loads(line); pid=r['pair_id'];pr=r['project'];y=labelint(r['label'],'label')
        if pid in pairs:fail('duplicate aligned pair '+pid)
        a,b=r['code_a'],r['code_b']
        if not isinstance(a,str) or not isinstance(b,str):fail('missing source code')
        la=len(a);lb=len(b)
        pairs[pid]={'pair_id':pid,'project':pr,'label':y,'code_a_characters':la,'code_b_characters':lb,
                    'combined_characters':la+lb,'max_side_characters':max(la,lb),
                    'max_side_length_bin':binname(max(la,lb)),
                    'identical_code_sha256':sha(a.encode('utf-8'))==sha(b.encode('utf-8')),
                    'code_a_sha256':sha(a.encode('utf-8')),'code_b_sha256':sha(b.encode('utf-8'))}
        project_counter[pr]+=1;labels[y]+=1
    if len(pairs)!=7942 or labels!={0:3971,1:3971} or len(project_counter)!=24:fail('unexpected aligned pair coverage')
    preds={}
    for model in MODELS:
        rows=csvrows(read(f'predictions_{model}.csv'),model)
        d={}
        for row in rows:
            pid=row['pair_id']
            if pid in d or pid not in pairs:fail('duplicate/unrecognized prediction '+model+' '+pid)
            y=labelint(row['true_label'],'true_label');p=labelint(row['predicted_label'],'prediction')
            if y!=pairs[pid]['label'] or row['project']!=pairs[pid]['project']:fail('label/project mismatch '+pid)
            try:margin=float(row['mean_logit_margin'])
            except Exception:fail('missing logit margin '+model)
            if not math.isfinite(margin) or p!=int(margin>0):fail('invalid margin/threshold '+model+' '+pid)
            d[pid]={'predicted_label':p,'margin':margin,'n_windows':int(row['n_windows']),
                    'original_input_tokens':int(row['original_input_tokens']),
                    'windowed':str(row['windowed']).lower()=='true'}
        if set(d)!=set(pairs):fail('prediction ID coverage mismatch '+model)
        ref=summary['model_metrics'][model]
        actual=counts((pairs[k]['label'],d[k]['predicted_label']) for k in pairs)
        for key in ['tp','tn','fp','fn','n']:
            if actual[key]!=ref[key]:fail('summary confusion mismatch '+model+' '+key)
        preds[model]=d
    ident=sum(p['label']==1 and p['identical_code_sha256'] for p in pairs.values())
    if ident!=85:fail('expected 85 identical-code Positive pairs; got '+str(ident))
    if any(p['label']==0 and p['identical_code_sha256'] for p in pairs.values()):fail('identical-code Negative observed')
    return pairs,preds,summary,msha,manifest

def writecsv(path,rows,fields):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def diagnostic(pairs,preds):
    ids=sorted(pairs)
    overall=[]; cases=[]; lengths=[]; project=[]; identical=[]; disagreement=[];concentration=[];windows=[]
    for m in MODELS:
        pred=preds[m]
        stats=counts((pairs[i]['label'],pred[i]['predicted_label']) for i in ids)
        overall.append({'model':m,**stats,'predicted_positive':stats['tp']+stats['fp'],
                        'predicted_negative':stats['tn']+stats['fn'],
                        'windowed_pairs':sum(pred[i]['windowed'] for i in ids)})
        for pid in ids:
            pair=pairs[pid];v=pred[pid];y=pair['label'];p=v['predicted_label']
            outcome=('TP' if y and p else 'FN' if y else 'FP' if p else 'TN')
            cases.append({'pair_id':pid,'project':pair['project'],'model':m,'true_label':y,
                          'predicted_label':p,'outcome':outcome,'mean_logit_margin':v['margin'],
                          'code_a_sha256':pair['code_a_sha256'],'code_b_sha256':pair['code_b_sha256'],
                          'code_a_characters':pair['code_a_characters'],'code_b_characters':pair['code_b_characters'],
                          'max_side_length_bin':pair['max_side_length_bin'],
                          'identical_code_sha256':int(pair['identical_code_sha256']),
                          'n_windows':v['n_windows'],'original_input_tokens':v['original_input_tokens'],'windowed':int(v['windowed'])})
        for pr in sorted({pairs[i]['project'] for i in ids}):
            ix=[i for i in ids if pairs[i]['project']==pr];c=counts((pairs[i]['label'],pred[i]['predicted_label']) for i in ix)
            project.append({'project':pr,'model':m,**c})
            concentration.append({'project':pr,'model':m,'pairs':len(ix),'total_errors':c['fp']+c['fn'],
                                  'false_positives':c['fp'],'false_negatives':c['fn'],
                                  'share_of_all_model_errors':rate(c['fp']+c['fn'],stats['fp']+stats['fn']),
                                  'share_of_all_pairs':len(ix)/len(ids)})
        for bn in [b[2] for b in LENGTH_BINS]:
            for y in (0,1):
                ix=[i for i in ids if pairs[i]['max_side_length_bin']==bn and pairs[i]['label']==y]
                c=counts((pairs[i]['label'],pred[i]['predicted_label']) for i in ix)
                lengths.append({'model':m,'max_side_length_bin':bn,'true_label':y,'n':len(ix),
                                'errors':c['fp']+c['fn'],'error_rate':rate(c['fp']+c['fn'],len(ix)),
                                'tp':c['tp'],'tn':c['tn'],'fp':c['fp'],'fn':c['fn']})
        for windowed in (False,True):
            for y in (0,1):
                ix=[i for i in ids if pred[i]['windowed']==windowed and pairs[i]['label']==y]
                c=counts((pairs[i]['label'],pred[i]['predicted_label']) for i in ix)
                windows.append({'model':m,'windowed':int(windowed),'true_label':y,'n':len(ix),
                                'errors':c['fp']+c['fn'],'error_rate':rate(c['fp']+c['fn'],len(ix))})
        ix=[i for i in ids if pairs[i]['label']==1 and pairs[i]['identical_code_sha256']]
        c=counts((pairs[i]['label'],pred[i]['predicted_label']) for i in ix)
        identical.append({'model':m,'n_identical_positive':len(ix),'detected_tp':c['tp'],
                          'missed_fn':c['fn'],'recall':rate(c['tp'],len(ix))})
    for pid in ids:
        r=pairs[pid];a=preds['M2'][pid];b=preds['M3'][pid];y=r['label']
        if a['predicted_label']!=b['predicted_label']:
            ac=a['predicted_label']==y;bc=b['predicted_label']==y
            disagreement.append({'pair_id':pid,'project':r['project'],'true_label':y,
                'M2_predicted':a['predicted_label'],'M3_predicted':b['predicted_label'],
                'M2_correct':int(ac),'M3_correct':int(bc),'winner':'M2' if ac else 'M3',
                'M2_margin':a['margin'],'M3_margin':b['margin'],
                'max_side_characters':r['max_side_characters'],'max_side_length_bin':r['max_side_length_bin'],
                'identical_code_sha256':int(r['identical_code_sha256']),
                'code_a_sha256':r['code_a_sha256'],'code_b_sha256':r['code_b_sha256']})
    wins=collections.Counter(x['winner'] for x in disagreement)
    if wins!={'M2':743,'M3':72}:fail('expected D-05f M2/M3 pair disagreement counts')
    return {'model_error_summary.csv':overall,'pair_error_cases.csv':cases,'project_error_metrics.csv':project,
            'project_error_concentration.csv':concentration,'length_bin_errors.csv':lengths,
            'windowing_errors.csv':windows,'identical_positive_detection.csv':identical,
            'm2_m3_disagreements.csv':disagreement},wins

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',required=True,help='D-05c run folder or ZIP (read-only)')
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    ap.add_argument('--out',type=Path,help='new output directory; existing paths forbidden')
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    read,provenance,z=source_open(args.input)
    try:
        pairs,preds,summary,msha,manifest=prep(read)
        results,wins=diagnostic(pairs,preds)
        baseline={'created_at_chicago':stamp(),'source':provenance,'source_manifest_sha256':msha,
                  'source_ground_truth_sha256':summary['ground_truth_sha256'],
                  'models':list(MODELS),'pairs':len(pairs),'projects':len(set(x['project'] for x in pairs.values())),
                  'identical_code_positive_pairs':85,
                  'm2_m3_disagreement':dict(wins),
                  'similarity_analysis_status':'NOT_AVAILABLE_IN_D05C_INPUT',
                  'similarity_reason':'No pairwise similarity score/metric provided in the frozen D-05c aligned pairs or predictions; no proxy score invented.',
                  'length_measure':'Python source character length (max side), not tokenizer tokens',
                  'limitations':['Descriptive post-hoc analysis; differences do not establish causality.',
                                 'Similarity values are unavailable; similarity-binned error rates intentionally not computed.',
                                 'Type-1 classification is not asserted: equal SHA256 means byte-identical code only.',
                                 'Code/text is excluded from output; hashes and lengths are retained.',
                                 'Balanced similarity-selected reviewer-confirmed labels are not population-prevalence estimates.',
                                 'Function and project dependencies can affect uncertainty.']}
        status='D05G_V1_PREFLIGHT_PASS_NOT_ANALYZED' if args.dry_run else 'D05G_V1_ERROR_ANALYSIS_COMPLETE'
        if args.dry_run:
            print(json.dumps({'status':status,'pairs':len(pairs),'projects':baseline['projects'],
                              'models':list(MODELS),'identical_code_positive_pairs':85,
                              'm2_m3_disagreement':dict(wins),'source_manifest_sha256':msha,
                              'similarity_analysis_status':baseline['similarity_analysis_status']},sort_keys=True))
            return
        out=args.out or args.root/'results/d-05g'/('python-error-analysis-v1-'+datetime.now(ZoneInfo('America/Chicago')).strftime('%Y%m%d-%H%M%S')+'-'+str(os.getpid()))
        if out.exists():fail('refuse overwrite '+str(out))
        if not out.parent.is_dir():out.parent.mkdir(parents=True,exist_ok=True)
        out.mkdir(exist_ok=False)
        try:
            for filename,rows in results.items():
                if not rows:fail('empty output '+filename)
                writecsv(out/filename,rows,list(rows[0]))
            summary_out={**baseline,'status':status,'counts':{'output_rows':{k:len(v) for k,v in results.items()},
                                  'M2_only_correct':wins['M2'],'M3_only_correct':wins['M3']},
                         'overall_errors':results['model_error_summary.csv'],
                         'identical_positive_detection':results['identical_positive_detection.csv']}
            (out/'analysis_summary.json').write_bytes(dumps(summary_out))
            files={p.name:{'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in sorted(out.iterdir()) if p.is_file()}
            (out/'manifest.json').write_bytes(dumps({'version':VERSION,'status':status,'created_at_chicago':stamp(),
                'source_manifest_sha256':msha,'files':files}))
        except BaseException:
            # Preserve partial directory for forensic recovery; never overwrite or delete.
            raise
        print(json.dumps({'status':status,'output':str(out),'pairs':len(pairs),
              'models':list(MODELS),'m2_m3_disagreements':len(results['m2_m3_disagreements.csv']),
              'identical_code_positive_pairs':85,'similarity_analysis_status':baseline['similarity_analysis_status']},sort_keys=True))
    finally:
        if z:z.close()
if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,FileNotFoundError,zipfile.BadZipFile) as e:
        print('D-05g v1 ERROR:',e,file=sys.stderr);sys.exit(2)
