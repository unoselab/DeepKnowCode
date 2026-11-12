#!/usr/bin/env python3
"""F-01c v22 explicit, non-GPU approval of the byte-identical v17 frozen dataset."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import f01c_label_review as lr


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))


# Immutable runtime set: unrelated/diagnostic .py files must not invalidate approval.
# Includes every local module imported on the approved smoke/train/worker path,
# and both executable shell entrypoints. Adding another runtime import requires
# deliberately updating this list and issuing a new explicit approval.
TRAINING_RUNTIME_FILES = (
    'src/py2/f-01c-train-standard-mixed-lora.py',
    'src/py2/f01c_core.py',
    'src/py2/f01c_shared.py',
    'src/py2/f01c_variants.py',
    'src/py2/f01c_comments.py',
    'src/py2/f01c_label_review.py',
    'src/py2/f01c_training_approval.py',
    'src/sh2/f-01c-train-standard-mixed-lora.sh',
    'src/sh2/f-01c-approve-training.sh',
)


def implementation_hashes():
    root=Path(__file__).resolve().parents[2]
    return {name:digest(root/name) for name in TRAINING_RUNTIME_FILES}


def check_implementation_hashes(expected):
    actual=implementation_hashes()
    # A v21 approval can include unrelated diagnostic files; compare only
    # the runtime files that v22 explicitly pins, without accepting any
    # modifications to approved runtime bytes.
    missing=sorted(set(actual)-set(expected))
    changed=sorted(name for name in actual if name in expected and actual[name]!=expected[name])
    if missing or changed:
        raise ValueError('Training implementation SHA256 mismatch; missing_pins=' + repr(missing) +
                         '; modified_runtime_files=' + repr(changed) +
                         '. Reinstall an unchanged runtime or issue a new explicit approval. ' +
                         'Do not bypass the frozen dataset or review checks.')


def check_approval(prepared, report):
    """Revalidate every approval pin at entrypoint AND worker; never mutate source."""
    prepared=Path(prepared).resolve();report=Path(report).resolve()
    if not report.is_file():raise ValueError('Approval report missing: '+str(report))
    approval=load(report)
    if approval.get('status')!='F01C_V21_EXPLICITLY_APPROVED' or approval.get('approved_for_training') is not True:
        raise ValueError('Training approval explicitly denied or incomplete')
    if approval.get('source_prepared_run')!=str(prepared):raise ValueError('Prepared run identity mismatch')
    check_implementation_hashes(approval.get('implementation_sha256') or {})
    paths={name:prepared/'prepared'/name for name in ('manifest.json','train.jsonl','validation.jsonl')}
    for name,p in paths.items():
        if not p.is_file() or digest(p)!=approval.get('prepared_sha256',{}).get(name):
            raise ValueError('Frozen prepared bytes mismatch: '+name)
    m=load(paths['manifest.json'])
    if m.get('status')!='PREPARED_PENDING_LABEL_REVIEW' or m.get('approved_for_training') is not False:
        raise ValueError('Original prepared manifest unexpectedly mutated')
    if m.get('sample',{}).get('train',{}).get('sha256')!=approval['prepared_sha256']['train.jsonl'] or m['sample']['validation']['sha256']!=approval['prepared_sha256']['validation.jsonl']:
        raise ValueError('Prepared manifest sample hashes mismatch')
    if m['sample']['train']['total']!=19464 or m['sample']['validation']['total']!=512:
        raise ValueError('Frozen prepared counts mismatch')
    evidence=approval.get('review_evidence',{})
    rp=Path(evidence.get('verification_report',''))
    dp=Path(evidence.get('decision_file',''))
    if not rp.is_file() or not dp.is_file() or digest(rp)!=evidence.get('verification_sha256') or digest(dp)!=evidence.get('decision_sha256'):
        raise ValueError('Review evidence files missing or changed')
    verify_review(prepared,dp,rp,approval['prepared_sha256'])
    return paths,approval,m


def verify_review(prepared, decisions, verification, hashes):
    r=load(verification);d=load(decisions)
    if r.get('status')!='F01C_V20_LABEL_REVIEW_ALL_NON_CLONE' or r.get('label_review_complete') is not True:
        raise ValueError('Human review did not pass')
    if r.get('approved_for_training') is not False or r.get('non_clone_count')!=8 or r.get('selected_pairs')!=8 or r.get('uncertain_or_clone_count')!=0:
        raise ValueError('Review report does not prove exactly 8 NON_CLONE labels')
    if r.get('prepared_sha256')!=hashes or r.get('decisions_sha256')!=digest(decisions):
        raise ValueError('Review report dataset/decisions hash mismatch')
    m,sel,actual=lr.inspect(prepared,hashes['train.jsonl'],hashes['validation.jsonl'],hashes['manifest.json'],8,2048)
    expected=sorted((lr.keys(row) for row in sel),key=lambda x:(x['id1'],x['id2']))
    observed=d.get('decisions')
    if not isinstance(observed,list) or len(observed)!=8:
        raise ValueError('Decision count mismatch')
    if d.get('prepared_sha256')!=hashes or d.get('train_pair_ids_sha256')!=m['sample']['train']['pair_ids_sha256']:
        raise ValueError('Decision dataset identity mismatch')
    for e,a in zip(expected,observed):
        if any(a.get(k)!=v for k,v in e.items()) or a.get('decision')!='NON_CLONE' or not isinstance(a.get('reason'),str) or not a['reason'].strip():
            raise ValueError('Invalid or unreviewed transformed pair: '+e['id1'])
    return len(sel)


def approve(prepared,decisions,verification,output,expected):
    prepared=prepared.resolve();decisions=decisions.resolve();verification=verification.resolve();output=output.resolve()
    paths={name:prepared/'prepared'/name for name in ('manifest.json','train.jsonl','validation.jsonl')}
    hashes={name:digest(path) for name,path in paths.items()}
    if hashes!=expected:raise ValueError('Frozen SHA256 differs from Shell pins')
    n=verify_review(prepared,decisions,verification,hashes)
    output.mkdir(parents=True,exist_ok=False)
    record={'version':'v21','status':'F01C_V21_EXPLICITLY_APPROVED','approved_for_training':True,
            'approval_scope':['smoke','train'],'created_at_chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat(),
            'source_prepared_run':str(prepared),'prepared_sha256':hashes,'transformed_reviewed':n,
            'implementation_sha256':implementation_hashes(),
            'review_evidence':{'decision_file':str(decisions),'decision_sha256':digest(decisions),
                               'verification_report':str(verification),'verification_sha256':digest(verification)},
            'note':'Explicit user-invoked approval, no GPU launched; source files remain immutable.'}
    dst=output/'training_approval.json'
    with dst.open('x',encoding='utf-8') as f:json.dump(record,f,sort_keys=True,indent=2);f.write('\n')
    check_approval(prepared,dst)
    print(json.dumps({'status':record['status'],'report':str(dst),'approval_sha256':digest(dst),'approved_for_training':True}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('approve','check'))
    for name in ('prepared-run','decisions','verification','approval-report','output'):
        p.add_argument('--'+name,required=True,type=Path)
    for name in ('expected-train-sha256','expected-validation-sha256','expected-manifest-sha256'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    expected={'train.jsonl':a.expected_train_sha256,'validation.jsonl':a.expected_validation_sha256,'manifest.json':a.expected_manifest_sha256}
    if a.mode=='approve':approve(a.prepared_run,a.decisions,a.verification,a.output,expected)
    else:
        paths,approval,_=check_approval(a.prepared_run,a.approval_report)
        if approval['prepared_sha256']!=expected:raise ValueError('Shell frozen hashes differ from approval')
        print(json.dumps({'status':'F01C_V21_APPROVAL_CHECK_PASS','report':str(a.approval_report),'approved_for_training':True}))


if __name__=='__main__':main()
