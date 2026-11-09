#!/usr/bin/env python3
"""D-05bh v1: source-validated offline HTML reviewer packets, no Gold labels."""
import argparse,csv,hashlib,io,json,os,re,sys,zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

V3='negative_annotation_packets.jsonl'; EXT='supplemental_negative_annotation_packets.jsonl'
def sha(b): return hashlib.sha256(b).hexdigest()
def fail(s): raise ValueError(s)
def resolve_input(root, explicit, prefix):
    if explicit: return Path(explicit).expanduser().resolve()
    d=root/'results'/'d-05b'
    cand=sorted((p for p in d.glob(prefix+'*') if p.is_dir()),key=lambda p:p.name)
    if not cand: fail('Input missing: supply --'+('v3' if 'negative-candidates' in prefix else 'supplement')+' PATH')
    return cand[-1]
def read_bundle(path,packet_name):
    if path.is_dir():
        pp=path/packet_name; mp=path/'manifest.json'
        raw=pp.read_bytes(); man=mp.read_bytes(); j=json.loads(man)
        for name,details in j.get('files',{}).items():
            data=(path/name).read_bytes()
            if sha(data)!=details['sha256'] or len(data)!=details['bytes']:fail('Manifest mismatch '+str(path/name))
    else:
        with zipfile.ZipFile(path) as z:
            bad=z.testzip()
            if bad:fail('Bad ZIP member '+bad)
            pp=next((n for n in z.namelist() if n.endswith('/'+packet_name)),None)
            mp=next((n for n in z.namelist() if n.endswith('/manifest.json') and n.rsplit('/',1)[0]==pp.rsplit('/',1)[0]),None) if pp else None
            if not pp or not mp:fail('Missing packet or manifest in '+str(path))
            raw=z.read(pp);man=z.read(mp);j=json.loads(man); base=mp.rsplit('/',1)[0]+'/'
            for name,details in j.get('files',{}).items():
                data=z.read(base+name)
                if sha(data)!=details['sha256'] or len(data)!=details['bytes']: fail('Manifest mismatch '+name)
    if sha(raw)!=j['files'][packet_name]['sha256']:fail('Packet hash differs from manifest')
    out=[json.loads(line) for line in raw.splitlines() if line.strip()]
    return out,sha(man)

def validate(rows,source):
    ids=set(); keys=set(); projects=set(); counts=Counter()
    for row in rows:
        pair=row['pair_id'];project=row['project'];projects.add(project)
        if pair in ids:fail('Duplicate pair_id '+pair)
        ids.add(pair)
        left,right=row['left'],row['right']
        pairkey=tuple(sorted((left['code_sha256'],right['code_sha256'])))
        if pairkey in keys:fail('Duplicate code pair '+pair)
        keys.add(pairkey)
        for side in (left,right):
            if sha(side['source_code'].encode('utf-8'))!=side['code_sha256']:fail('Code hash mismatch '+pair)
        if row.get('gold_label') is not None or row.get('evaluation_eligible') is not False or row.get('negative_label_assumed') is not False: fail('Nonpending candidate '+pair)
        if row.get('c01_detected_pair') is not False or row.get('active_training_validation_exact_normalized_overlap') is not False: fail('Excluded candidate slipped into '+pair)
        sim=float(row['pair_similarity'])
        if source=='v3' and not (0.1<=sim<0.5):fail('v3 out of similarity range '+pair)
        if source=='supplement' and (project!='paramiko' or not (0.05<=sim<0.1)):fail('Supplement outside paramiko and range '+pair)
        counts[row['similarity_bin']]+=1
    return ids,keys,projects,counts

def write_csv(path,rows,fields):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

# Untrusted source text is serialized as JSON into data.js, never interpolated into HTML markup.
# JS uses textContent to render source code, never innerHTML with source.
HTML=r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>D-05bh Negative Pair Review — REVIEWER_SLOT</title>
<style>
:root{color-scheme:dark;--bg:#0c1220;--panel:#172236;--line:#314056;--fg:#ebf1fa;--muted:#a5b4c9;--accent:#7bc5fb}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui, sans-serif}header{padding:15px 21px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px}h1{font-size:19px;margin:0}small,.muted{color:var(--muted)}button,select,input,textarea{background:#1b2a42;color:var(--fg);border:1px solid #455c78;border-radius:7px;padding:9px;font:inherit}button{cursor:pointer}button:hover{border-color:var(--accent)}button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--accent)}.primary{background:#1d5b88}.toolbar{display:flex;flex-wrap:wrap;gap:8px;padding:12px 20px;border-bottom:1px solid var(--line);align-items:center}.toolbar input{min-width:190px;flex:2}.toolbar select{flex:1;max-width:240px}.layout{display:grid;grid-template-columns:minmax(230px,290px) minmax(0,1fr);height:calc(100vh - 145px);min-height:520px}.side{overflow:auto;border-right:1px solid var(--line);padding:10px}.item{width:100%;text-align:left;margin:2px 0;background:none;border:0;border-bottom:1px solid #263449;border-radius:0;padding:9px 8px}.item.active{background:#294362;border-radius:7px}.item em{font-style:normal;color:#8ca3bd;font-size:12px}.item .tag{float:right;color:#9ed1a9}.main{min-width:0;overflow:auto;padding:16px 20px}.meta{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px}.codegrid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}.codebox{min-width:0;border:1px solid var(--line);border-radius:10px;overflow:hidden}.codetitle{background:#203049;padding:9px 12px;white-space:normal;overflow-wrap:anywhere}.codebox pre{margin:0;background:#111929;padding:15px;overflow:auto;max-height:52vh;white-space:pre;tab-size:4;font:12px/1.55 ui-monospace,monospace}.review{margin-top:12px;display:flex;flex-wrap:wrap;align-items:center;gap:10px;padding:13px;border:1px solid var(--line);border-radius:10px}.review label{display:flex;align-items:center;gap:6px}.review textarea{min-width:180px;flex:1;resize:vertical;min-height:42px}.foot{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}.banner{margin:8px 20px;background:#344122;color:#f0efc3;padding:9px;border-radius:7px}.pill{border:1px solid #52708e;padding:4px 8px;border-radius:20px;color:#c4def5}.status{color:#d5c18c}.error{color:#ffb9b9}@media(max-width:900px){.layout{grid-template-columns:1fr;height:auto}.side{max-height:150px;border-right:0;border-bottom:1px solid var(--line)}.codegrid{grid-template-columns:1fr}.codebox pre{max-height:42vh}}
</style></head><body>
<header><div><h1>D-05bh · Negative Candidate Review · REVIEWER_SLOT</h1><small>Offline · source-verified · labels not inferred · saved locally in this browser</small></div><div><span class="pill" id="progress">Loading…</span> <button id="export" class="primary">Export Reviewer CSV</button> <button id="importBtn">Import Reviewer CSV</button><input type="file" id="import" accept=".csv,text/csv" hidden></div></header>
<div class="banner">Review each code pair independently. <strong>Blank = Pending.</strong> Select <strong>0 Non-clone</strong>, <strong>1 Clone</strong>, or <strong>uncertain</strong>. Existing C-01 detection and training leakage flags are intentionally not displayed as hints. Reviewer A/B use separate HTML pages and separate browser storage. Export CSV regularly; browser storage is not a secure backup.</div>
<div class="toolbar"><input id="search" placeholder="Search pair ID, project, function, path…"><select id="project"><option value="">All projects</option></select><select id="bin"><option value="">All similarity bins</option></select><select id="status"><option value="">All review statuses</option><option value="pending">Pending</option><option value="0">Non-clone (0)</option><option value="1">Clone (1)</option><option value="uncertain">Uncertain</option></select><span id="filtered" class="muted"></span></div>
<div class="layout"><aside class="side"><div id="items"></div><button id="more" hidden>Show next 100</button></aside><main class="main"><div class="meta"><div><b id="pairtitle">Select a pair</b><div id="details" class="muted"></div></div><div><button id="prev">← Previous</button> <button id="next">Next →</button></div></div><section class="codegrid"><div class="codebox"><div id="lefttitle" class="codetitle">LEFT</div><pre id="leftcode"></pre></div><div class="codebox"><div id="righttitle" class="codetitle">RIGHT</div><pre id="rightcode"></pre></div></section><section class="review"><strong>Review:</strong><label><input type="radio" name="label" value="0">0 Non-clone</label><label><input type="radio" name="label" value="1">1 Clone</label><label><input type="radio" name="label" value="uncertain">Uncertain</label><button id="clear">Clear / Pending</button><label>Confidence <select id="confidence"><option value="">Not set</option><option value="high">High</option><option value="medium">Medium</option><option value="low">Low</option></select></label><textarea id="notes" placeholder="Rationale / notes"></textarea></section><div class="foot"><small>Keyboard: ← → navigate; press 0 / 1 / U to label when not typing. CSV export preserves original pair IDs.</small><span class="status" id="message"></span></div></main></div>
<script src="review_data.js"></script><script>
(()=>{'use strict';const REVIEWER='REVIEWER_SLOT',rows=window.D05BH_DATA||[],audit=window.D05BH_META||{};
const byid=new Map(rows.map((x,i)=>[x.pair_id,i]));const key='d05bh-v1-'+audit.dataset_sha256+'-'+REVIEWER;let annotations={};
try{annotations=JSON.parse(localStorage.getItem(key)||'{}')||{}}catch(e){annotations={}}
const el=id=>document.getElementById(id),all=x=>document.querySelectorAll(x);let filtered=[],index=0,shown=100;
const get=(id)=>annotations[id]||{clone_label:'',confidence:'',notes:''};
function persist(){try{localStorage.setItem(key,JSON.stringify(annotations));el('message').textContent='Saved locally';}catch(e){el('message').textContent='Local save failed: export CSV now';el('message').className='error'}}
function progress(){let yes=0,no=0,uncertain=0;for(const r of rows){const l=get(r.pair_id).clone_label;if(l==='0')no++;else if(l==='1')yes++;else if(l==='uncertain')uncertain++;}el('progress').textContent=`${rows.length-no-yes-uncertain} pending · ${no} non-clone · ${yes} clone · ${uncertain} uncertain`}
function renderList(){const list=el('items');list.replaceChildren();const f=document.createDocumentFragment();for(let i=0;i<Math.min(shown,filtered.length);i++){const r=rows[filtered[i]],l=get(r.pair_id).clone_label,b=document.createElement('button');b.className='item'+(index===i?' active':'');b.type='button';const txt=document.createElement('div');txt.textContent=`${i+1}. ${r.project} · ${r.left.qualname||''}`;const sub=document.createElement('em');sub.textContent=`${r.right.qualname||''} · ${r.pair_id.slice(-10)}`;const tag=document.createElement('span');tag.className='tag';tag.textContent=l===''?'○':l;b.append(txt,sub,tag);b.onclick=()=>{index=i;renderPair()};f.append(b)}list.append(f);el('more').hidden=shown>=filtered.length;el('filtered').textContent=filtered.length+' shown by filters';}
function renderPair(){renderList();if(!filtered.length){for(let name of ['pairtitle','details','lefttitle','righttitle','leftcode','rightcode'])el(name).textContent='';return;}const r=rows[filtered[Math.min(index,filtered.length-1)]],v=get(r.pair_id);el('pairtitle').textContent=r.pair_id;el('details').textContent=`${r.project} · similarity ${Number(r.pair_similarity).toFixed(5)} · bin ${r.similarity_bin} · ${r.coverage_extension_policy?'supplemental':'v3'}`;for(let side of ['left','right']){const s=r[side];el(side+'title').textContent=`${side.toUpperCase()}: ${s.qualname||'Function'} · ${s.path||''}:${s.start_line||'?'}–${s.end_line||'?'}`;el(side+'code').textContent=s.source_code||'';}all('input[name="label"]').forEach(x=>x.checked=x.value===v.clone_label);el('confidence').value=v.confidence||'';el('notes').value=v.notes||'';el('prev').disabled=index===0;el('next').disabled=index>=filtered.length-1;const active=el('items').querySelector('.active');if(active)active.scrollIntoView({block:'nearest'});}
function update(fields){if(!filtered.length)return;const r=rows[filtered[index]];annotations[r.pair_id]={...get(r.pair_id),...fields};persist();progress();renderList()}
function refilter(){const p=el('project').value,bin=el('bin').value,st=el('status').value,q=el('search').value.trim().toLowerCase();filtered=[];for(let i=0;i<rows.length;i++){const r=rows[i];if(p&&p!==r.project||bin&&bin!==r.similarity_bin||st&&((get(r.pair_id).clone_label||'pending')!==st))continue;if(q&&!([r.pair_id,r.project,r.left.qualname,r.right.qualname,r.left.path,r.right.path].join(' ').toLowerCase().includes(q)))continue;filtered.push(i)}index=0;shown=100;renderPair()}
for(const id of ['search','project','bin','status'])el(id).addEventListener(id==='search'?'input':'change',refilter);
for(const id of ['prev','next'])el(id).onclick=()=>{if(!filtered.length)return;index=Math.max(0,Math.min(filtered.length-1,index+(id==='next'?1:-1)));if(index>=shown)shown=Math.ceil((index+1)/100)*100;renderPair()};el('more').onclick=()=>{shown+=100;renderList()};
all('input[name="label"]').forEach(x=>x.onchange=()=>update({clone_label:x.value}));el('clear').onclick=()=>{update({clone_label:''});renderPair()};el('confidence').onchange=()=>update({confidence:el('confidence').value});el('notes').oninput=()=>update({notes:el('notes').value});
function quote(s){s=String(s??'');if(/^[=+@-]/.test(s))s="'"+s;return '"'+s.replaceAll('"','""')+'"'}
function download(name,content){const blob=new Blob([content],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000)}
el('export').onclick=()=>{const data=['pair_id,project,clone_label,confidence,notes'];for(const r of rows){const v=get(r.pair_id);data.push([r.pair_id,r.project,v.clone_label,v.confidence,v.notes].map(quote).join(','))}download('d-05bh-'+REVIEWER.toLowerCase()+'-submitted.csv','\ufeff'+data.join('\r\n')+'\r\n');el('message').textContent='Reviewer CSV exported'};
function csvParse(t){t=t.replace(/^\ufeff/,'');let lines=[],row=[],cell='',quoted=false;for(let i=0;i<t.length;i++){let c=t[i];if(quoted){if(c==='"'&&t[i+1]==='"'){cell+='"';i++;}else if(c==='"')quoted=false;else cell+=c;}else if(c==='"')quoted=true;else if(c===','){row.push(cell);cell='';}else if(c==='\n'){row.push(cell.replace(/\r$/,''));lines.push(row);row=[];cell='';}else cell+=c;}if(quoted)throw Error('Unterminated CSV quote');if(cell||row.length){row.push(cell.replace(/\r$/,''));lines.push(row);}return lines}
el('importBtn').onclick=()=>el('import').click();el('import').onchange=async e=>{try{const text=await e.target.files[0].text(),lines=csvParse(text),head=lines.shift(),fields=['pair_id','project','clone_label','confidence','notes'],pos=fields.map(k=>head.indexOf(k));if(pos.some(i=>i<0))throw Error('CSV schema mismatch');const incoming={};for(const line of lines){if(line.length===1&&!line[0])continue;const obj=Object.fromEntries(fields.map((k,j)=>[k,line[pos[j]]||'']));const x=byid.get(obj.pair_id);if(x===undefined||rows[x].project!==obj.project)throw Error('Unknown pair or project: '+obj.pair_id);if(!['','0','1','uncertain'].includes(obj.clone_label))throw Error('Invalid label: '+obj.clone_label);if(incoming[obj.pair_id])throw Error('Repeated pair: '+obj.pair_id);incoming[obj.pair_id]={clone_label:obj.clone_label,confidence:obj.confidence,notes:obj.notes}}if(Object.keys(incoming).length!==rows.length)throw Error('CSV must contain all '+rows.length+' pairs');if(!confirm('Import '+rows.length+' rows into '+REVIEWER+' local review? This replaces current browser annotations.'))return;annotations=incoming;persist();progress();refilter();el('message').textContent='Imported '+rows.length+' rows';}catch(err){el('message').textContent='Import error: '+err.message;el('message').className='error'}e.target.value=''};
document.addEventListener('keydown',e=>{const tag=e.target.tagName;if(['TEXTAREA','INPUT','SELECT'].includes(tag)||e.ctrlKey||e.metaKey||e.altKey)return;if(e.key==='ArrowLeft')el('prev').click();else if(e.key==='ArrowRight')el('next').click();else if(['0','1','u','U'].includes(e.key)){let label=e.key.toLowerCase()==='u'?'uncertain':e.key;update({clone_label:label});renderPair()}});
for(const project of [...new Set(rows.map(r=>r.project))].sort()){let o=document.createElement('option');o.value=project;o.textContent=project;el('project').append(o)}for(const bin of [...new Set(rows.map(r=>r.similarity_bin))].sort()){let o=document.createElement('option');o.value=bin;o.textContent=bin;el('bin').append(o)}progress();refilter();})();
</script></body></html>'''

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--repo-root',default='.')
    a.add_argument('--v3',help='v3 result directory or ZIP')
    a.add_argument('--supplement',help='paramiko supplemental result directory or ZIP')
    a.add_argument('--out',help='new output directory; must not exist')
    a.add_argument('--dry-run',action='store_true')
    args=a.parse_args();root=Path(args.repo_root).resolve()
    v3path=resolve_input(root,args.v3,'python-negative-candidates-v3-')
    suppath=resolve_input(root,args.supplement,'python-paramiko-supplemental-negative-v5-')
    v3,hv3=read_bundle(v3path,V3);sup,hsup=read_bundle(suppath,EXT)
    iv,kv,pv,cv=validate(v3,'v3');is_,ks,ps,cs=validate(sup,'supplement')
    if iv&is_ or kv&ks:fail('Duplicate ID or code hash pair across inputs')
    if len(v3)!=3971 or len(sup)!=4:fail('Unexpected input counts; expected 3971+4')
    allrows=v3+sup;projects=set(x['project'] for x in allrows)
    if len(projects)!=24:fail('Expected 24 projects, got '+str(len(projects)))
    if len(pv)!=23 or ps!={'paramiko'}:fail('Unexpected project coverage shape')
    ts=datetime.now(ZoneInfo('America/Chicago')).strftime('%Y%m%d-%H%M%S-%f')
    out=Path(args.out).expanduser().resolve() if args.out else root/'results'/'d-05b'/('python-unified-negative-review-v1-'+ts)
    public=[{'pair_id':x['pair_id'],'project':x['project'],'pair_similarity':x['pair_similarity'],'similarity_bin':x['similarity_bin'], 'left':{k:x['left'].get(k) for k in ('code_sha256','path','qualname','start_line','end_line','source_code')},'right':{k:x['right'].get(k) for k in ('code_sha256','path','qualname','start_line','end_line','source_code')},'coverage_extension_policy':x.get('coverage_extension_policy')} for x in allrows]
    dataset_digest=sha(('\n'.join(r['pair_id']+':'+r['left']['code_sha256']+':'+r['right']['code_sha256'] for r in public)).encode())
    summary={'status':'DRY_RUN' if args.dry_run else 'D05BH_V1_NEGATIVE_REVIEW_READY_NOT_GOLD','created_at_chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat(),'candidate_count':len(allrows),'v3_candidates':len(v3),'paramiko_supplemental_candidates':len(sup),'project_coverage':len(projects),'positive_reference_count':3971,'verified_negative_gold_labels':0,'blank_label_meaning':'PENDING_EXPLICIT_ZERO_REQUIRED','source_policy':'C01_ONLY_NO_LEGACY','input_manifest_sha256':{'negative_v3':hv3,'paramiko_supplemental_v5':hsup},'dataset_sha256':dataset_digest,'similarity_bin_counts':dict(sorted((cv+cs).items())),'offline_html_reviewers':['reviewer_a.html','reviewer_b.html']}
    print(json.dumps(summary,ensure_ascii=False,sort_keys=True))
    if args.dry_run:return
    if out.exists():fail('Output path already exists: '+str(out))
    out.mkdir(parents=True)
    with (out/'unified_negative_annotation_packets.jsonl').open('w',encoding='utf8') as f:
        for item in allrows:f.write(json.dumps(item,ensure_ascii=False,sort_keys=True)+'\n')
    fields=['pair_id','project','clone_label','confidence','notes'];template=[{'pair_id':r['pair_id'],'project':r['project'],'clone_label':'','confidence':'','notes':''} for r in allrows]
    write_csv(out/'reviewer_a_template.csv',template,fields);write_csv(out/'reviewer_b_template.csv',template,fields)
    write_csv(out/'review_manifest.csv',[{'pair_id':r['pair_id'],'project':r['project'],'similarity_bin':r['similarity_bin'],'pair_similarity':r['pair_similarity'],'source':'paramiko_supplemental' if r.get('coverage_extension_policy') else 'v3'} for r in allrows],['pair_id','project','similarity_bin','pair_similarity','source'])
    (out/'review_data.js').write_text('window.D05BH_META='+json.dumps({'dataset_sha256':dataset_digest},ensure_ascii=False)+';\nwindow.D05BH_DATA='+json.dumps(public,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')+';\n',encoding='utf-8')
    for r in ('A','B'):(out/('reviewer_'+r.lower()+'.html')).write_text(HTML.replace('REVIEWER_SLOT','Reviewer '+r),encoding='utf-8')
    (out/'negative_review_audit.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    files={p.name:{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()}
    (out/'manifest.json').write_text(json.dumps({'created_at_chicago':summary['created_at_chicago'],'input_manifest_sha256':summary['input_manifest_sha256'],'files':files},indent=2)+'\n',encoding='utf8')
    print('OUT:',out)
if __name__=='__main__':
    try:main()
    except (ValueError,FileNotFoundError,KeyError,zipfile.BadZipFile) as ex:print('ERROR:',ex,file=sys.stderr);sys.exit(2)
