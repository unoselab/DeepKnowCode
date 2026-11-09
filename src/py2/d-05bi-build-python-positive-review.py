#!/usr/bin/env python3
"""D-05bi v1: source-verified, fresh HTML re-review of 3,971 positive pairs.
No automatic label changes or Gold reassignment; no third-party dependencies.
"""
import argparse, csv, hashlib, json, sys, zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

def digest(x): return hashlib.sha256(x).hexdigest()
def require(flag,msg):
    if not flag: raise ValueError(msg)

def discover(root, explicit, pattern):
    if explicit:return Path(explicit).expanduser().resolve()
    folders=sorted(p for p in (root/'results'/'d-05b').glob(pattern) if p.is_dir())
    require(folders, 'Missing input '+pattern+'; supply explicit --prepared or --confirmed')
    return folders[-1]

def get_bundle(path, packet_name):
    if path.is_file():
        with zipfile.ZipFile(path) as z:
            require(z.testzip() is None,'Corrupt input ZIP')
            names=[n for n in z.namelist() if n.endswith('/'+packet_name)]
            require(len(names)==1,'Missing or duplicated '+packet_name)
            name=names[0]; folder=name.rsplit('/',1)[0]; manifest=json.loads(z.read(folder+'/manifest.json'))
            read=lambda short:z.read(folder+'/'+short)
            verify(manifest,read)
            raw=read(packet_name); manraw=read('manifest.json')
    else:
        manifest=json.loads((path/'manifest.json').read_text())
        read=lambda short:(path/short).read_bytes()
        verify(manifest,read)
        raw=read(packet_name); manraw=read('manifest.json')
    return [json.loads(line) for line in raw.splitlines() if line.strip()],digest(manraw)

def verify(man,reader):
    records=man.get('files') or {k:{'sha256':v} for k,v in man.get('artifacts_sha256',{}).items()}
    require(records,'Missing checksums in source manifest')
    for filename,meta in records.items():
        raw=reader(filename)
        require(digest(raw)==meta['sha256'],'Upstream SHA256 mismatch: '+filename)
        if 'bytes' in meta:require(len(raw)==meta['bytes'],'Upstream byte count mismatch: '+filename)

def write_csv(path,rows,cols):
    with path.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)

HTML=r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>D-05bi Positive Re-review — REVIEWER_SLOT</title>
<style>
:root{color-scheme:dark;--bg:#0c1220;--panel:#172236;--line:#314056;--fg:#ebf1fa;--muted:#a5b4c9;--accent:#7bc5fb}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui, sans-serif}header{padding:15px 21px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px}h1{font-size:19px;margin:0}small,.muted{color:var(--muted)}button,select,input,textarea{background:#1b2a42;color:var(--fg);border:1px solid #455c78;border-radius:7px;padding:9px;font:inherit}button{cursor:pointer}button:hover{border-color:var(--accent)}button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--accent)}.primary{background:#1d5b88}.toolbar{display:flex;flex-wrap:wrap;gap:8px;padding:12px 20px;border-bottom:1px solid var(--line);align-items:center}.toolbar input{min-width:190px;flex:2}.toolbar select{flex:1;max-width:240px}.layout{display:grid;grid-template-columns:minmax(230px,290px) minmax(0,1fr);height:calc(100vh - 145px);min-height:520px}.side{overflow:auto;border-right:1px solid var(--line);padding:10px}.item{width:100%;text-align:left;margin:2px 0;background:none;border:0;border-bottom:1px solid #263449;border-radius:0;padding:9px 8px}.item.active{background:#294362;border-radius:7px}.item em{font-style:normal;color:#8ca3bd;font-size:12px}.item .tag{float:right;color:#9ed1a9}.main{min-width:0;overflow:auto;padding:16px 20px}.meta{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px}.codegrid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}.codebox{min-width:0;border:1px solid var(--line);border-radius:10px;overflow:hidden}.codetitle{background:#203049;padding:9px 12px;white-space:normal;overflow-wrap:anywhere}.codebox pre{margin:0;background:#111929;padding:15px;overflow:auto;max-height:52vh;white-space:pre;tab-size:4;font:12px/1.55 ui-monospace,monospace}.review{margin-top:12px;display:flex;flex-wrap:wrap;align-items:center;gap:10px;padding:13px;border:1px solid var(--line);border-radius:10px}.review label{display:flex;align-items:center;gap:6px}.review textarea{min-width:180px;flex:1;resize:vertical;min-height:42px}.foot{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}.banner{margin:8px 20px;background:#344122;color:#f0efc3;padding:9px;border-radius:7px}.pill{border:1px solid #52708e;padding:4px 8px;border-radius:20px;color:#c4def5}.status{color:#d5c18c}.error{color:#ffb9b9}@media(max-width:900px){.layout{grid-template-columns:1fr;height:auto}.side{max-height:150px;border-right:0;border-bottom:1px solid var(--line)}.codegrid{grid-template-columns:1fr}.codebox pre{max-height:42vh}}
</style></head><body>
<header><div><h1>D-05bi · Positive Re-review · REVIEWER_SLOT</h1><small>Offline · source-verified · fresh labels default to Pending · saved locally in this browser</small></div><div><span class="pill" id="progress">Loading…</span> <button id="export" class="primary">Export Reviewer CSV</button> <button id="importBtn">Import Reviewer CSV</button><input type="file" id="import" accept=".csv,text/csv" hidden></div></header>
<div class="banner">Re-review each previously reviewer-confirmed C-01 positive code pair independently. <strong>Blank = Pending.</strong> Select <strong>0 Non-clone</strong>, <strong>1 Clone</strong>, or <strong>uncertain</strong>. The earlier positive verdict is intentionally hidden during this fresh review to reduce confirmation bias. Reviewer A/B use separate HTML pages and separate browser storage. Export CSV regularly; browser storage is not a secure backup.</div>
<div class="toolbar"><input id="search" placeholder="Search pair ID, project, function, path…"><select id="project"><option value="">All projects</option></select><select id="bin"><option value="">All review groups</option></select><select id="status"><option value="">All review statuses</option><option value="pending">Pending</option><option value="0">Non-clone (0)</option><option value="1">Clone (1)</option><option value="uncertain">Uncertain</option></select><span id="filtered" class="muted"></span></div>
<div class="layout"><aside class="side"><div id="items"></div><button id="more" hidden>Show next 100</button></aside><main class="main"><div class="meta"><div><b id="pairtitle">Select a pair</b><div id="details" class="muted"></div></div><div><button id="prev">← Previous</button> <button id="next">Next →</button></div></div><section class="codegrid"><div class="codebox"><div id="lefttitle" class="codetitle">LEFT</div><pre id="leftcode"></pre></div><div class="codebox"><div id="righttitle" class="codetitle">RIGHT</div><pre id="rightcode"></pre></div></section><section class="review"><strong>Review:</strong><label><input type="radio" name="label" value="0">0 Non-clone</label><label><input type="radio" name="label" value="1">1 Clone</label><label><input type="radio" name="label" value="uncertain">Uncertain</label><button id="clear">Clear / Pending</button><label>Confidence <select id="confidence"><option value="">Not set</option><option value="high">High</option><option value="medium">Medium</option><option value="low">Low</option></select></label><textarea id="notes" placeholder="Rationale / notes"></textarea></section><div class="foot"><small>Keyboard: ← → navigate; press 0 / 1 / U to label when not typing. CSV export preserves original pair IDs.</small><span class="status" id="message"></span></div></main></div>
<script src="review_data.js"></script><script>
(()=>{'use strict';const REVIEWER='REVIEWER_SLOT',rows=window.D05BI_DATA||[],audit=window.D05BI_META||{};
const byid=new Map(rows.map((x,i)=>[x.pair_id,i]));const key='d05bi-v1-'+audit.dataset_sha256+'-'+REVIEWER;let annotations={};
try{annotations=JSON.parse(localStorage.getItem(key)||'{}')||{}}catch(e){annotations={}}
const el=id=>document.getElementById(id),all=x=>document.querySelectorAll(x);let filtered=[],index=0,shown=100;
const get=(id)=>annotations[id]||{clone_label:'',confidence:'',notes:''};
function persist(){try{localStorage.setItem(key,JSON.stringify(annotations));el('message').textContent='Saved locally';}catch(e){el('message').textContent='Local save failed: export CSV now';el('message').className='error'}}
function progress(){let yes=0,no=0,uncertain=0;for(const r of rows){const l=get(r.pair_id).clone_label;if(l==='0')no++;else if(l==='1')yes++;else if(l==='uncertain')uncertain++;}el('progress').textContent=`${rows.length-no-yes-uncertain} pending · ${no} non-clone · ${yes} clone · ${uncertain} uncertain`}
function renderList(){const list=el('items');list.replaceChildren();const f=document.createDocumentFragment();for(let i=0;i<Math.min(shown,filtered.length);i++){const r=rows[filtered[i]],l=get(r.pair_id).clone_label,b=document.createElement('button');b.className='item'+(index===i?' active':'');b.type='button';const txt=document.createElement('div');txt.textContent=`${i+1}. ${r.project} · ${r.left.qualname||''}`;const sub=document.createElement('em');sub.textContent=`${r.right.qualname||''} · ${r.pair_id.slice(-10)}`;const tag=document.createElement('span');tag.className='tag';tag.textContent=l===''?'○':l;b.append(txt,sub,tag);b.onclick=()=>{index=i;renderPair()};f.append(b)}list.append(f);el('more').hidden=shown>=filtered.length;el('filtered').textContent=filtered.length+' shown by filters';}
function renderPair(){renderList();if(!filtered.length){for(let name of ['pairtitle','details','lefttitle','righttitle','leftcode','rightcode'])el(name).textContent='';return;}const r=rows[filtered[Math.min(index,filtered.length-1)]],v=get(r.pair_id);el('pairtitle').textContent=r.pair_id;el('details').textContent=`${r.project} · Fresh positive re-review · original detector label hidden`;for(let side of ['left','right']){const s=r[side];el(side+'title').textContent=`${side.toUpperCase()}: ${s.qualname||'Function'} · ${s.path||''}:${s.start_line||'?'}–${s.end_line||'?'}`;el(side+'code').textContent=s.source_code||'';}all('input[name="label"]').forEach(x=>x.checked=x.value===v.clone_label);el('confidence').value=v.confidence||'';el('notes').value=v.notes||'';el('prev').disabled=index===0;el('next').disabled=index>=filtered.length-1;const active=el('items').querySelector('.active');if(active)active.scrollIntoView({block:'nearest'});}
function update(fields){if(!filtered.length)return;const r=rows[filtered[index]];annotations[r.pair_id]={...get(r.pair_id),...fields};persist();progress();renderList()}
function refilter(){const p=el('project').value,bin=el('bin').value,st=el('status').value,q=el('search').value.trim().toLowerCase();filtered=[];for(let i=0;i<rows.length;i++){const r=rows[i];if(p&&p!==r.project||bin&&bin!==r.review_group||st&&((get(r.pair_id).clone_label||'pending')!==st))continue;if(q&&!([r.pair_id,r.project,r.left.qualname,r.right.qualname,r.left.path,r.right.path].join(' ').toLowerCase().includes(q)))continue;filtered.push(i)}index=0;shown=100;renderPair()}
for(const id of ['search','project','bin','status'])el(id).addEventListener(id==='search'?'input':'change',refilter);
for(const id of ['prev','next'])el(id).onclick=()=>{if(!filtered.length)return;index=Math.max(0,Math.min(filtered.length-1,index+(id==='next'?1:-1)));if(index>=shown)shown=Math.ceil((index+1)/100)*100;renderPair()};el('more').onclick=()=>{shown+=100;renderList()};
all('input[name="label"]').forEach(x=>x.onchange=()=>update({clone_label:x.value}));el('clear').onclick=()=>{update({clone_label:''});renderPair()};el('confidence').onchange=()=>update({confidence:el('confidence').value});el('notes').oninput=()=>update({notes:el('notes').value});
function quote(s){s=String(s??'');if(/^[=+@-]/.test(s))s="'"+s;return '"'+s.replaceAll('"','""')+'"'}
function download(name,content){const blob=new Blob([content],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000)}
el('export').onclick=()=>{const data=['pair_id,project,clone_label,confidence,notes'];for(const r of rows){const v=get(r.pair_id);data.push([r.pair_id,r.project,v.clone_label,v.confidence,v.notes].map(quote).join(','))}download('d-05bi-'+REVIEWER.toLowerCase()+'-submitted.csv','\ufeff'+data.join('\r\n')+'\r\n');el('message').textContent='Reviewer CSV exported'};
function csvParse(t){t=t.replace(/^\ufeff/,'');let lines=[],row=[],cell='',quoted=false;for(let i=0;i<t.length;i++){let c=t[i];if(quoted){if(c==='"'&&t[i+1]==='"'){cell+='"';i++;}else if(c==='"')quoted=false;else cell+=c;}else if(c==='"')quoted=true;else if(c===','){row.push(cell);cell='';}else if(c==='\n'){row.push(cell.replace(/\r$/,''));lines.push(row);row=[];cell='';}else cell+=c;}if(quoted)throw Error('Unterminated CSV quote');if(cell||row.length){row.push(cell.replace(/\r$/,''));lines.push(row);}return lines}
el('importBtn').onclick=()=>el('import').click();el('import').onchange=async e=>{try{const text=await e.target.files[0].text(),lines=csvParse(text),head=lines.shift(),fields=['pair_id','project','clone_label','confidence','notes'],pos=fields.map(k=>head.indexOf(k));if(pos.some(i=>i<0))throw Error('CSV schema mismatch');const incoming={};for(const line of lines){if(line.length===1&&!line[0])continue;const obj=Object.fromEntries(fields.map((k,j)=>[k,line[pos[j]]||'']));const x=byid.get(obj.pair_id);if(x===undefined||rows[x].project!==obj.project)throw Error('Unknown pair or project: '+obj.pair_id);if(!['','0','1','uncertain'].includes(obj.clone_label))throw Error('Invalid label: '+obj.clone_label);if(incoming[obj.pair_id])throw Error('Repeated pair: '+obj.pair_id);incoming[obj.pair_id]={clone_label:obj.clone_label,confidence:obj.confidence,notes:obj.notes}}if(Object.keys(incoming).length!==rows.length)throw Error('CSV must contain all '+rows.length+' pairs');if(!confirm('Import '+rows.length+' rows into '+REVIEWER+' local review? This replaces current browser annotations.'))return;annotations=incoming;persist();progress();refilter();el('message').textContent='Imported '+rows.length+' rows';}catch(err){el('message').textContent='Import error: '+err.message;el('message').className='error'}e.target.value=''};
document.addEventListener('keydown',e=>{const tag=e.target.tagName;if(['TEXTAREA','INPUT','SELECT'].includes(tag)||e.ctrlKey||e.metaKey||e.altKey)return;if(e.key==='ArrowLeft')el('prev').click();else if(e.key==='ArrowRight')el('next').click();else if(['0','1','u','U'].includes(e.key)){let label=e.key.toLowerCase()==='u'?'uncertain':e.key;update({clone_label:label});renderPair()}});
for(const project of [...new Set(rows.map(r=>r.project))].sort()){let o=document.createElement('option');o.value=project;o.textContent=project;el('project').append(o)}for(const bin of [...new Set(rows.map(r=>r.review_group))].sort()){let o=document.createElement('option');o.value=bin;o.textContent=bin;el('bin').append(o)}progress();refilter();})();
</script></body></html>'''

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',default='.')
    parser.add_argument('--prepared',help='D-05bd v1 directory or ZIP')
    parser.add_argument('--confirmed',help='D-05be v2 directory or ZIP')
    parser.add_argument('--out',help='New, nonexistent output directory')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(); root=Path(args.repo_root).resolve()
    pp=discover(root,args.prepared,'python-gold-annotation-preparation-v1-*')
    cp=discover(root,args.confirmed,'python-confirmed-positive-v2-*')
    packets,prepmanifest=get_bundle(pp,'annotation_packets.jsonl')
    confirmed,confirmmanifest=get_bundle(cp,'reviewer_confirmed_positive_pairs.jsonl')
    require(len(packets)==3971 and len(confirmed)==3971,'Expected 3971 pairs in both sources')
    pmap={}; projects=set(); pairkeys=set()
    for p in packets:
        pid=p['pair_id'];require(pid not in pmap,'Repeated prepared Pair ID: '+pid)
        l,r=p['left'],p['right']
        require(digest(l['code'].encode('utf-8'))==l['code_sha256'],'Left source hash mismatch: '+pid)
        require(digest(r['code'].encode('utf-8'))==r['code_sha256'],'Right source hash mismatch: '+pid)
        k=tuple(sorted((l['code_sha256'],r['code_sha256'])))
        require(k not in pairkeys,'Duplicate code pair: '+pid)
        pairkeys.add(k);pmap[pid]=p;projects.add(p['project'])
    cmap={}
    for c in confirmed:
        pid=c['pair_id'];require(pid not in cmap and pid in pmap,'Unexpected confirmed Pair ID: '+pid)
        require(c['project']==pmap[pid]['project'],'Project mismatch: '+pid)
        require(c['gold_label']==1 and c['reviewer_a_confirmed'] is True and c['reviewer_b_confirmed'] is True,'Not reviewer-confirmed positive: '+pid)
        require(c['label_source']=='C01_DETECTED_REVIEWER_A_B_CONFIRMED','Unexpected source: '+pid)
        require(c['independent_blind_annotation'] is False,'Blind review status mismatch: '+pid)
        for side in ['left','right']:
            raw=pmap[pid][side]
            require(c[side+'_code_sha256']==raw['code_sha256'],'Code hash identity mismatch: '+pid)
            loc=c[side+'_locator']
            for attr in ['path','start_line','end_line','function']:
                require(loc[attr]==raw[attr],'Source locator mismatch '+pid+': '+attr)
        cmap[pid]=c
    require(len(cmap)==len(pmap) and len(projects)==24,'Incomplete coverage')
    public=[]
    for p in packets:
        sides={}
        for side in ['left','right']:
            x=p[side];sides[side]={'code_sha256':x['code_sha256'],'source_code':x['code'],'path':x['path'],'qualname':x['function'],'start_line':x['start_line'],'end_line':x['end_line']}
        public.append({'pair_id':p['pair_id'],'project':p['project'],'review_group':'C01_positive_recheck',**sides})
    ds=digest(('\n'.join(p['pair_id']+':'+p['left']['code_sha256']+':'+p['right']['code_sha256'] for p in public)).encode())
    timestamp=datetime.now(ZoneInfo('America/Chicago'))
    summary={'status':'DRY_RUN' if args.dry_run else 'D05BI_V1_POSITIVE_REREVIEW_READY_GOLD_UNCHANGED','created_at_chicago':timestamp.isoformat(),'candidate_count':3971,'projects':24,'previous_reviewer_confirmed_positive_pairs':3971,'fresh_re_review_labels':0,'gold_labels_modified':0,'blank_label_meaning':'PENDING_FRESH_REVIEW','prior_label_policy':'REVIEWER_AGREEMENT_WITH_C01_DETECTION','independent_blind_annotation_claim':False,'input_manifest_sha256':{'d05bd_v1_preparation':prepmanifest,'d05be_v2_positive':confirmmanifest},'dataset_sha256':ds,'output_html':['reviewer_a.html','reviewer_b.html']}
    print(json.dumps(summary,sort_keys=True,ensure_ascii=False))
    if args.dry_run:return
    out=Path(args.out).expanduser().resolve() if args.out else root/'results'/'d-05b'/('python-positive-rereview-v1-'+timestamp.strftime('%Y%m%d-%H%M%S-%f'))
    require(not out.exists(),'Output already exists: '+str(out));out.mkdir(parents=True)
    with (out/'positive_rereview_packets.jsonl').open('w',encoding='utf8') as f:
        for p in packets:f.write(json.dumps(p,ensure_ascii=False,sort_keys=True)+'\n')
    fields=['pair_id','project','clone_label','confidence','notes']
    templates=[{'pair_id':p['pair_id'],'project':p['project'],'clone_label':'','confidence':'','notes':''} for p in public]
    for letter in 'ab':write_csv(out/('reviewer_'+letter+'_template.csv'),templates,fields)
    write_csv(out/'review_manifest.csv',[{'pair_id':p['pair_id'],'project':p['project'],'previous_label':1,'previous_annotation_method':'reviewer_confirmation_of_detector_prediction','fresh_rereview_label':''} for p in public],['pair_id','project','previous_label','previous_annotation_method','fresh_rereview_label'])
    # Source code is treated as untrusted and JSON-escaped; HTML renders only with textContent.
    payload='window.D05BI_META='+json.dumps({'dataset_sha256':ds})+';\nwindow.D05BI_DATA='+json.dumps(public,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')+';\n'
    (out/'review_data.js').write_text(payload,encoding='utf8')
    for letter in 'AB':(out/('reviewer_'+letter.lower()+'.html')).write_text(HTML.replace('REVIEWER_SLOT','Reviewer '+letter),encoding='utf8')
    (out/'positive_rereview_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    files={f.name:{'sha256':digest(f.read_bytes()),'bytes':f.stat().st_size} for f in sorted(out.iterdir()) if f.is_file()}
    (out/'manifest.json').write_text(json.dumps({'version':'v1','created_at_chicago':timestamp.isoformat(),'input_manifest_sha256':summary['input_manifest_sha256'],'files':files},indent=2)+'\n',encoding='utf8')
    print('OUT:',out)

if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,FileNotFoundError,zipfile.BadZipFile) as e:print('ERROR:',e,file=sys.stderr);sys.exit(2)
