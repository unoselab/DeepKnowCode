#!/usr/bin/env python3
"""Generate an offline, read-only-source HTML human adjudication interface for C-06."""
import argparse
import csv
import hashlib
import html
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HTML_TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>C-07 · Negative Pair Review Studio</title>
<style>
:root{color-scheme:dark;--bg:#0d111b;--panel:#151c2a;--panel2:#1a2435;--ink:#e8edf5;--muted:#a7b4c8;--stroke:#344158;--accent:#72bcf7;--good:#48c49e;--bad:#fc8f97;--warn:#efbf6c;--mono:ui-monospace,SFMono-Regular,Consolas,monospace}
*{box-sizing:border-box}html,body{margin:0;min-height:100%;background:var(--bg);color:var(--ink);font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}button,input,select,textarea{font:inherit}button{cursor:pointer}button:focus-visible,input:focus-visible,select:focus-visible,textarea:focus-visible{outline:2px solid var(--accent);outline-offset:2px}header{padding:18px 24px;border-bottom:1px solid var(--stroke);background:#121a27;position:sticky;top:0;z-index:5}.topline{display:flex;gap:20px;align-items:center;justify-content:space-between;flex-wrap:wrap}.brand{display:flex;align-items:center;gap:12px}.mark{background:#183c55;color:#8dd6ff;border:1px solid #28749b;border-radius:13px;padding:12px;font-weight:900;font-family:var(--mono)}h1{font-size:19px;letter-spacing:-.02em;margin:0}header p{font-size:12px;color:var(--muted);margin:5px 0 0}.head-actions{display:flex;gap:8px;flex-wrap:wrap}button,.linkbtn{background:#223049;border:1px solid #3a4e6b;color:var(--ink);padding:9px 13px;border-radius:9px;font-weight:600;font-size:12px}button:hover,.linkbtn:hover{background:#304361}.primary{background:#165c8a;border-color:#2b91c2}.primary:hover{background:#227aaa}.shell{max-width:1800px;margin:auto;padding:18px 22px 40px}.statline{display:grid;grid-template-columns:repeat(4,minmax(130px,1fr));gap:10px;margin-bottom:12px}.stat{background:var(--panel);border:1px solid var(--stroke);border-radius:12px;padding:12px 15px}.stat .value{font-size:23px;font-weight:750;font-variant-numeric:tabular-nums}.stat .tag{font-size:12px;color:var(--muted);margin-top:2px}.bar{height:7px;background:#283349;border-radius:20px;overflow:hidden;margin:12px 0 15px}.fill{height:100%;background:linear-gradient(90deg,#48c49e,#64b2ff);width:0}.toolbar{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin:0 0 14px}.toolbar input,.toolbar select,.reviewbox input,.reviewbox select,.reviewbox textarea{background:#0f1725;border:1px solid #40516b;color:var(--ink);border-radius:8px;padding:9px 11px;font-size:13px}.toolbar input{min-width:220px;flex:1}.layout{display:grid;grid-template-columns:275px minmax(0,1fr);gap:14px}.sidebar{background:var(--panel);border:1px solid var(--stroke);border-radius:13px;max-height:calc(100vh - 245px);overflow:auto}.sidebar-head{position:sticky;top:0;background:var(--panel);padding:12px 14px;border-bottom:1px solid var(--stroke);z-index:2;font-size:12px;color:var(--muted)}.item{width:100%;text-align:left;padding:11px 13px;border:0;border-bottom:1px solid #293347;border-radius:0;background:transparent;display:grid;gap:4px}.item:hover{background:#243147}.item.active{background:#1b4059;border-left:3px solid var(--accent);padding-left:10px}.item .name{font-size:13px;color:#f0f4fa;font-weight:700}.item .sub{font-size:11px;color:#b1bfd0;display:flex;justify-content:space-between;gap:8px}.pill{display:inline-block;padding:3px 8px;border-radius:30px;background:#283448;color:#c6d4e8;font-size:11px;font-weight:700}.pill.high{background:#51343d;color:#ffbec2}.pill.length{background:#57452b;color:#ffdaa0}.pill.screen{background:#204b44;color:#a2e9d4}.pill.reviewed{background:#204b44;color:#adf2dc}.main{min-width:0;display:flex;flex-direction:column;gap:12px}.panel{background:var(--panel);border:1px solid var(--stroke);border-radius:13px}.pair-top{padding:15px 18px;display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:12px}.pair-title{font-size:17px;font-weight:750;margin:0 0 7px}.badges{display:flex;gap:7px;flex-wrap:wrap}.pair-nav{display:flex;gap:8px}.metrics{padding:0 18px 14px;display:flex;gap:8px;flex-wrap:wrap}.metric{background:#101b2b;border:1px solid #35455d;padding:7px 10px;border-radius:8px;color:var(--muted);font-size:11px}.metric strong{display:block;color:var(--ink);font-size:13px;margin-top:3px}.codegrid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:12px}.codepanel{min-width:0;overflow:hidden}.codehead{display:flex;align-items:center;justify-content:space-between;padding:12px 15px;border-bottom:1px solid var(--stroke);gap:12px}.codehead .title{font-size:13px;font-weight:750}.codehead .subtitle{font-family:var(--mono);font-size:10px;color:var(--muted);overflow-wrap:anywhere;margin-top:5px}.code{max-height:500px;overflow:auto;background:#0b1320;padding:11px 0;tab-size:4;font:12px/1.65 var(--mono)}.code-line{display:grid;grid-template-columns:43px minmax(max-content,1fr);min-height:20px;white-space:pre}.code-line .num{color:#72849c;text-align:right;padding-right:13px;user-select:none;border-right:1px solid #29394f}.code-line .src{padding:0 14px;white-space:pre}.code-line.changed{background:#422e3b}.code-line.changed .num{color:#ffb1bc}.code-line.same{background:transparent}.code-line .kw{color:#c5aaff}.code-line .str{color:#a8dc9d}.code-line .com{color:#8b9baf}.code-line .numtok{color:#ffc188}.reviewbox{padding:17px 18px;display:grid;gap:13px}.reviewbox .sectionhead{font-size:14px;font-weight:750}.decisions{display:flex;flex-wrap:wrap;gap:9px}.decision{padding:11px 15px;background:#172438;border:1px solid #40516b;color:#e0eaf7;font-weight:700}.decision.chosen{border-color:#75d5e7;box-shadow:inset 0 0 0 1px #75d5e7;background:#174963}.decision[data-choice=CONFIRMED_CLONE].chosen{background:#693644;border-color:#fc8f97}.decision[data-choice=CONFIRMED_NONCLONE].chosen{background:#245747;border-color:#48c49e}.fields{display:grid;grid-template-columns:1fr 1fr;gap:11px}.field{display:grid;gap:5px}.field label{color:var(--muted);font-size:12px}.field.full{grid-column:1/-1}.reviewbox textarea{min-height:75px;resize:vertical}.notice{font-size:12px;color:#ebcc8a;background:#342d23;border:1px solid #665230;padding:11px 13px;border-radius:9px}.foot{font-size:11px;color:var(--muted)}.empty{padding:35px;color:var(--muted);text-align:center}.hidden{display:none!important}#toast{position:fixed;bottom:20px;right:20px;z-index:20;background:#2c5265;padding:10px 16px;border-radius:9px;box-shadow:0 6px 30px #0008;display:none}@media(max-width:1040px){.layout{grid-template-columns:210px minmax(0,1fr)}.codegrid{grid-template-columns:1fr}}@media(max-width:640px){header{position:static}.statline{grid-template-columns:1fr 1fr}.layout{display:block}.sidebar{max-height:220px;margin-bottom:12px}.fields{grid-template-columns:1fr}.shell{padding:12px}}
</style></head><body>
<header><div class="topline"><div class="brand"><div class="mark">C-07</div><div><h1>Negative Pair Review Studio</h1><p>DeepKnowCode · Offline Human Adjudication · C-06 queue</p></div></div><div class="head-actions"><button id="exportCsv" class="primary">Export adjudication CSV</button><button id="exportJson">Backup JSON</button><button id="importBtn">Import review</button><input id="importFile" type="file" accept=".csv,.json" hidden><button id="clearBtn">Clear local draft</button></div></div></header>
<main class="shell"><section class="statline"><div class="stat"><div class="value" id="total">274</div><div class="tag">Pairs in review queue</div></div><div class="stat"><div class="value" id="reviewed">0</div><div class="tag">Reviewed / adjudicated</div></div><div class="stat"><div class="value" id="cloneCount">0</div><div class="tag">Confirmed clones</div></div><div class="stat"><div class="value" id="nonCount">0</div><div class="tag">Confirmed non-clones</div></div></section><div class="bar"><div class="fill" id="progress"></div></div>
<div class="toolbar"><input id="search" placeholder="Search project, ID, flags or code…"><select id="group"><option value="">All risk groups</option><option value="HIGH_PRIORITY_SIMILARITY_AST">High priority (Similarity / AST)</option><option value="LENGTH_ONLY_REVIEW">Length-only review</option><option value="SCREENED_BASELINE">Screened baseline</option></select><select id="project"><option value="">All projects</option></select><select id="status"><option value="">All statuses</option><option value="pending">Pending only</option><option value="reviewed">Reviewed only</option><option value="CONFIRMED_CLONE">Confirmed clone</option><option value="CONFIRMED_NONCLONE">Confirmed non-clone</option><option value="UNCERTAIN">Uncertain</option></select><span class="pill" id="shown"></span></div>
<section class="layout"><aside class="sidebar"><div class="sidebar-head">REVIEW QUEUE · select a pair</div><div id="items"></div></aside><section class="main"><article class="panel"><div class="pair-top"><div><div class="pair-title" id="pairTitle">Select a pair</div><div class="badges" id="pairBadges"></div></div><div class="pair-nav"><button id="prev">← Previous</button><button id="next">Next →</button></div></div><div class="metrics" id="metrics"></div></article><div class="codegrid"><article class="panel codepanel"><div class="codehead"><div><div class="title">Function A</div><div class="subtitle" id="ida"></div></div><button id="copyA">Copy</button></div><div class="code" id="codeA"></div></article><article class="panel codepanel"><div class="codehead"><div><div class="title">Function B</div><div class="subtitle" id="idb"></div></div><button id="copyB">Copy</button></div><div class="code" id="codeB"></div></article></div><article class="panel reviewbox"><div class="sectionhead">Human adjudication</div><div class="decisions"><button class="decision" data-choice="CONFIRMED_NONCLONE">✓ NON-CLONE <small>[1]</small></button><button class="decision" data-choice="CONFIRMED_CLONE">✕ CLONE <small>[2]</small></button><button class="decision" data-choice="UNCERTAIN">? UNCERTAIN <small>[3]</small></button><button id="resetPair">Reset pair</button></div><div class="fields"><div class="field"><label for="reviewer">Reviewer</label><input id="reviewer" placeholder="Reviewer identifier"></div><div class="field"><label for="confidence">Confidence</label><select id="confidence"><option value="">Not specified</option><option value="HIGH">High</option><option value="MEDIUM">Medium</option><option value="LOW">Low</option></select></div><div class="field full"><label for="notes">Evidence / review notes</label><textarea id="notes" placeholder="Explain why these functions are or are not code clones, or why uncertain…"></textarea></div></div><div class="notice">Review choices are human annotations, not automated ground truth. Offline draft may be stored in this browser; export CSV periodically. Source snippets are embedded in this HTML—do not publish it.</div><div class="foot">Keyboard: 1 Non-clone · 2 Clone · 3 Uncertain · ← / → navigate (outside text fields). Changed lines are highlighted for reading only, not proof of semantic similarity.</div></article></section></section></main><div id="toast"></div>
<script id="dataset" type="application/json">__DATA__</script>
<script>
'use strict';
const DATA=JSON.parse(document.getElementById('dataset').textContent);
const PAIRS=DATA.pairs, MAP=Object.fromEntries(PAIRS.map(p=>[p.review_id,p]));
const KEY='deepknowcode-c07-review-'+DATA.source_digest;
let saved={}, current=null, view=[];
const $=id=>document.getElementById(id);
const ESC=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const safe=()=>{try{localStorage.setItem(KEY,JSON.stringify(saved))}catch(e){toast('Browser storage unavailable; export CSV regularly.')}};
try{saved=JSON.parse(localStorage.getItem(KEY)||'{}')||{}}catch(e){saved={}}
function toast(t){const x=$('toast');x.textContent=t;x.style.display='block';setTimeout(()=>x.style.display='none',2900)}
function chicago(){return new Intl.DateTimeFormat('sv-SE',{timeZone:'America/Chicago',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}).format(new Date())+' America/Chicago'}
function rec(id){return saved[id]||{}}
function updateCounts(){const a=PAIRS.map(p=>rec(p.review_id).adjudication).filter(Boolean);$('total').textContent=PAIRS.length;$('reviewed').textContent=a.length;$('cloneCount').textContent=a.filter(x=>x==='CONFIRMED_CLONE').length;$('nonCount').textContent=a.filter(x=>x==='CONFIRMED_NONCLONE').length;$('progress').style.width=(a.length/PAIRS.length*100)+'%'}
function badge(t,kind=''){return `<span class="pill ${kind}">${ESC(t)}</span>`}
function filter(){const q=$('search').value.trim().toLowerCase(),g=$('group').value,p=$('project').value,s=$('status').value;view=PAIRS.filter(x=>{const r=rec(x.review_id),jud=r.adjudication||'',searched=[x.project,x.id_a,x.id_b,x.flags,x.review_id,x.code_a,x.code_b].join(' ').toLowerCase();return (!q||searched.includes(q))&&(!g||x.review_group===g)&&(!p||x.project===p)&&(!s||(s==='pending'?!jud:s==='reviewed'?!!jud:jud===s))});$('shown').textContent=view.length+' shown';if(!view.some(x=>x.review_id===current))current=view[0]?.review_id||null;sidebar();render()}
function sidebar(){const c=$('items');c.innerHTML=view.map((p,i)=>{const r=rec(p.review_id);return `<button class="item ${p.review_id===current?'active':''}" data-id="${ESC(p.review_id)}"><div class="name">${String(i+1).padStart(3,'0')} · ${ESC(p.project)} ${r.adjudication?badge('✓ Reviewed','reviewed'):''}</div><div class="sub"><span>${ESC(p.review_group==='HIGH_PRIORITY_SIMILARITY_AST'?'Similarity/AST':p.review_group==='LENGTH_ONLY_REVIEW'?'Length-only':'Screened')}</span><span>${ESC(p.split)}</span></div></button>`}).join('')||'<div class="empty">No matching pairs</div>';c.querySelectorAll('[data-id]').forEach(b=>b.onclick=()=>{current=b.dataset.id;sidebar();render()})}
function highlightLine(s){const rx=/(#[^\n]*|\b(?:def|return|class|if|else|elif|for|while|in|not|is|and|or|try|except|raise|yield|import|from|with|as|async|await|True|False|None|pass|continue|break|lambda)\b|\b\d+(?:\.\d+)?\b|(?:"""[^"]*"""|\x27\x27\x27[^\x27]*\x27\x27\x27|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'))/g;let out='',last=0,m;while((m=rx.exec(s))){out+=ESC(s.slice(last,m.index));const t=m[0],kind=t[0]==='#'?'com':t[0]==='"'||t[0]==="'"?'str':/^\d/.test(t)?'numtok':'kw';out+=`<span class="${kind}">${ESC(t)}</span>`;last=rx.lastIndex}return out+ESC(s.slice(last))}
function codeLines(text,changed){const lines=String(text).replace(/\r\n/g,'\n').split('\n');return lines.map((s,i)=>`<div class="code-line ${changed.includes(i)?'changed':'same'}"><span class="num">${i+1}</span><span class="src">${highlightLine(s)||' '}</span></div>`).join('')}
function metric(k,v){return `<div class="metric">${ESC(k)}<strong>${ESC(v)}</strong></div>`}
function render(){const p=MAP[current];if(!p){$('pairTitle').textContent='No pairs match the filters';$('pairBadges').innerHTML='';$('metrics').innerHTML='';$('codeA').innerHTML=$('codeB').innerHTML='';$('ida').textContent=$('idb').textContent='';document.querySelector('.reviewbox').classList.add('hidden');return}document.querySelector('.reviewbox').classList.remove('hidden');const r=rec(current);$('pairTitle').textContent=p.project+' · '+(view.findIndex(x=>x.review_id===current)+1)+' / '+view.length;$('pairBadges').innerHTML=badge(p.review_group,p.review_group==='HIGH_PRIORITY_SIMILARITY_AST'?'high':p.review_group==='LENGTH_ONLY_REVIEW'?'length':'screen')+badge(p.split)+badge(p.screening);(p.flags||'').split('|').filter(Boolean).forEach(f=>$('pairBadges').innerHTML+=badge(f,'high'));$('metrics').innerHTML=metric('Token Jaccard',p.token_jaccard)+metric('Sequence similarity',p.sequence_similarity)+metric('Length ratio',p.length_ratio)+metric('AST same shape',p.ast_shape_equal)+metric('Code equal',p.exact_code_equal);$('ida').textContent=p.id_a;$('idb').textContent=p.id_b;$('codeA').innerHTML=codeLines(p.code_a,p.diff_a);$('codeB').innerHTML=codeLines(p.code_b,p.diff_b);$('codeA').scrollTop=$('codeB').scrollTop=0;document.querySelectorAll('[data-choice]').forEach(b=>b.classList.toggle('chosen',b.dataset.choice===r.adjudication));$('reviewer').value=r.reviewer||'';$('confidence').value=r.confidence||'';$('notes').value=r.review_notes||''}
function update(key,value){if(!current)return;const r={...rec(current),[key]:value};r.reviewed_at_chicago=chicago();if(!(r.adjudication||r.reviewer||r.confidence||r.review_notes))delete saved[current];else saved[current]=r;safe();updateCounts();sidebar();if(key==='adjudication')render()}
document.querySelectorAll('[data-choice]').forEach(b=>b.onclick=()=>update('adjudication',b.dataset.choice));$('reviewer').onchange=e=>update('reviewer',e.target.value);$('confidence').onchange=e=>update('confidence',e.target.value);$('notes').onchange=e=>update('review_notes',e.target.value);$('resetPair').onclick=()=>{if(!current||!confirm('Clear this pair\'s annotations?'))return;delete saved[current];safe();updateCounts();sidebar();render()};
function nav(delta){if(!view.length)return;const i=view.findIndex(x=>x.review_id===current);current=view[Math.min(view.length-1,Math.max(0,i+delta))].review_id;sidebar();render();$('items').querySelector('.item.active')?.scrollIntoView({block:'nearest'})}
$('prev').onclick=()=>nav(-1);$('next').onclick=()=>nav(1);['search','group','project','status'].forEach(id=>$(id).addEventListener(id==='search'?'input':'change',filter));
document.addEventListener('keydown',e=>{if(e.ctrlKey||e.metaKey||e.altKey||['INPUT','TEXTAREA','SELECT'].includes(document.activeElement?.tagName))return;if(e.key==='1')update('adjudication','CONFIRMED_NONCLONE');else if(e.key==='2')update('adjudication','CONFIRMED_CLONE');else if(e.key==='3')update('adjudication','UNCERTAIN');else if(e.key==='ArrowLeft')nav(-1);else if(e.key==='ArrowRight')nav(1)});
function download(name,text,type){const blob=new Blob([text],{type}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000)}
const COLS=['review_id','project','split','id_a','id_b','review_group','reviewer','adjudication','confidence','review_notes','reviewed_at_chicago'];
function csvCell(v){const s=String(v??'');return /[",\r\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s}
$('exportCsv').onclick=()=>{const rows=[COLS.join(','),...PAIRS.map(p=>COLS.map(k=>csvCell(k in p?p[k]:rec(p.review_id)[k]||'')).join(','))];download('c-07-adjudication-'+new Date().toISOString().slice(0,10)+'.csv','\ufeff'+rows.join('\r\n'),'text/csv;charset=utf-8');toast('CSV exported: '+Object.keys(saved).length+' annotated rows')};
$('exportJson').onclick=()=>download('c-07-review-backup.json',JSON.stringify({source_digest:DATA.source_digest,annotations:saved,exported_at_chicago:chicago()},null,2),'application/json');
function parseCSV(text){let rows=[],row=[],cell='',quoted=false;for(let i=0;i<text.length;i++){const c=text[i];if(quoted){if(c==='"'&&text[i+1]==='"'){cell+='"';i++}else if(c==='"')quoted=false;else cell+=c}else if(c==='"')quoted=true;else if(c===','){row.push(cell);cell=''}else if(c==='\n'||c==='\r'){if(c==='\r'&&text[i+1]==='\n')i++;row.push(cell);if(row.some(Boolean))rows.push(row);row=[];cell=''}else cell+=c}row.push(cell);if(row.some(Boolean))rows.push(row);const h=rows.shift()||[];return rows.map(a=>Object.fromEntries(h.map((k,i)=>[k.replace(/^\ufeff/,''),a[i]||''])))}
function imported(x){if(!x||!MAP[x.review_id])return false;const keys=['reviewer','adjudication','confidence','review_notes','reviewed_at_chicago'];const obj=Object.fromEntries(keys.map(k=>[k,String(x[k]||'')]));if(!['','CONFIRMED_CLONE','CONFIRMED_NONCLONE','UNCERTAIN'].includes(obj.adjudication))return false;if(obj.adjudication||obj.reviewer||obj.confidence||obj.review_notes)saved[x.review_id]=obj;return true}
$('importBtn').onclick=()=>$('importFile').click();$('importFile').onchange=async e=>{const f=e.target.files[0];if(!f)return;try{const t=await f.text();let n=0;if(f.name.toLowerCase().endsWith('.json')){const o=JSON.parse(t);if(o.source_digest!==DATA.source_digest)throw Error('Different C-06 source digest; import refused');for(const [id,r] of Object.entries(o.annotations||{}))n+=imported({...r,review_id:id})?1:0}else{const rows=parseCSV(t);if(!rows.some(r=>r.review_id&&MAP[r.review_id]))throw Error('No matching review IDs');for(const row of rows)n+=imported(row)?1:0}safe();updateCounts();filter();toast('Imported '+n+' matching review records')}catch(err){alert('Import error: '+err.message)}e.target.value=''};
$('clearBtn').onclick=()=>{if(!confirm('Clear ALL locally saved annotations? Export your CSV or JSON first.'))return;saved={};safe();updateCounts();filter();toast('Local draft cleared')};
const projects=[...new Set(PAIRS.map(p=>p.project))].sort((a,b)=>a.localeCompare(b));$('project').innerHTML+=[...projects].map(p=>`<option value="${ESC(p)}">${ESC(p)}</option>`).join('');updateCounts();filter();
</script></body></html>'''


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def load(args):
    d = Path(args.c06).expanduser().resolve()
    q, code, tmpl = (d / n for n in ('review_queue.csv', 'review_queue_code.jsonl', 'adjudication_template.csv'))
    for p in (q, code, tmpl):
        if not p.is_file():
            raise ValueError(f'Missing C-06 input: {p}')
    with q.open(newline='', encoding='utf-8') as f:
        queue = list(csv.DictReader(f))
    with tmpl.open(newline='', encoding='utf-8') as f:
        template = list(csv.DictReader(f))
    codes = {}
    with code.open(encoding='utf-8') as f:
        for line in f:
            v = json.loads(line)
            rid = v['review_id']
            if rid in codes:
                raise ValueError(f'Duplicate review ID in Code JSONL: {rid}')
            codes[rid] = v
    ids = [v['review_id'] for v in queue]
    if len(ids) != len(set(ids)) or len(ids) != len(template) or set(ids) != set(codes) or set(ids) != {v['review_id'] for v in template}:
        raise ValueError('Queue, Source Code and Adjudication Template review IDs/counts differ')
    if len(ids) == 0:
        raise ValueError('Empty review queue')
    output = []
    import difflib
    for row in queue:
        c = codes[row['review_id']]
        for field in ('project', 'split', 'id_a', 'id_b', 'review_group'):
            if str(row[field]) != str(c.get(field, '')):
                raise ValueError(f'Mismatched {field} for {row["review_id"]}')
        a, b = c['code_a'], c['code_b']
        aa, bb = a.replace('\r\n', '\n').split('\n'), b.replace('\r\n', '\n').split('\n')
        diff_a, diff_b = set(), set()
        for tag, i, j, k, l in difflib.SequenceMatcher(None, aa, bb, autojunk=False).get_opcodes():
            if tag != 'equal':
                diff_a.update(range(i,j));diff_b.update(range(k,l))
        output.append({**row, 'code_a': a, 'code_b': b, 'diff_a': sorted(diff_a), 'diff_b': sorted(diff_b)})
    return d, q, code, tmpl, output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument('--c06', type=Path, help='C-06 result directory (default newest matching results/c-06 run)')
    ap.add_argument('--output', type=Path, help='New output directory (must not exist)')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    root = args.root.expanduser().resolve()
    if args.c06 is not None and not args.c06.is_absolute():
        args.c06 = root / args.c06
    if args.output is not None and not args.output.is_absolute():
        args.output = root / args.output
    if not args.c06:
        base = root / 'results/c-06'
        options = sorted(p for p in base.iterdir() if p.is_dir() and (p/'review_queue_code.jsonl').is_file()) if base.is_dir() else []
        if not options: ap.error(f'No C-06 result directories in {base}')
        args.c06 = options[-1]
    try:
        folder,q,c,t,rows = load(args)
    except (ValueError, OSError, json.JSONDecodeError) as e:
        ap.error(str(e))
    groups = {}
    for row in rows: groups[row['review_group']] = groups.get(row['review_group'], 0) + 1
    report = {'version':'v1','status':'HTML_REVIEW_READY_NOT_ADJUDICATED','review_pairs':len(rows),'projects':len({r['project'] for r in rows}),'groups':groups,'c06_input':str(folder),'source_sha256':sha256(c),'training_approval':False,'negative_labels_verified':False}
    print('C-07 ' + json.dumps(report, ensure_ascii=False, sort_keys=True))
    if args.dry_run:
        print('DRY RUN: no output created')
        return
    output = args.output.expanduser().resolve() if args.output else root / 'results/c-07' / datetime.now(ZoneInfo('America/Chicago')).strftime('%Y%m%d-%H%M%S-%f')
    if output.exists(): ap.error(f'Output already exists; refusing overwrite: {output}')
    output.mkdir(parents=True, exist_ok=False)
    payload = {'source_digest': sha256(c), 'pairs': rows}
    serialized = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    page = HTML_TEMPLATE.replace('__DATA__', serialized)
    f = output / 'review_studio.html'
    f.write_text(page, encoding='utf-8')
    report['created_at_chicago'] = datetime.now(ZoneInfo('America/Chicago')).isoformat()
    report['output_html'] = 'review_studio.html'
    report['output_html_sha256'] = sha256(f)
    (output/'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print('OUTPUT:', output)
    print('HTML:', f)
    print('WARNING: HTML includes source code; do not publish publicly.')

if __name__ == '__main__': main()
