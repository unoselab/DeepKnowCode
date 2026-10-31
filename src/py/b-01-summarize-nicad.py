#!/usr/bin/env python3
"""Inventory NiCad XML output, with explicitly scoped counts and failure if absent."""
import argparse, csv, json, shutil, sys, xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',required=True)
    ap.add_argument('--stage',required=True)
    ap.add_argument('--nicad-home',required=True)
    ap.add_argument('--config',required=True)
    a=ap.parse_args()
    out=Path(a.output).resolve();stage=Path(a.stage).resolve();home=Path(a.nicad_home).resolve()
    xml=[]
    # Search bounded directories only; ignore other experiment outputs.
    # NiCad keys its cache/output by staged input basename. Scope strictly to
    # the fixed B-01 stage name after removing stale cache before execution.
    cache_dir=home/'nicadclones'/stage.name
    if cache_dir.is_dir():
        for p in cache_dir.rglob('*.xml'):
            if p.is_file() and ('clone' in p.name.lower() or 'class' in p.name.lower()): xml.append(p)
    xml=sorted(set(xml))
    rows=[]; counts=Counter();errors=[]
    dest=out/'nicad_xml';dest.mkdir(exist_ok=True)
    for i,p in enumerate(xml):
        # Do not overwrite and do not move NiCad's originals.
        target=dest/(f'{i:03d}_'+p.name)
        if p.resolve()!=target.resolve():shutil.copy2(p,target)
        try:
            root=ET.parse(p).getroot()
            clone_nodes=sum(1 for x in root.iter() if x.tag.rsplit('}',1)[-1].lower()=='clone')
            class_nodes=sum(1 for x in root.iter() if x.tag.rsplit('}',1)[-1].lower()=='class')
            counts['xml_files']+=1
            rows.append({'file':str(p),'saved_as':str(target),'root':root.tag,'clone_elements':clone_nodes,'class_elements':class_nodes})
            counts['clone_elements']+=clone_nodes;counts['class_elements']+=class_nodes
        except ET.ParseError as e:errors.append(f'{p}: {e}')
    with (out/'nicad_xml_inventory.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['file','saved_as','root','clone_elements','class_elements']);w.writeheader();w.writerows(rows)
    report={'nicad_cache_dir':str(cache_dir),'staged_input_name':stage.name,'status':'COMPLETE' if rows else 'NO_XML_FOUND','config':a.config,'xml_files':len(rows),
            'clone_elements_all_xml':counts['clone_elements'],'class_elements_all_xml':counts['class_elements'],
            'warnings':['Counts may duplicate the same clones across NiCad XML formats; inspect each XML separately.',
                        'NiCad findings are candidates, not validated ground truth.'], 'parse_errors':errors,
            'xml_inventory':'nicad_xml_inventory.csv'}
    (out/'nicad_summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f"[B-01] XML files parsed: {len(rows)}, XML clone elements (not deduplicated): {counts['clone_elements']}")
    for r in rows:print(f"  {Path(r['file']).name}: clone={r['clone_elements']}, class={r['class_elements']}")
    if errors:print('WARNING: XML parsing errors:',*errors,sep='\n  ',file=sys.stderr)
    if not rows:
        print(f'ERROR: No XML under current run cache: {cache_dir}',file=sys.stderr)
        return 5
    return 0
if __name__=='__main__':raise SystemExit(main())
