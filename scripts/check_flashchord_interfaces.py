"""Run relevant upstream GPU checks after materializing actual robot collision assets."""
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
LANE=ROOT/'vendor/video_to_data/robotic_grounding/flash_chord'
REPORT=ROOT/'notes/track3-flashchord-interfaces.json'
OUT=ROOT/'outputs/track3/episode_000011/flashchord_interfaces'
OUT.mkdir(exist_ok=True)


def main():
    assets=json.loads((ROOT/'notes/track3-flashchord-assets.json').read_text())
    assert assets['state']=='completed'
    report={'state':'running','started_utc':datetime.now(timezone.utc).isoformat(),
            'asset_bytes':assets['total_bytes'],'training':False,
            'interpretation':'Upstream robot assembly and contact/wrench implementation checks, not episode policy success.'}
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    testfiles=['tests/embodiments/test_sharpa_hands.py','tests/objectives/test_wrench.py',
               'tests/objectives/test_contact_wrench.py']
    result=subprocess.run([sys.executable,'-m','pytest','-q',*testfiles,'--junitxml='+str(OUT/'upstream-tests.xml')],cwd=LANE)
    tree=ET.parse(OUT/'upstream-tests.xml');suites=list(tree.getroot().iter('testsuite'))
    counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
    report.update(state='completed' if result.returncode==0 and counts['skipped']==0 else 'failed',
                  exit_code=result.returncode,counts=counts,completed_utc=datetime.now(timezone.utc).isoformat())
    REPORT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
    if report['state']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
