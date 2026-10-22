"""Automated quality expectation: at least 90% of frozen ordered triples pass.

Reports remain available on failure for manual listening/review. This threshold is
a stated initial interpretation of 'almost always', configurable by the owner.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('package',type=Path)
parser.add_argument('--min-pass-fraction',type=float,default=0.9)
args=parser.parse_args()
if not 0<=args.min_pass_fraction<=1:parser.error('fraction must be in [0,1]')
exe=args.package/('grain-elevator.exe' if sys.platform=='win32' else 'grain-elevator')
folder=root/'test-results';folder.mkdir(exist_ok=True)
report=folder/(args.package.name+'-quality.json')
subprocess.run([str(exe.resolve()),'synth-test','--report',str(report)],check=True)
data=json.loads(report.read_text())
data['required_fraction']=args.min_pass_fraction
data['expectation_met']=data['pass_fraction']>=args.min_pass_fraction
report.write_text(json.dumps(data,indent=2)+'\n')
print(f"Quality expectation {'PASS' if data['expectation_met'] else 'FAIL'}: {data['pass_fraction']:.0%} vs {args.min_pass_fraction:.0%}; {report}")
sys.exit(0 if data['expectation_met'] else 1)
