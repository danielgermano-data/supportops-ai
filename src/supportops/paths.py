from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if ROOT.name == 'src':
    ROOT = ROOT.parent
DATA = ROOT / 'data'
ARTIFACTS = ROOT / 'artifacts'
WEB = ROOT / 'web'
