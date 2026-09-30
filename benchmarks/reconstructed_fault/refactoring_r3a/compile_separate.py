#!/usr/bin/env python3
import json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
for i,command in enumerate(json.loads((root/'evidence/separate-commands.json').read_text())):
 subprocess.run(['python3',str(root/'run_logged.py'),f'separate-compile-{i}','300',*command],check=True)
