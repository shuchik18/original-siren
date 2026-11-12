import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys

repo, destination = map(Path, sys.argv[1:])
sys.path.insert(0,str(repo))
from siren.parser import parse_program
from siren.evaluate import evaluate
from llcsiren.src.compiler import compile_program,compile_source
cases={'gaussian-gaussian':'gaussuan-gaussian','gtree':'gtree','kalman':'kalman','wheels':'wheels-example','slam':'slam-example','outlier':'outlier','coin':'beta-example','gaussian-gaussian-hybrid':'gaussian-gaussian-hybrid'}
destination.mkdir(parents=True,exist_ok=True)
manifest={}
for name,folder in cases.items():
 source=(repo/'tests/test_llcsiren'/folder/'source.si').read_text()
 program= parse_program(source)
 with contextlib.redirect_stdout(io.StringIO()):
  direct=compile_program(program)
  transformed=evaluate(program,str(destination))
  generated=compile_source(transformed) if isinstance(transformed,str) else compile_program(transformed)
 (destination/f'{name}-direct.py').write_text(direct)
 (destination/f'{name}-compiled.py').write_text(generated)
 manifest[name]={'direct':hashlib.sha256(direct.encode()).hexdigest(),'compiled':hashlib.sha256(generated.encode()).hexdigest()}
 print(name,'generated',flush=True)
(destination/'manifest.json').write_text(json.dumps(manifest,indent=2))
