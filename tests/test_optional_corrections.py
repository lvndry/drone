"""Core workflows and schema reading must not import the optional adapter hash."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_blake3_is_optional_but_available_to_default_development() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert not any(dep.split(">")[0] == "blake3" for dep in project["project"]["dependencies"])
    assert "blake3>=1.0.11" in project["project"]["optional-dependencies"]["corrections"]
    assert "blake3>=1.0.11" in project["dependency-groups"]["dev"]


def test_missing_blake3_subprocess_preserves_core_and_schema_workflows(tmp_path: Path) -> None:
    # A fresh interpreter must behave as a base-only installation even though
    # the default development environment supplies blake3 for the other tests.
    program = r"""
import importlib.abc
import importlib.machinery
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
sys.path.insert(0,sys.argv[1])
attempted=[]
class MissingBlake3(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname:str,path:Sequence[str]|None=None,target:ModuleType|None=None) -> importlib.machinery.ModuleSpec|None:
        if fullname=='blake3' or fullname.startswith('blake3.'):
            attempted.append(fullname)
            raise ModuleNotFoundError('optional blake3 is absent',name='blake3')
        return None
sys.meta_path.insert(0,MissingBlake3())
import torch
import drone
import train
from drone.checkpoints import load_checkpoint,save_checkpoint
from drone.corrections import euphoria_content_hash,read_corrections_export
from drone.export import export_checkpoint
from drone.model import KeyCNN
from tools.synthetic_corrections import camelot
assert callable(drone.predict) and callable(train.train_model) and callable(export_checkpoint)
parsed=read_corrections_export(Path(sys.argv[1])/'tests/fixtures/euphoria-corrections.jsonl')
assert len(parsed.key_corrections)==1 and parsed.tempo_only_count==1
assert camelot(0,'major')=='8B'
torch.set_num_threads(1)
checkpoint=Path('synthetic.pt')
save_checkpoint(checkpoint,KeyCNN())
model,_=load_checkpoint(checkpoint,'cpu')
with torch.inference_mode():
    assert model(torch.zeros(1,192,8)).shape==(1,24)
assert attempted==[],attempted
try:
    euphoria_content_hash('not-opened-without-extra.wav')
except ValueError as error:
    message=str(error)
    assert 'uv sync --extra corrections' in message and 'drone-key[corrections]' in message
else:
    raise AssertionError('Hash matching must request its missing optional extra')
assert attempted==['blake3'],attempted
print('core imports, inference and schema parsing work; optional hash gives actionable extra error')
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, str(ROOT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "optional hash gives actionable extra error" in completed.stdout
