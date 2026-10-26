"""Save a generated forecast on the latest main without rebasing SVG edits.

Intended for the clean GitHub Actions checkout. Only the two generated files
may be modified; all other local changes cause an immediate refusal.
"""
from pathlib import Path
import json
import subprocess
import xml.etree.ElementTree as ET

FILES=('docs/final_surf_forecast.svg','.forecast_state/latest_gfs_cycle.txt')


def git(*args):
    return subprocess.check_output(['git',*args],text=True).strip()


def save():
    changed=set(git('diff','--name-only').splitlines()) | set(git('diff','--cached','--name-only').splitlines())
    if changed-set(FILES):
        raise RuntimeError('Refusing to save forecast with unrelated local changes')
    snapshot={name:Path(name).read_bytes() for name in FILES}
    svg=ET.fromstring(snapshot[FILES[0]])
    embedded=next(e for e in svg.iter() if e.attrib.get('id')=='sykemeter-model')
    generated_config=json.loads(embedded.text)
    generated_cycle=snapshot[FILES[1]].decode().strip()
    # Restore ONLY the captured generated files, then fast-forward before committing.
    # A complete SVG is reapplied; no line-level SVG merge is attempted.
    git('restore','--source=HEAD','--staged','--worktree','--',*FILES)
    git('pull','--ff-only','origin','main')
    latest_config=json.loads(Path('model_config.json').read_text())
    latest_cycle=Path(FILES[1]).read_text().strip()
    if generated_config!=latest_config:
        print('A newer model configuration is on main; skipped stale forecast commit.')
        return
    if latest_cycle>generated_cycle:
        print('A newer forecast cycle is on main; skipped older generated forecast.')
        return
    for name,data in snapshot.items():
        Path(name).write_bytes(data)
    git('add','--',*FILES)
    if not git('diff','--cached','--name-only'):
        print('No generated changes to commit.')
        return
    git('commit','-m',f'Update forecast {generated_cycle}')
    git('push','origin','HEAD:main')
    print(f'Saved forecast {generated_cycle} on the latest main.')


if __name__=='__main__':
    save()
