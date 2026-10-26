"""Offload one explicitly selected stopped evidence tree with three hash copies."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ara06_service_control import ROOT, HOST, OPTIONS
import ara_demo as demo


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def inventory(root):
    assert not any(p.is_symlink() for p in root.rglob('*'))
    return {p.relative_to(root).as_posix(): dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            bytes=p.stat().st_size) for p in root.rglob('*') if p.is_file()}


def offload(release, session, trial, destination, allow_error=False):
    demo.select_release(release)
    assert session and session.replace('_', '').isalnum()
    trial = trial.resolve(); destination = destination.resolve()
    assert trial.is_relative_to(ROOT.resolve()) and destination.is_relative_to(ROOT.resolve())
    remote = demo.RELEASE + '/workspace/runs/' + session
    receipt = read(trial / 'collection_receipt.json'); expected = read(trial / 'remote_manifest.json')
    assert receipt['all_hashes_match'] and receipt['source'] == remote
    assert inventory(trial / 'evidence') == expected
    proof = read(trial / 'evidence/end_proof.json')
    if not allow_error:
        assert proof['error'] is None
    assert proof['desktop_restored'] and not proof['device_holders']
    if proof.get('mode') == 'framebuffer_render_only':
        assert proof['new_recording_windows'] == 0 and proof['model_inferences'] == 0
    else:
        assert proof['all_owned_processes_exited'] and proof['workers_exited'] and not proof['cleanup_errors']
        assert all(proof['threads_stopped'].values())
    destination.mkdir(exist_ok=False)
    restore = destination / 'restore_sandbox/1'
    restore.parent.mkdir()
    shutil.copytree(trial / 'evidence', restore)
    assert inventory(restore) == expected
    demo.save(destination / 'archive_restore_1.json', dict(original=str(trial / 'evidence'),
        restore=str(restore), remote=remote, all_hashes_match=True, files=len(expected), manifest=expected))
    code = f'''import json,hashlib,runpy,shutil
from pathlib import Path
release=Path({demo.RELEASE!r});api=runpy.run_path(str(release/'runtime.py'))
p=Path({remote!r});assert p.resolve()==p and p.parent==release/'workspace/runs'
is_resident = (api['health']()['state'] == 'ready')
assert not api['holders']() and (api['health']()['state'] in ('ready', 'unreachable'))
assert all(v=='active' for v in api['desktop']().values())
for q in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  argv=q.read_bytes()
  assert not any(n in argv for n in (b'modules.live_pipeline',b'observation_worker.py',b'/release/runtime.py'))
  if not is_resident:
   assert b'modules.ara_llm_service' not in argv
 except FileNotFoundError:pass
for name in ('probe_identity.json','supervisor_identity.json','pipeline_identity.json','ara_identity.json','service_identity.json'):
 if (p/name).exists():assert not api['matches'](json.loads((p/name).read_text())),name
assert not (p/'OFFLOADED_TO_PC.json').exists()
assert all(not q.is_symlink() for q in p.rglob('*'))
files={{q.relative_to(p).as_posix():dict(sha256=hashlib.sha256(q.read_bytes()).hexdigest(),bytes=q.stat().st_size) for q in p.rglob('*') if q.is_file()}}
print(json.dumps(dict(files=files,free_bytes=shutil.disk_usage('/').free)))
'''
    before = json.loads(demo.ssh(code, 'offload_before')['stdout'])
    assert before['files'] == expected
    archive = destination / 'board_archive/evidence_1'
    archive.parent.mkdir()
    subprocess.run(['scp', *OPTIONS, '-r', f'{HOST}:{remote}', str(archive)], check=True, timeout=180)
    assert inventory(archive) == expected
    after = json.loads(demo.ssh(code, 'offload_recheck')['stdout'])
    assert after['files'] == expected
    marker = dict(remote_source=remote, pc_original=str(trial / 'evidence'),
        pc_restore_sandbox=str(restore), pc_archive=str(archive), files=expected,
        bytes=sum(v['bytes'] for v in expected.values()), pc_all_hashes_verified=True,
        restore='Verify saved hashes, copy exact archived contents back, verify all remote hashes, then remove only the marker')
    demo.save(destination / 'board_archive/evidence_1_receipt.json', marker)
    value = json.loads(demo.ssh(f'''import json,hashlib,shutil
from pathlib import Path
p=Path({remote!r});assert p.resolve()==p and str(p)=={remote!r}
expected={expected!r}
assert all(not q.is_symlink() for q in p.rglob('*'))
actual={{q.relative_to(p).as_posix():dict(sha256=hashlib.sha256(q.read_bytes()).hexdigest(),bytes=q.stat().st_size) for q in p.rglob('*') if q.is_file()}}
assert actual==expected
assert not (p/'OFFLOADED_TO_PC.json').exists()
(p/'OFFLOADED_TO_PC.json').write_text(json.dumps({marker!r},indent=2))
for name in expected:
 target=p/name;assert target.resolve().is_relative_to(p)
 target.unlink()
for q in sorted([q for q in p.rglob('*') if q.is_dir()],key=lambda q:len(q.parts),reverse=True):q.rmdir()
assert sorted(q.name for q in p.iterdir())==['OFFLOADED_TO_PC.json']
print(json.dumps(dict(offloaded_files=len(expected),free_bytes=shutil.disk_usage('/').free)))
''', 'offload_complete')['stdout'])
    demo.save(destination / 'offload_result.json', dict(**value, free_before_bytes=before['free_bytes']))
    print(json.dumps(value))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', required=True)
    parser.add_argument('--session', required=True)
    parser.add_argument('--trial', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--allow-error', action='store_true', help='Allow offloading sessions stopped with a recorded error')
    args = parser.parse_args()
    offload(args.release, args.session, args.trial, args.destination, allow_error=args.allow_error)
