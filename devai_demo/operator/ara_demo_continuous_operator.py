"""SSH-only exhibition autonomous continuous rolling demo operator.

Enables hands-free autonomous continuous operation:
- Automatically acquires rolling 8s windows (camera + mic)
- Runs real-time visual emotion recognition and ASR
- Prompts resident Qwen2.5-7B (timeout: 30s) for natural Japanese companion dialogue
- Continuously purges older windows (retaining only latest 3 windows, disk permanently safe > 214 MiB)
- Provides real-time status monitoring, on-demand HDMI screenshot capture, and safe stop/collection.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = next((p for p in Path(__file__).resolve().parents if (p / '.git').exists() or (p / 'NEXT_STEP.md').exists()), Path(__file__).resolve().parents[2])
OPERATOR_DIR = Path(__file__).resolve().parent
SCRIPTS = ROOT / 'devai/devai_ara/scripts'
sys.path.insert(0, str(OPERATOR_DIR))
sys.path.insert(1, str(SCRIPTS))

import ara_demo as demo
from ara06_service_control import HOST, OPTIONS


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


class ContinuousOperator:
    def __init__(self, release, session):
        if release not in ('ara_demo_20261025_v22', 'ara_demo_20261026_v23', 'ara_demo_20261026_v24'):
            raise ValueError('Use reviewed v22, v23, or v24 release')
        if not re.fullmatch(r'demo_(?:continuous|image)_[0-9]{8}_[0-9]{6}_[a-zA-Z0-9_]+', session):
            raise ValueError('Use demo_continuous_YYYYMMDD_HHMMSS_suffix or demo_image_ session name')
        demo.select_release(release)
        self.session = session
        self.release = demo.RELEASE
        self.remote = self.release + '/workspace/runs/' + session
        self.case = ROOT / 'devai/devai_ara/runs' / session
        self.header = f'''import json,runpy,time,shutil
from pathlib import Path
api=runpy.run_path({self.release!r}+'/runtime.py')
p=Path({self.remote!r})
'''

    def remote_call(self, code, label):
        receipt = demo.ssh(self.header + code, 'cont_' + label)
        return json.loads(receipt['stdout'])

    def require_case(self):
        protocol = read(self.case / 'protocol.json')
        if protocol['session'] != self.session or protocol['release'] != self.release:
            raise ValueError('Local protocol mismatch')

    def preflight(self):
        self.case.mkdir(parents=True, exist_ok=False)
        (self.case / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        protocol = dict(
            session=self.session,
            release=self.release,
            mode='autonomous_continuous_rolling',
            window_seconds=8,
            minimum_free_bytes=214 * 2**20,
            autonomous_storage_protection=True,
            resident_qwen_companion_dialogue=True,
            user_preparation_reply_required=True,
            created_utc=datetime.now(timezone.utc).isoformat()
        )
        save(self.case / 'protocol.json', protocol)
        self.refresh(full=True)

    def refresh(self, full=False):
        self.require_case()
        code = f'''d=api['doctor'](full={full!r})
is_resident=(d['health']['state']=='ready')
processes=[]
for q in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  argv=q.read_bytes().split(bytes([0]))
  if b'modules.live_pipeline' in argv or any(x.endswith(b'/observation_worker.py') for x in argv):processes.append(int(q.parent.name))
  elif not is_resident and b'modules.ara_llm_service' in argv:processes.append(int(q.parent.name))
 except OSError:pass
out=dict(doctor=d,processes=processes,session_absent=not p.exists(),
 boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),sampled_ns=time.monotonic_ns(),resident_ara=is_resident)
print(json.dumps(out))
'''
        value = self.remote_call(code, 'preflight')
        save(self.case / f'preflight_{value["sampled_ns"]}.json', value)
        d = value['doctor']
        is_resident = value.get('resident_ara', False)
        valid_health = (d['health']['state'] == 'ready') if is_resident else (d['health']['state'] == 'unreachable')
        if not (d['ok'] and d['root_free_bytes'] >= 214 * 2**20
                and not d['device_holders'] and not value['processes']
                and valid_health and value['session_absent']):
            raise RuntimeError('Preflight rejected; inspect saved receipt, do not start')
        print(json.dumps(value, ensure_ascii=False, indent=2))

    def start(self, reply):
        self.refresh()
        save(self.case / 'startup_user_readiness.json', dict(
            session=self.session, reply=reply, received_utc=datetime.now(timezone.utc).isoformat(),
            autonomous_continuous_mode=True
        ))
        value = self.remote_call(f'''assert shutil.disk_usage('/').free>=214*2**20
print(json.dumps(api['execute']('start',{self.session!r},False,False)))
''', 'start')
        save(self.case / 'start_receipt.json', value)
        print("Continuous session started successfully:", json.dumps(value, ensure_ascii=False, indent=2))

    def status(self):
        self.require_case()
        code = f'''out=api['execute']('status',{self.session!r})
out['sampled_ns']=time.monotonic_ns()
out['boot_id']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
windows_dir = p / 'live/windows'
if windows_dir.is_dir():
    out['recorded_windows'] = len(list(windows_dir.glob('*/capture_manifest.json')))
    out['active_windows'] = sorted([w.name for w in windows_dir.iterdir() if w.is_dir()])
else:
    out['recorded_windows'] = 0
    out['active_windows'] = []
out['display'] = api['read'](p/'live/status.json') if (p/'live/status.json').exists() else None
out['root_free_bytes'] = shutil.disk_usage('/').free
print(json.dumps(out))
'''
        value = self.remote_call(code, 'status')
        save(self.case / f'status_{value["sampled_ns"]}.json', value)
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return value

    def screenshot(self, label='latest'):
        self.require_case()
        target_png = self.case / f'screenshot_{label}_{int(time.time())}.png'
        code = f'''from pathlib import Path
png_path = p / 'live/display_latest.png'
assert png_path.is_file(), 'display_latest.png not found yet'
print(json.dumps(dict(png_bytes=png_path.stat().st_size)))
'''
        res = self.remote_call(code, 'screenshot')
        # SCP copy to local PC
        subprocess.run(['scp', *OPTIONS, f'{HOST}:{self.remote}/live/display_latest.png', str(target_png)],
                       check=True, timeout=30)
        print(f"Screenshot successfully retrieved and saved to: {target_png} ({target_png.stat().st_size} bytes)")
        return target_png

    def stop(self):
        self.require_case()
        value = self.remote_call(f"print(json.dumps(api['execute']('stop',{self.session!r})))\n", 'stop')
        save(self.case / 'stop_receipt.json', value)
        print("Session stopped:", json.dumps(value, ensure_ascii=False, indent=2))

    def collect(self):
        self.require_case()
        target = self.case / 'trial'
        target.mkdir(parents=True, exist_ok=False)
        code = '''state=api['read'](p/'state.json');assert state['stage'] in ('stopped','failed')
assert (p/'end_proof.json').exists()
for name in ('supervisor_identity.json','pipeline_identity.json','ara_identity.json','service_identity.json'):
 if (p/name).exists():assert not api['matches'](api['read'](p/name)),name
files={}
import hashlib
for q in p.rglob('*'):
 assert not q.is_symlink(),str(q)
 if q.is_file():files[q.relative_to(p).as_posix()]=dict(sha256=hashlib.sha256(q.read_bytes()).hexdigest(),bytes=q.stat().st_size)
print(json.dumps(files))
'''
        save(target / 'collection_source.json', dict(code=self.header + code))
        before = self.remote_call(code, 'collect_before')
        save(target / 'remote_manifest.json', before)
        subprocess.run(['scp', *OPTIONS, '-r', f'{HOST}:{self.remote}', str(target / 'evidence')],
                       check=True, timeout=180)
        after = self.remote_call(code, 'collect_after')
        save(target / 'remote_manifest_after.json', after)
        local = {q.relative_to(target / 'evidence').as_posix(): dict(
            sha256=hashlib.sha256(q.read_bytes()).hexdigest(), bytes=q.stat().st_size)
            for q in (target / 'evidence').rglob('*') if q.is_file()}
        if before != after or before != local:
            raise RuntimeError('Collection mismatch; retain partial evidence and investigate')
        receipt = dict(
            session=self.session,
            files=len(local),
            bytes=sum(v['bytes'] for v in local.values()),
            all_hashes_match=True,
            remote_stable_before_after=True,
            source=self.remote
        )
        save(target / 'collection_receipt.json', receipt)
        print("Collection completed successfully:", json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'start', 'status', 'screenshot', 'stop', 'collect'])
    parser.add_argument('--release', default='ara_demo_20261026_v23')
    parser.add_argument('--session', required=True)
    parser.add_argument('--reply', help='User explicit readiness confirmation reply')
    parser.add_argument('--label', default='current', help='Label for screenshot')
    args = parser.parse_args()

    if args.action == 'start' and not (args.reply and args.reply.strip()):
        parser.error('A fresh actual user readiness reply is required in --reply')

    operator = ContinuousOperator(args.release, args.session)
    if args.action == 'start':
        operator.start(args.reply)
    elif args.action == 'screenshot':
        operator.screenshot(args.label)
    else:
        getattr(operator, args.action)()
