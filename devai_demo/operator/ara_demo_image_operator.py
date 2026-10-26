"""SSH-only pinned image demo operator. Never runs models or tests on the PC."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ara_demo as demo
from ara06_service_control import ROOT, HOST, OPTIONS


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


class Operator:
    def __init__(self, release, session):
        # Only the explicitly prepared configurations are accepted.
        if release not in ('ara_demo_20261023_v8', 'ara_demo_20261024_v9', 'ara_demo_20261024_v10', 'ara_demo_20261024_v11', 'ara_demo_20261024_v12', 'ara_demo_20261024_v13', 'ara_demo_20261024_v14', 'ara_demo_20261025_v15', 'ara_demo_20261025_v16', 'ara_demo_20261025_v17', 'ara_demo_20261025_v18', 'ara_demo_20261025_v19', 'ara_demo_20261025_v20', 'ara_demo_20261025_v21', 'ara_demo_20261025_v22', 'ara_demo_20261026_v23'):
            raise ValueError('Use the reviewed v8-v22 releases explicitly')
        if not re.fullmatch(r'demo_image_[0-9]{8}_[0-9]{6}_[a-zA-Z0-9_]+', session):
            raise ValueError('Use a fresh demo_image_YYYYMMDD_HHMMSS_suffix session')
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
        receipt = demo.ssh(self.header + code, 'image_' + label)
        return json.loads(receipt['stdout'])

    def require_case(self):
        protocol = read(self.case / 'protocol.json')
        if protocol['session'] != self.session or protocol['release'] != self.release:
            raise ValueError('Local protocol mismatch')

    def preflight(self):
        self.case.mkdir(exist_ok=False)
        (self.case / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        demo.save(self.case / 'protocol.json', dict(
            session=self.session, release=self.release, maximum_new_windows=1,
            window_seconds=8, minimum_free_bytes=214 * 2**20,
            current_startup_reply_required=True, separate_after_ready_reply_required=True,
            camera_and_microphone_captured=True, speech_required=False,
            action='Slowly nod once when the display shows recording',
            independent_quality_evaluated=False, pc_app_tests_run=0))
        self.refresh(full=True)

    def refresh(self, full=False):
        self.require_case()
        value = self.remote_call('''d=api['doctor'](full=FULL_CHECK)
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
'''.replace('FULL_CHECK', repr(full)), 'preflight')
        demo.save(self.case / ('preflight_' + str(value['sampled_ns']) + '.json'), value)
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
        demo.save(self.case / 'startup_user_readiness.json', dict(
            session=self.session, reply=reply, received_utc=datetime.now(timezone.utc).isoformat(),
            authorizes_recording=False))
        value = self.remote_call(f'''assert shutil.disk_usage('/').free>=214*2**20
print(json.dumps(api['execute']('start',{self.session!r},False,False)))
''', 'start')
        demo.save(self.case / 'start_receipt.json', value)
        print(json.dumps(value))

    def status(self):
        self.require_case()
        value = self.remote_call(f'''out=api['execute']('status',{self.session!r})
out['sampled_ns']=time.monotonic_ns()
out['boot_id']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
out['recorded_windows']=len(list((p/'live/windows').glob('*/capture_manifest.json')))
out['capture_command_exists']=(p/'live/capture_command.json').exists()
out['display']=api['read'](p/'live/status.json') if (p/'live/status.json').exists() else None
print(json.dumps(out))
''', 'status')
        demo.save(self.case / ('status_' + str(value['sampled_ns']) + '.json'), value)
        request = self.case / 'ready_confirmation_requested.json'
        if (value['state']['stage'] == 'ready' and value['recorded_windows'] == 0
                and not value['capture_command_exists'] and not request.exists()):
            demo.save(request, dict(session=self.session, sampled_ns=value['sampled_ns'],
                boot_id=value['boot_id'], ready_observed=True,
                prompt='今から8秒の画像確認に対応できますか。発話不要ですがカメラとマイクを取得します。回答後、開始を案内します。'))
        print(json.dumps(value, ensure_ascii=False, indent=2))

    def record(self, reply, countdown_seconds=10, prompt=None, window_seconds=8):
        self.require_case()
        request = read(self.case / 'ready_confirmation_requested.json')
        if request['session'] != self.session or not request['ready_observed']:
            raise ValueError('Current-session ready observation required')
        # The operator must have asked the user after ready and announced the start.
        # Replies are stored verbatim; the program cannot judge their meaning.
        display_prompt = prompt or 'ゆっくり1回うなずき、元の姿勢へ戻してください（映像・音声を記録中）'
        demo.save(self.case / 'immediate_action_readiness.json', dict(
            session=self.session, reply=reply, after_ready_prompt=True,
            microphone_capture_disclosed=True, speech_required=False,
            countdown_seconds=countdown_seconds, prompt=display_prompt,
            window_seconds=window_seconds,
            start_announced=True, received_utc=datetime.now(timezone.utc).isoformat()))
        code = f'''active=api['read'](Path({self.release!r})/'workspace/runs/active.json')
assert active['session']=={self.session!r} and active['release']=={self.release!r}
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()=={request['boot_id']!r}
state=api['read'](p/'state.json');assert state['stage']=='ready' and not state['check_only']
assert not (p/'stop.request').exists() and not (p/'end_proof.json').exists()
for name in ('supervisor_identity.json','pipeline_identity.json'):
 assert api['matches'](api['read'](p/name))
prep=api['read'](p/'live/preparation_state.json');assert prep['ready'] and prep['state']=='ready'
assert not list((p/'live/windows').glob('*/capture_manifest.json'))
assert not (p/'live/capture_command.json').exists()
command=dict(request_id='image_nod_01',not_before_ns=time.monotonic_ns()+{countdown_seconds!r}*10**9,
 window_seconds={window_seconds!r},
 prompt={display_prompt!r},
 operator_contract='only_after_explicit_user_ready_reply')
assert command['not_before_ns']>={request['sampled_ns']!r}
with (p/'recording_operator_ready.json').open('x') as stream:
 json.dump(dict(session={self.session!r},explicit_user_ready=True,command=command,
 scope='camera and microphone one window; video and audio recording; no speech required'),stream)
api['write'](p/'live/capture_command.json',command)
print(json.dumps(dict(recording_seconds={window_seconds!r},command=command)))
'''
        demo.save(self.case / 'record_source.json', dict(code=self.header + code))
        value = self.remote_call(code, 'record')
        demo.save(self.case / 'record_receipt.json', value)
        print(json.dumps(value, ensure_ascii=False))

    def stop(self):
        self.require_case()
        print(json.dumps(self.remote_call(
            f"print(json.dumps(api['execute']('stop',{self.session!r})))\n", 'stop')))

    def report(self, action_report, display_report, audio_report):
        self.require_case()
        demo.save(self.case / 'current_user_report.json', dict(session=self.session,
            action_report=action_report, display_report=display_report, audio_report=audio_report,
            received_utc=datetime.now(timezone.utc).isoformat(), independent_quality_evaluated=False))

    def collect(self):
        self.require_case()
        target = self.case / 'trial'
        target.mkdir(exist_ok=False)
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
        demo.save(target / 'collection_source.json', dict(code=self.header + code))
        before = self.remote_call(code, 'collect_before')
        demo.save(target / 'remote_manifest.json', before)
        subprocess.run(['scp', *OPTIONS, '-r', f'{HOST}:{self.remote}', str(target / 'evidence')],
                       check=True, timeout=120)
        after = self.remote_call(code, 'collect_after')
        demo.save(target / 'remote_manifest_after.json', after)
        local = {q.relative_to(target / 'evidence').as_posix(): dict(
            sha256=hashlib.sha256(q.read_bytes()).hexdigest(), bytes=q.stat().st_size)
            for q in (target / 'evidence').rglob('*') if q.is_file()}
        if before != after or before != local:
            raise RuntimeError('Collection mismatch; retain partial evidence and investigate')
        demo.save(target / 'collection_receipt.json', dict(session=self.session,
            files=len(local), bytes=sum(v['bytes'] for v in local.values()),
            all_hashes_match=True, remote_stable_before_after=True, source=self.remote))
        print(json.dumps(dict(files=len(local), all_hashes_match=True)))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'start', 'status', 'record', 'report', 'stop', 'collect'])
    parser.add_argument('--release', required=True)
    parser.add_argument('--session', required=True)
    parser.add_argument('--reply')
    parser.add_argument('--start-announced', action='store_true')
    parser.add_argument('--countdown-seconds', type=int, default=10,
                        help='Lead time after the start announcement; default 10 seconds')
    parser.add_argument('--window-seconds', type=int, default=8,
                        help='Recording window duration in seconds (1 to 30); default 8')
    parser.add_argument('--prompt', default=None,
                        help='Prompt text displayed on the screen during the recording window')
    parser.add_argument('--action-report')
    parser.add_argument('--display-report')
    parser.add_argument('--audio-report')
    args = parser.parse_args()
    if args.action in ('start', 'record') and not (args.reply and args.reply.strip()):
        parser.error('A fresh actual user reply is required in --reply')
    if args.action == 'record' and not args.start_announced:
        parser.error('Announce the start after the ready-stage reply, then use --start-announced')
    if not 0 <= args.countdown_seconds <= 15:
        parser.error('--countdown-seconds must be between 0 and 15')
    if not 1 <= args.window_seconds <= 30:
        parser.error('--window-seconds must be between 1 and 30')
    if args.action == 'report' and not all((args.action_report, args.display_report, args.audio_report)):
        parser.error('Supply all three verbatim reports; use 未確認 for unanswered items')
    operator = Operator(args.release, args.session)
    if args.action == 'record':
        operator.record(args.reply, args.countdown_seconds, args.prompt, args.window_seconds)
    elif args.action == 'start':
        getattr(operator, args.action)(args.reply)
    elif args.action == 'report':
        operator.report(args.action_report, args.display_report, args.audio_report)
    else:
        getattr(operator, args.action)()
