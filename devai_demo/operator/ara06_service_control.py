"""Own one immutable ARA service attempt; preserve old logs and reject stale PIDs."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess

_PARENTS = list(Path(__file__).resolve().parents)
ROOT = next((p for p in _PARENTS if (p / 'NEXT_STEP.md').exists()),
            next((p for p in _PARENTS if (p / '.git').exists()), Path(__file__).resolve().parents[2]))
STRUCTURED = ROOT / 'devai/devai_ara/runs/ara06_structured_client_20261013_v1'
RUN = ROOT / 'devai/devai_ara/runs/ara06_dev20_20261017_v1'
REMOTE = '/root/devai_ara/ara06_structured_client_20261013_v1'
HOST = 'root@192.168.1.208'
OPTIONS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=yes',
           '-o', f'UserKnownHostsFile={ROOT.as_posix()}/.tmp_devai_ara/imx95_known_hosts']

# Passed through stdin to python3, never interpolated into a remote shell command.
WORKER = r'''
import hashlib, json, os, signal, socket, subprocess, time, urllib.request
from pathlib import Path
TASK = Path('/root/devai_ara/ara06_structured_client_20261013_v1')
PYTHON = '/root/devai_ara/ara06_minimal_llm_20261013_v1/venv/bin/python'
def identity(pid):
    try:
        proc = Path('/proc') / str(pid)
        stat = (proc / 'stat').read_text().rsplit(') ', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return {'pid': pid, 'start_ticks': stat[19],
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'argv': (proc / 'cmdline').read_bytes().decode().strip('\0').split('\0')}
    except FileNotFoundError:
        return None
def free_port():
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', 18081))
def health():
    try:
        with urllib.request.urlopen('http://127.0.0.1:18081/health', timeout=2) as response:
            return json.load(response)
    except Exception as error:
        return {'state': 'unreachable', 'error': str(error)}
def execute(config):
    action = config['action']
    if action == 'prepare':
        free_port()
        # No concurrent dedicated service, even if bound to another port.
        for path in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                if b'modules.ara_llm_service' in path.read_bytes().split(b'\0'):
                    raise RuntimeError('Dedicated service already running')
            except FileNotFoundError:
                pass
        attempt = TASK / config['attempt']
        attempt.mkdir(exist_ok=False)
        (attempt / 'python/modules').mkdir(parents=True)
        return {'prepared': str(attempt)}
    if action == 'start':
        free_port()
        attempt = TASK / config['attempt']
        if (attempt / 'service_identity.json').exists() or (attempt / 'service.log').exists():
            raise RuntimeError('Attempt already started; choose a fresh attempt')
        for name, digest in config['source_sha256'].items():
            if hashlib.sha256((attempt / 'python/modules' / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('Deployed source hash mismatch')
        env = os.environ.copy()
        env.pop('DV_TGT_ROOT', None)
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2',
                   PYTHONPATH=str(attempt / 'python'))
        audit = str(attempt / 'audit.jsonl')
        with (attempt / 'service.log').open('xb') as log:
            proc = subprocess.Popen([PYTHON, '-B', '-u', '-m', 'modules.ara_llm_service',
                '--port', '18081', '--audit', audit, '--prompt-format', config['prompt_format']],
                cwd=attempt, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        owned = identity(proc.pid)
        if owned is None:
            raise RuntimeError('Service exited during startup; inspect service.log')
        result = {**config, **owned, 'remote': str(TASK), 'service_pid': str(proc.pid),
                  'result_dir': config['attempt'], 'audit_path': audit}
        (attempt / 'service_identity.json').write_text(json.dumps(result, indent=2) + '\n')
        (TASK / 'service.pid').write_text(str(proc.pid) + '\n')
        return result
    owned = config['identity']
    current = identity(int(owned['pid']))
    matches = current is not None and all(current[key] == owned[key] for key in ('pid', 'boot_id', 'start_ticks', 'argv'))
    expected_audit = str(TASK / owned['result_dir'] / 'audit.jsonl')
    if owned['audit_path'] != expected_audit:
        raise RuntimeError('Audit ownership mismatch')
    if action == 'status':
        return {'identity_matches': matches, 'current_identity': current, 'health': health()}
    if current is not None and not matches:
        raise RuntimeError('Stale or reused PID; no signal sent')
    if current is not None:
        argv = current['argv']
        if 'modules.ara_llm_service' not in argv or '--audit' not in argv or argv[argv.index('--audit') + 1] != expected_audit:
            raise RuntimeError('Service argv ownership mismatch; no signal sent')
        state = health()
        if state.get('inference_active') is not False or state.get('state') not in ('ready', 'failed'):
            raise RuntimeError('Wait for native load/inference to finish before stopping')
        os.kill(current['pid'], signal.SIGINT)
        for _ in range(100):
            if identity(current['pid']) is None:
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Owned service did not exit; no forced kill')
    free_port()
    return {'status': 'stopped', 'identity_matched_before_signal': matches, 'health': health()}
'''


def remote(config):
    RUN.mkdir(parents=True, exist_ok=True)
    code = WORKER + '\nprint(json.dumps(execute(' + repr(config) + ')))\n'
    result = subprocess.run(['ssh', *OPTIONS, HOST, 'python3 -'], input=code.encode(),
                            capture_output=True, timeout=30)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    record = {'action': config['action'], 'returncode': result.returncode,
              'stdout': result.stdout.decode(errors='replace'), 'stderr': result.stderr.decode(errors='replace')}
    (RUN / f'control_{stamp}.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    if result.returncode:
        raise RuntimeError(record['stderr'])
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'status', 'stop'))
    parser.add_argument('--attempt')
    parser.add_argument('--prompt-format', choices=('legacy-v1', 'compact-json-v1'), default='compact-json-v1')
    args = parser.parse_args()
    if args.action == 'start':
        if not args.attempt or not re.fullmatch(r'attempt_\d{2,4}', args.attempt) or int(args.attempt[8:]) < 5:
            parser.error('Fresh attempt_05 or later required')
        target = STRUCTURED / args.attempt
        target.mkdir(exist_ok=False)
        stage = target / 'python/modules'
        stage.mkdir(parents=True)
        for name in ('ara_llm_service.py', 'ara_json_logits.py'):
            (stage / name).write_bytes((ROOT / 'multimodal-emotion-analyzer/modules' / name).read_bytes())
        (stage / '__init__.py').write_text('', encoding='utf-8')
        hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in stage.iterdir()}
        remote({'action': 'prepare', 'attempt': args.attempt})
        subprocess.run(['scp', *OPTIONS, *map(str, stage.iterdir()),
                        f'{HOST}:{REMOTE}/{args.attempt}/python/modules/'], check=True, timeout=30)
        result = remote({'action': 'start', 'attempt': args.attempt,
                         'prompt_format': args.prompt_format, 'source_sha256': hashes})
        (target / 'source_manifest.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        (target / 'service_identity.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        (STRUCTURED / 'active_attempt.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        (STRUCTURED / 'service.pid').write_text(result['service_pid'] + '\n', encoding='utf-8')
    else:
        active = json.loads((STRUCTURED / 'active_attempt.json').read_text())
        if not re.fullmatch(r'attempt_\d{2,4}', active['result_dir']):
            raise ValueError('Invalid active attempt')
        owned = json.loads((STRUCTURED / active['result_dir'] / 'service_identity.json').read_text())
        result = remote({'action': args.action, 'identity': owned})
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
