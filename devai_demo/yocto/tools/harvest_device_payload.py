"""Harvest the device-verified V25 payloads for meta-ara-demo into tarballs on the PC.

Each payload is streamed with ``tar`` over SSH (nothing is written on the device,
whose root file system has only about 215 MB free).  The result is
``<out>/<payload>.tar`` plus ``<out>/payload_manifest.json`` with sha256, size and
file count, which the ara-demo-payload bbclass checks before installing.

The release payload is additionally checked file by file against the device's
``release/manifest.json``.

Usage (PowerShell, from D:\\source\\repos\\ara):
  python -B .\\multimodal-emotion-analyzer\\devai_demo\\yocto\\tools\\harvest_device_payload.py --out D:\\ara_payload
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'operator'))
from ara06_service_control import HOST, OPTIONS  # noqa: E402

# name: (parent directory on the device, top entry, tar --exclude patterns)
PAYLOADS = {
    'ara06-recording-app': ('/root/devai_ara', 'ara06_recording_app_20261014_v1',
                            ['ara06_recording_app_20261014_v1/workspace/runs']),
    'ara-demo-release-v25': ('/root/devai_ara/ara_demo_20261026_v25', 'release',
                             ['release/workspace/runs/*']),
    'ara-llm-runtime': ('/root/devai_ara', 'ara06_minimal_llm_20261013_v1 services/ara_llm',
                        ['ara06_minimal_llm_20261013_v1/uv_cache', 'ara06_minimal_llm_20261013_v1/xet_cache',
                         'ara06_minimal_llm_20261013_v1/minimal_probe.log', 'services/ara_llm/ara_audit.jsonl']),
    'qwen25-7b-ara240': ('/usr/share/llm', 'Qwen2.5-7B-Instruct', []),
}


def harvest(name, out):
    parent, tops, excludes = PAYLOADS[name]
    target = out / f'{name}.tar'
    if target.exists():
        raise FileExistsError(target)
    command = ' '.join(['tar', '-C', parent, '--numeric-owner', '-cf', '-',
                        *[f"--exclude='{pattern}'" for pattern in excludes], tops])
    started = time.monotonic()
    digest = hashlib.sha256()
    with target.open('xb') as stream:
        process = subprocess.Popen(['ssh', *OPTIONS, HOST, command], stdout=subprocess.PIPE)
        for chunk in iter(lambda: process.stdout.read(1 << 20), b''):
            digest.update(chunk)
            stream.write(chunk)
        if process.wait() != 0:
            raise RuntimeError(f'{name}: remote tar failed ({process.returncode})')
    files = 0
    with tarfile.open(target) as archive:
        for member in archive:
            files += member.isfile()
    return {'sha256': digest.hexdigest(), 'bytes': target.stat().st_size, 'files': files,
            'device_parent': parent, 'device_entries': tops.split(), 'excluded': excludes,
            'seconds': round(time.monotonic() - started, 1)}


def check_release(out):
    """Every file the device release manifest lists must be in the tarball with that sha256."""
    with tarfile.open(out / 'ara-demo-release-v25.tar') as archive:
        names = {member.name: member for member in archive.getmembers() if member.isfile()}
        manifest = json.loads(archive.extractfile(names['release/manifest.json']).read())
        bad = [path for path, expected in manifest['files'].items()
               if f'release/{path}' not in names or
               hashlib.sha256(archive.extractfile(names[f'release/{path}']).read()).hexdigest() != expected]
    if bad:
        raise RuntimeError(f'release payload differs from release/manifest.json: {bad[:10]}')
    return len(manifest['files'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--only', nargs='*', choices=sorted(PAYLOADS))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / 'payload_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {'payloads': {}}
    for name in args.only or PAYLOADS:
        print(f'harvesting {name} ...', flush=True)
        manifest['payloads'][name] = harvest(name, args.out)
        print(json.dumps({name: manifest['payloads'][name]}), flush=True)
    if 'ara-demo-release-v25' in manifest['payloads']:
        manifest['release_manifest_files_verified'] = check_release(args.out)
    manifest.update(source_host=HOST, harvested_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                    release='ara_demo_20261026_v25', release_commit='6723061')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
