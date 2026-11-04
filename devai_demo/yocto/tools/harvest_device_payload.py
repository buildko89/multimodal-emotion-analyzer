"""Harvest the device-verified V25 payloads for meta-ara-demo into tarballs on the PC.

Each payload is streamed with ``tar`` over SSH (nothing is written on the device,
whose root file system has only about 215 MB free).  The result is
``<out>/<payload>.tar`` plus ``<out>/payload_manifest.json`` with sha256, size and
file count, which the ara-demo-payload bbclass checks before installing.

The release payload is additionally checked file by file against the device's
``release/manifest.json``.

The transfer goes to ``<payload>.tar.part`` and is renamed only after it succeeded
and was verified, and the manifest is saved after every payload, so an interrupted
run can simply be run again:

* a tarball matching its manifest entry (size and sha256) is kept and skipped;
* a tarball without a matching entry is incomplete and is harvested again;
* ``--force`` harvests again even if complete.  The old tarball stays in place
  until the new one has been verified.

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

RELEASE_PAYLOAD = 'ara-demo-release-v25'
# name: (parent directory on the device, top entry, tar --exclude patterns)
PAYLOADS = {
    'ara06-recording-app': ('/root/devai_ara', 'ara06_recording_app_20261014_v1',
                            ['ara06_recording_app_20261014_v1/workspace/runs']),
    RELEASE_PAYLOAD: ('/root/devai_ara/ara_demo_20261026_v25', 'release',
                      ['release/workspace/runs/*']),
    'ara-llm-runtime': ('/root/devai_ara', 'ara06_minimal_llm_20261013_v1 services/ara_llm',
                        ['ara06_minimal_llm_20261013_v1/uv_cache', 'ara06_minimal_llm_20261013_v1/xet_cache',
                         'ara06_minimal_llm_20261013_v1/minimal_probe.log', 'services/ara_llm/ara_audit.jsonl']),
    'qwen25-7b-ara240': ('/usr/share/llm', 'Qwen2.5-7B-Instruct', []),
}


def is_file_member(member):
    """A second name of a hard-linked file is stored as a link member, not a regular one."""
    return member.isfile() or member.islnk()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def already_harvested(name, out, manifest):
    """True only if the tarball is the complete one the manifest recorded."""
    target = out / f'{name}.tar'
    if not target.exists():
        return False
    entry = manifest['payloads'].get(name)
    if entry is None:
        print(f'{target} is not in the manifest (incomplete run); harvesting again', flush=True)
        return False
    if target.stat().st_size != entry['bytes'] or file_sha256(target) != entry['sha256']:
        print(f'{target} does not match the manifest; harvesting again', flush=True)
        return False
    return True


def harvest(name, out):
    parent, tops, excludes = PAYLOADS[name]
    target = out / f'{name}.tar'
    part = out / f'{name}.tar.part'
    command = ' '.join(['tar', '-C', parent, '--numeric-owner', '-cf', '-',
                        *[f"--exclude='{pattern}'" for pattern in excludes], tops])
    started = time.monotonic()
    digest = hashlib.sha256()
    process = subprocess.Popen(['ssh', *OPTIONS, HOST, command], stdout=subprocess.PIPE)
    try:
        with part.open('wb') as stream:
            for chunk in iter(lambda: process.stdout.read(1 << 20), b''):
                digest.update(chunk)
                stream.write(chunk)
        if process.wait() != 0:
            raise RuntimeError(f'{name}: remote tar failed ({process.returncode})')
        with tarfile.open(part) as archive:  # also rejects a truncated stream
            files = sum(is_file_member(member) for member in archive)
        entry = {'sha256': digest.hexdigest(), 'bytes': part.stat().st_size, 'files': files,
                 'device_parent': parent, 'device_entries': tops.split(), 'excluded': excludes,
                 'seconds': round(time.monotonic() - started, 1)}
        if name == RELEASE_PAYLOAD:
            entry['release_manifest_files_verified'] = check_release(part)
        part.replace(target)
    except BaseException:
        process.kill()
        process.wait()
        part.unlink(missing_ok=True)
        raise
    return entry


def check_release(tarball):
    """Every file the device release manifest lists must be in the tarball with that sha256."""
    with tarfile.open(tarball) as archive:
        names = {member.name: member for member in archive.getmembers() if is_file_member(member)}
        manifest = json.loads(archive.extractfile(names['release/manifest.json']).read())
        bad = [path for path, expected in manifest['files'].items()
               if f'release/{path}' not in names or
               hashlib.sha256(archive.extractfile(names[f'release/{path}']).read()).hexdigest() != expected]
    if bad:
        raise RuntimeError(f'release payload differs from release/manifest.json: {bad[:10]}')
    return len(manifest['files'])


def save_manifest(path, manifest):
    manifest.update(source_host=HOST, harvested_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                    release='ara_demo_20261026_v25', release_commit='6723061')
    temporary = path.with_name(path.name + '.part')
    temporary.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--only', nargs='*', choices=sorted(PAYLOADS))
    parser.add_argument('--force', action='store_true', help='harvest again even if the tarball is complete')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / 'payload_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {'payloads': {}}
    for name in args.only or PAYLOADS:
        if not args.force and already_harvested(name, args.out, manifest):
            print(f'{name} already harvested and verified; skipping', flush=True)
            continue
        print(f'harvesting {name} ...', flush=True)
        manifest['payloads'][name] = harvest(name, args.out)
        save_manifest(manifest_path, manifest)
        print(json.dumps({name: manifest['payloads'][name]}), flush=True)


if __name__ == '__main__':
    main()
