"""PC entry point for deploying and operating the pinned ARA+i.MX95 demo."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import re
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ara06_service_control import ROOT,HOST,OPTIONS

DEFAULT_RELEASE='ara_demo_20261018_v1'
NAME=DEFAULT_RELEASE
RUN=ROOT/'devai/devai_ara/runs'/NAME
REMOTE='/root/devai_ara/'+NAME
RELEASE=REMOTE+'/release'
PREVIOUS=ROOT/'devai/devai_ara/runs/ara05_step9_device_demo_20261018_v1'
OLD='/root/devai_ara/ara06_recording_app_20261014_v1/workspace'

def select_release(name):
    if not re.fullmatch(r'ara_demo_[0-9]{8}_v[1-9][0-9]*',name):
        raise ValueError('Expected an explicit ara_demo_YYYYMMDD_vN release name')
    global NAME,RUN,REMOTE,RELEASE
    NAME=name
    RUN=ROOT/'devai/devai_ara/runs'/NAME
    REMOTE='/root/devai_ara/'+NAME
    RELEASE=REMOTE+'/release'


def remap_release(value,old_prefix,new_prefix):
    if isinstance(value,dict):return {k:remap_release(v,old_prefix,new_prefix) for k,v in value.items()}
    if isinstance(value,list):return [remap_release(v,old_prefix,new_prefix) for v in value]
    if isinstance(value,str) and (value==old_prefix or value.startswith(old_prefix+'/')):
        return new_prefix+value[len(old_prefix):]
    return value


def deploy_required_bytes(package):
    # Keep 192MiB for start, plus actual payload and 32MiB copy margin.
    return 224*1024**2+sum(((p.stat().st_size+4095)//4096)*4096 for p in package.rglob('*') if p.is_file())


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')

def ssh(code,label,timeout=180,expected_failure=False):
    p=subprocess.run(['ssh',*OPTIONS,HOST,'python3 -'],input=code.encode(),capture_output=True,timeout=timeout)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    result=dict(release=NAME,remote_release=RELEASE,returncode=p.returncode,stdout=p.stdout.decode(errors='replace'),stderr=p.stderr.decode(errors='replace'))
    save(RUN/'logs'/(label+'_'+stamp+'.json'),result)
    if p.returncode and not expected_failure:raise RuntimeError(result['stderr'])
    return result

def capture_adapters(package):
    result=ssh(f'''import hashlib,json,os
from pathlib import Path
root=Path({OLD!r})/'devai/driver_eval'
records={{}}
for base,dirs,files in os.walk(root):
 dirs[:]=[name for name in dirs if name not in ('runtime','__pycache__','.venv')]
 for name in files:
  p=Path(base)/name
  records[p.relative_to(root).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
print(json.dumps(records))
''','capture_deployed_adapters')
    records=json.loads(result['stdout']);driver=package/'workspace/devai/driver_eval'
    for name,digest in records.items():
        target=driver/name;target.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['scp',*OPTIONS,f'{HOST}:{OLD}/devai/driver_eval/{name}',str(target)],check=True,timeout=30)
        assert hashlib.sha256(target.read_bytes()).hexdigest()==digest,name
    return records

def build(from_release=None):
    if from_release is not None:return build_from_release(from_release)
    if NAME!=DEFAULT_RELEASE:
        raise ValueError('New releases require --from-release to preserve a verified source configuration')
    package=RUN/'release';package.mkdir(parents=True,exist_ok=False)
    shutil.copytree(PREVIOUS/'workspace/multimodal-emotion-analyzer',package/'workspace/multimodal-emotion-analyzer')
    assets=package/'workspace/assets';assets.mkdir()
    shutil.copy2(PREVIOUS/'workspace/assets/meiryo.ttc',assets/'meiryo.ttc')
    catalog=json.loads((PREVIOUS/'audio_mqs_louder_01/speech_manifest_mqs_louder_01.json').read_text(encoding='utf-8'))
    for row in catalog['records']:
        if not row.get('generated'):continue
        name=Path(row['path']).name.removeprefix('mqs_louder_01_')
        shutil.copy2(PREVIOUS/'audio_mqs_louder_01'/name,assets/name)
        row['path']=RELEASE+'/workspace/assets/'+name
    save(assets/'speech_manifest.json',catalog)
    # The board has platform-specific runtime guards and metadata; capture those exact files.
    capture_adapters(package)
    inputs=package/'workspace/saved_inputs';inputs.mkdir()
    for name in ('audio.wav','video.mp4'):
        subprocess.run(['scp',*OPTIONS,f'{HOST}:/root/devai_ara/ara05_step9_device_demo_20261018_v1/workspace/saved_inputs/{name}',str(inputs/name)],check=True,timeout=30)
    config=json.loads((PREVIOUS/'demo_live_03/evidence/live_config.json').read_text(encoding='utf-8'))
    # Only deployment paths and the separately verified louder catalog change.
    def remap(value):
        if isinstance(value,dict):return {k:remap(v) for k,v in value.items()}
        if isinstance(value,list):return [remap(v) for v in value]
        if isinstance(value,str):return value.replace('/root/devai_ara/ara05_step9_device_demo_20261018_v1/workspace',RELEASE+'/workspace')
        return value
    config=remap(config)
    config['live']['speech_output']['catalog']=RELEASE+'/workspace/assets/speech_manifest.json'
    config['live']['duration_seconds']=600
    config['live']['output']='SET_BY_SUPERVISOR'
    save(package/'config.json',config)
    shutil.copy2(Path(__file__).with_name('ara_demo_runtime.py'),package/'runtime.py')
    shutil.copy2(Path(__file__).with_name('ara_audio_devices.py'),package/'audio_devices.py')
    from ara_demo_pin_dependencies import pin
    pin(package)
    dependencies=[
        '/root/devai_ara/ara06_recording_app_20261014_v1/app_venv/bin/python',
        config['observation_integration']['audio_python'],
        '/root/devai_ara/ara06_minimal_llm_20261013_v1/venv/bin/python',
        OLD+'/multimodal-emotion-analyzer/trained_models',OLD+'/devai/driver_eval/runtime',
        '/usr/share/llm/Qwen2.5-7B-Instruct','/dev/fb0',config['live']['camera']]
    from ara_audio_devices import requirements
    audio_devices=requirements(config)
    hashes={p.relative_to(package).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in package.rglob('*') if p.is_file()}
    save(package/'manifest.json',dict(release=NAME,files=hashes,dependencies=dependencies,audio_devices=audio_devices,
         original_live='demo_live_03',louder_audio='audio_mqs_louder_01',
         normal_window_seconds=8,maximum_processed_windows=1,automatic_recording=False,
         configuration_changes=['deployment_paths','verified_louder_catalog']))
    print(json.dumps(dict(built=str(package),files=len(hashes))))

def build_from_release(source_name):
    if not re.fullmatch(r'ara_demo_[0-9]{8}_v[1-9][0-9]*',source_name) or source_name==NAME:
        raise ValueError('A different pinned source release is required')
    source=ROOT/'devai/devai_ara/runs'/source_name/'release'
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256((source/n).read_bytes()).hexdigest()==h for n,h in manifest['files'].items()),'Source release integrity'
    package=RUN/'release';package.parent.mkdir(parents=True,exist_ok=True)
    if package.exists():
        current=json.loads((package/'manifest.json').read_text(encoding='utf-8'))
        if current.get('release')!=NAME or current.get('parent_release')!=source_name:
            raise ValueError('Existing fixed release does not match requested V25 lineage')
        assert all(hashlib.sha256((package/n).read_bytes()).hexdigest()==h
                   for n,h in current['files'].items()),'Existing fixed release integrity'
    else:
        shutil.copytree(source,package)
    old_prefix='/root/devai_ara/'+source_name+'/release'
    # devai_demo/device_app is the V25 source of truth.  Overlay it on the
    # verified release so dependencies and immutable assets remain pinned.
    device_app=Path(__file__).resolve().parents[1]/'device_app'
    shutil.copytree(device_app,package,dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for cache in package.rglob('__pycache__'):
        shutil.rmtree(cache)
    for bytecode in package.rglob('*.pyc'):
        bytecode.unlink()
    run_state = package/'workspace/runs'
    if run_state.exists():
        if any(run_state.iterdir()):
            raise ValueError('Local build artifact unexpectedly contains run evidence')
    else:
        run_state.mkdir()
    config=remap_release(json.loads((package/'config.json').read_text(encoding='utf-8')),old_prefix,RELEASE)
    # The centralized source may already point at another reviewed release.
    def remap_any_demo_release(value):
        if isinstance(value,dict):return {k:remap_any_demo_release(v) for k,v in value.items()}
        if isinstance(value,list):return [remap_any_demo_release(v) for v in value]
        if isinstance(value,str):
            return re.sub(r'^/root/devai_ara/ara_demo_[0-9]{8}_v[1-9][0-9]*/release',RELEASE,value)
        return value
    config=remap_any_demo_release(config)
    catalog=remap_any_demo_release(json.loads((package/'workspace/assets/speech_manifest.json').read_text(encoding='utf-8')))
    for target,value in ((package/'config.json',config),(package/'workspace/assets/speech_manifest.json',catalog)):
        target.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    shutil.copy2(device_app/'runtime.py',package/'runtime.py')
    shutil.copy2(Path(__file__).with_name('ara_audio_devices.py'),package/'audio_devices.py')
    from ara_audio_devices import requirements
    numeric={'/dev/snd/pcmC2D0p','/dev/snd/pcmC3D0c'}
    candidate={**manifest,'release':NAME,'parent_release':source_name,
        'dependencies':[p for p in manifest['dependencies'] if p not in numeric],
        'audio_devices':requirements(config),
        'configuration_changes':['v25_centralized_device_app','bounded_worker_pool',
                                 '24_hour_runtime','visual_stride_4','release_paths',
                                 'speech_priority_queue','four_second_realtime_window',
                                 'sensevoice_four_thread_live_priority','four_second_ara_budget',
                                 'strongest_audio_processed_window_retention',
                                 'remove_unused_framebuffer_backup',
                                 'sixty_second_resource_evidence_with_five_second_guard',
                                 'allow_capture_xrun_recovery',
                                 'asr_energy_gate_for_silent_live_windows',
                                 'non_blocking_ara_exact_input_cache',
                                 'visual_display_before_ara',
                                 'noise_floor_relative_asr_energy_gate',
                                 'pre_arm_noise_floor_and_window_summary_events',
                                 'visual_stride_5_and_lower_visual_priority'],
        'files':{p.relative_to(package).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in package.rglob('*') if p.is_file() and p!=package/'manifest.json'}}
    (package/'manifest.json').write_text(json.dumps(candidate,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(release=NAME,parent_release=source_name,built=str(package),files=len(candidate['files']))))


def deploy():
    package=RUN/'release';assert package.is_dir(),'Run build first'
    manifest=json.loads((package/'manifest.json').read_text(encoding='utf-8'))
    parent=manifest.get('parent_release')
    if not parent:raise ValueError('Hardlink deployment requires a parent_release')
    parent_local=ROOT/'devai/devai_ara/runs'/parent/'release'
    parent_manifest=json.loads((parent_local/'manifest.json').read_text(encoding='utf-8'))
    changed=[]
    for path in package.rglob('*'):
        if path.is_file():
            relative=path.relative_to(package).as_posix()
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if parent_manifest['files'].get(relative)!=digest:changed.append(relative)
    removed=sorted(set(parent_manifest['files'])-set(manifest['files']))
    delta_bytes=sum((package/name).stat().st_size for name in changed)
    staging=REMOTE+'.deploy_staging'
    base='/root/devai_ara/'+parent+'/release'
    ssh(f'''from pathlib import Path
import json,os,shutil
target=Path({RELEASE!r});stage=Path({staging!r});base=Path({base!r})
assert base.is_dir() and not stage.exists()
assert shutil.disk_usage('/').free>=192*2**20+{delta_bytes}+8*2**20,'Insufficient space for hardlink delta deployment'
if target.exists():
 deployed=json.loads((target/'manifest.json').read_text())
 assert deployed['release']=={NAME!r},'Existing target is not the fixed V25 release'
else:
 target.parent.mkdir(exist_ok=False)
 shutil.copytree(base,target,copy_function=os.link,symlinks=True)
''','deploy_prepare')
    with tempfile.TemporaryDirectory(dir=RUN,prefix='deploy_delta_') as temporary:
        delta=Path(temporary)
        for name in changed:
            target=delta/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(package/name,target)
        subprocess.run(['scp',*OPTIONS,'-r',str(delta),f'{HOST}:{staging}'],check=True,timeout=90)
    result=ssh(f'''from pathlib import Path
import os,runpy,json,shutil
stage=Path({staging!r});target=Path({RELEASE!r})
for source in stage.rglob('*'):
 if source.is_file():
  relative=source.relative_to(stage);destination=target/relative
  destination.parent.mkdir(parents=True,exist_ok=True)
  os.replace(source,destination)
for relative in {removed!r}:
 destination=target/relative
 if destination.is_file() or destination.is_symlink():destination.unlink()
shutil.rmtree(stage)
root=Path({RELEASE!r})/'workspace'
ns=runpy.run_path({RELEASE!r}+'/runtime.py')
result=ns['doctor']();assert result['ok'],result
result['hardlink_parent']={base!r};result['delta_files']={len(changed)};result['removed_files']={len(removed)};result['delta_bytes']={delta_bytes}
print(json.dumps(result))
''','deploy_verify')
    print(result['stdout'])

def command(action,session=None,check=False,user_ready=False):
    result=ssh(f'''import runpy,json
ns=runpy.run_path({RELEASE!r}+'/runtime.py')
print(json.dumps(ns['execute']({action!r},{session!r},{check!r},{user_ready!r}),ensure_ascii=False))
''',action)
    print(result['stdout'])
    if action=='start':
        (RUN/'active.json').write_text(json.dumps(dict(session=session,release=NAME),indent=2)+'\n',encoding='utf-8')

def collect(session):
    if not session:session=json.loads((RUN/'active.json').read_text(encoding='utf-8'))['session']
    assert session.replace('_','').isalnum()
    check=ssh(f'''import json
from pathlib import Path
p=Path({RELEASE!r})/'workspace/runs'/{session!r}
assert (p/'end_proof.json').exists(),'Wait for stopped/failed before collecting'
print((p/'end_proof.json').read_text())
''','collect_preflight')
    target=RUN/'sessions'/session;target.parent.mkdir(exist_ok=True)
    assert not target.exists(),'Evidence already collected'
    subprocess.run(['scp',*OPTIONS,'-r',f'{HOST}:{RELEASE}/workspace/runs/{session}',str(target)],check=True,timeout=120)
    print(check['stdout'])

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['build','deploy','doctor','start','status','stop','record','collect'])
    p.add_argument('--release',default=DEFAULT_RELEASE,help='Explicit pinned release; default preserves the original fixed version')
    p.add_argument('--from-release',help='For build: copy a verified fixed release without switching its app/model configuration')
    p.add_argument('--session');p.add_argument('--check',action='store_true');p.add_argument('--user-ready',action='store_true')
    a=p.parse_args()
    if a.from_release and a.action!='build':p.error('--from-release is only valid for build')
    select_release(a.release)
    if a.action=='build':build(a.from_release)
    elif a.action=='deploy':deploy()
    elif a.action=='collect':collect(a.session)
    else:command(a.action,a.session,a.check,a.user_ready)
