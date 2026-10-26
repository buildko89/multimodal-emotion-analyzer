"""Board-side lifecycle supervisor for the frozen ARA demo release."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
import urllib.request
import runpy

RELEASE=Path(__file__).resolve().parent
STATE=RELEASE/'workspace/runs'
APP='/root/devai_ara/ara06_recording_app_20261014_v1/app_venv/bin/python'
LLM='/root/devai_ara/ara06_minimal_llm_20261013_v1/venv/bin/python'
UNITS=['weston.socket','weston.service']

def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def read(path):return json.loads(path.read_text())

def identity(pid):
    try:
        p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(') ',1)[1].split()
        if s[0]=='Z':return None
        return dict(pid=pid,start_ticks=s[19],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    argv=(p/'cmdline').read_bytes().decode().strip('\0').split('\0'))
    except FileNotFoundError:return None

def matches(owned):return identity(owned['pid'])==owned

def health():
    try:
        with urllib.request.urlopen('http://127.0.0.1:18081/health',timeout=2) as r:return json.load(r)
    except Exception:return {'state':'unreachable'}

def desktop():
    return {u:subprocess.run(['systemctl','is-active',u],capture_output=True,text=True).stdout.strip() for u in UNITS}

def holders():
    rows=[]
    for fd in Path('/proc').glob('[0-9]*/fd/*'):
        try:
            target=os.readlink(fd)
            if target.startswith(('/dev/video','/dev/snd/pcm')):rows.append(dict(pid=int(fd.parent.parent.name),path=target))
        except OSError:pass
    return rows

def doctor(full=False):
    manifest=read(RELEASE/'manifest.json')
    bad=[name for name,digest in manifest['files'].items()
         if not (RELEASE/name).is_file() or hashlib.sha256((RELEASE/name).read_bytes()).hexdigest()!=digest]
    missing=[p for p in manifest['dependencies'] if not Path(p).exists()]
    audio=None
    if 'audio_devices' in manifest:
        helper=RELEASE/'audio_devices.py'
        # Run only the helper pinned in this release, after its integrity check.
        if not helper.is_file() or 'audio_devices.py' not in manifest['files'] or 'audio_devices.py' in bad:
            if 'audio_devices.py' not in bad:bad.append('audio_devices.py')
            audio=dict(ok=False,resolved=[],errors=[dict(reason='audio_helper_integrity')])
        else:
            api=runpy.run_path(str(helper))
            try:
                if api['requirements'](read(RELEASE/'config.json'))!=manifest['audio_devices']:
                    raise ValueError('Audio requirements do not match pinned configuration')
                audio=api['check'](manifest['audio_devices'])
            except (OSError,ValueError,KeyError,TypeError) as error:
                audio=dict(ok=False,resolved=[],errors=[dict(reason='audio_configuration',detail=str(error))])
    changed=[]
    pins=read(RELEASE/'dependency_manifest.json')
    for name,record in pins['files'].items():
        path=Path(name)
        if not path.is_file():changed.append(name);continue
        stat=path.stat()
        if str(path.resolve())!=record['resolved'] or stat.st_size!=record['bytes'] or stat.st_mtime_ns!=record['mtime_ns']:
            changed.append(name);continue
        if full:
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(8*1024**2),b''):digest.update(chunk)
            if digest.hexdigest()!=record['sha256']:changed.append(name)
    free=shutil.disk_usage('/').free
    return dict(ok=not bad and not missing and not changed and (audio is None or audio['ok']) and free>=192*1024**2,release=manifest['release'],
                audio_devices=audio,
                bad_hashes=bad,missing_dependencies=missing,changed_dependencies=changed,
                dependency_files=len(pins['files']),dependency_check='sha256' if full else 'resolved_path_size_mtime',
                root_free_bytes=free,minimum_free_bytes=192*1024**2,
                health=health(),desktop=desktop(),device_holders=holders())

def spawn(argv,folder,name,env,cwd):
    with (folder/(name+'.log')).open('xb') as log:
        p=subprocess.Popen(argv,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    for _ in range(100):
        owned=identity(p.pid)
        if owned and owned['argv']==argv:break
        if p.poll() is not None:raise RuntimeError(name+' exited during startup')
        time.sleep(.01)
    else:raise RuntimeError(name+' identity unavailable')
    write(folder/(name+'_identity.json'),owned)
    return p,owned

def wait_exit(proc,owned,seconds):
    limit=time.monotonic()+seconds
    while proc.poll() is None and time.monotonic()<limit:time.sleep(.2)
    if proc.poll() is None:raise RuntimeError('Owned process still busy; no forced kill: '+str(owned['pid']))

def resident_active():
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            if b'modules.ara_llm_service' in p.read_bytes():return True
        except (FileNotFoundError, ProcessLookupError, PermissionError):pass
    return False

def supervise(folder,check):
    # One lock covers all releases, held through cleanup. No unrelated PID is signalled.
    inherited=os.environ.pop('ARA_RUNTIME_LOCK_FD',None)
    lock=os.fdopen(int(inherited),'a') if inherited else (RELEASE.parent/'runtime.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    service=live=None;service_owned=live_owned=None;lease=None;failure=None;ready=False;first_ready_ns=None
    guard=None;live_deadline=None;overlapped=False
    signal.signal(signal.SIGTERM,lambda *_:(folder/'stop.request').touch())
    signal.signal(signal.SIGINT,lambda *_:(folder/'stop.request').touch())
    write(folder/'supervisor_identity.json',identity(os.getpid()))
    def state(stage,**extra):write(folder/'state.json',dict(stage=stage,check_only=check,**extra))
    try:
        probe=doctor();write(folder/'doctor.json',probe)
        assert probe['ok'],probe
        resident_ara=(probe['health']['state']=='ready')
        if not resident_ara:
            assert probe['health']['state']=='unreachable','ARA port already in use'
        assert not probe['device_holders'],'Camera or audio device already held'
        for path in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                argv=path.read_bytes().split(b'\0')
                assert b'modules.live_pipeline' not in argv,'Other live pipeline running'
                if not resident_ara:
                    assert b'modules.ara_llm_service' not in argv,'Other ARA service running'
            except FileNotFoundError:pass
        env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',
                 OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',TF_CPP_MIN_LOG_LEVEL='1')
        env.pop('DV_TGT_ROOT',None)
        env['PYTHONPATH']=str(RELEASE/'workspace/multimodal-emotion-analyzer')
        if resident_ara:
            state('ara_ready',resident=True)
            service=None;service_owned=None
        else:
            state('ara_loading')
            service,service_owned=spawn([LLM,'-B','-u','-m','modules.ara_llm_service','--port','18081',
                '--audit',str(folder/'ara_audit.jsonl'),'--prompt-format','compact-json-v1'],folder,'ara',env,RELEASE)
        config=read(RELEASE/'config.json')
        overlap=config.get('startup_overlap',{}).get('enabled',False)
        if overlap:
            guard=runpy.run_path(str(RELEASE/'startup_guard.py'))['StartupGuard'](folder)
        def launch_pipeline():
            nonlocal lease,live,live_owned,live_deadline
            config['live']['output']=str(folder/'live')
            config['live']['sensor_capture']=not check
            write(folder/'config.json',config)
            lease=desktop();write(folder/'desktop_lease.json',lease)
            subprocess.run(['systemctl','stop',*UNITS],check=True,timeout=30)
            live,live_owned=spawn([APP,'-B','-u','-m','modules.live_pipeline','--config',str(folder/'config.json')],
                                 folder,'pipeline',env,RELEASE/'workspace/multimodal-emotion-analyzer')
            live_deadline=time.monotonic()+480
        if resident_ara:
            launch_pipeline();overlapped=True
            write(folder/'overlap_launch.json',dict(launched_ns=time.monotonic_ns(),ara_health=health(),resident_ara=True))
            state('preparing_models')
        else:
            deadline=time.monotonic()+240
            while health().get('state')!='ready':
                if (folder/'stop.request').exists():return
                if guard:guard.sample('ara_loading')
                if overlap and live is None and health().get('state')=='loading' and time.monotonic_ns()-guard.baseline['sampled_ns']>=config['startup_overlap'].get('minimum_delay_seconds',0)*10**9:
                    guard.sample('overlap_admission',admission=True)
                    launch_pipeline();overlapped=True
                    write(folder/'overlap_launch.json',dict(launched_ns=time.monotonic_ns(),ara_health=health(),resource_gate_passed=True))
                    state('overlap_preparing_models')
                if live is not None:assert live.poll() is None,'Pipeline exited during ARA load'
                assert service.poll() is None,'ARA exited during load'
                assert health().get('state')!='failed','ARA load failed'
                assert time.monotonic()<deadline,'ARA load deadline exceeded'
                time.sleep(.5)
            if (folder/'stop.request').exists():return
            if live is None:launch_pipeline()
            state('preparing_models')
        deadline=live_deadline
        while live.poll() is None:
            if (folder/'stop.request').exists():break
            if service is not None:assert service.poll() is None,'ARA exited during preparation'
            else:assert resident_active(),'Resident ARA service process died'
            if guard and not ready:guard.sample('preparing_models')
            prep=folder/'live/preparation_state.json'
            if prep.exists():
                current=read(prep)
                display_state=folder/'live/status.json'
                if current.get('ready') and not ready and display_state.exists() and read(display_state).get('preparation_state')=='ready':
                    ready=True
                    for line in (folder/'live/live_events.jsonl').open():
                        event=json.loads(line)
                        if event.get('event')=='display_readback' and event.get('preparation_state')=='ready':
                            first_ready_ns=event['displayed_ns'];break
                    state('ready',preparation_seconds=current.get('elapsed_seconds'),recording_requested=False)
                    if check:break
                if current.get('state')=='failed':raise RuntimeError('Model preparation failed')
            if not ready:assert time.monotonic()<deadline,'Preparation deadline exceeded'
            time.sleep(.3)
        if not ready and not (folder/'stop.request').exists():raise RuntimeError('Pipeline exited before ready')
    except Exception as error:
        failure=repr(error)
    finally:
        state('stopping',error=failure)
        if guard:write(folder/'startup_guard_summary.json',guard.summary())
        write(folder/'startup_overlap_result.json',dict(overlapped=overlapped,guard_enabled=guard is not None,resident_ara=resident_ara if 'resident_ara' in locals() else False))
        if 'resident_ara' in locals() and resident_ara:
            audit_src=Path('/root/devai_ara/services/ara_llm/ara_audit.jsonl')
            if audit_src.exists():shutil.copyfile(audit_src,folder/'ara_audit.jsonl')
            else:(folder/'ara_audit.jsonl').touch()
        cleanup_errors=[]
        if live is not None and live.poll() is None:
            try:
                assert matches(live_owned),'Pipeline identity changed'
                os.kill(live.pid,signal.SIGINT);wait_exit(live,live_owned,480)
            except Exception as error:cleanup_errors.append(repr(error))
        if service is not None and service.poll() is None:
            try:
                assert matches(service_owned),'ARA identity changed'
                deadline=time.monotonic()+480
                while health().get('inference_active') is not False or health().get('state') not in ('ready','failed'):
                    if service.poll() is not None:break
                    assert time.monotonic()<deadline,'ARA remains busy; no forced kill'
                    time.sleep(.5)
                if service.poll() is None:os.kill(service.pid,signal.SIGINT)
                wait_exit(service,service_owned,20)
            except Exception as error:cleanup_errors.append(repr(error))
        if lease is not None and (live is None or live.poll() is not None):
            try:
                for unit,prior in lease.items():
                    if prior=='active':subprocess.run(['systemctl','start',unit],check=True,timeout=30)
                assert desktop()==lease,'Desktop restoration mismatch'
            except Exception as error:cleanup_errors.append(repr(error))
        final=folder/'live/final_report.json'
        report=read(final) if final.exists() else None
        remaining=holders()
        runtime=folder/'live/window_runtime_final.json'
        proof=dict(ready_reached=ready,error=failure,cleanup_errors=cleanup_errors,
                   all_owned_processes_exited=all(p is None or p.poll() is not None for p in (live,service)),
                   desktop_restored=lease is None or desktop()==lease,device_holders=remaining,
                   new_recording_windows=len(list((folder/'live/windows').glob('*/capture_manifest.json'))),
                   threads_stopped=report.get('threads_stopped') if report else None,
                   workers_exited=read(runtime).get('all_owned_workers_exited') if runtime.exists() else None)
        request=folder/'start_request.json'
        if first_ready_ns is not None and request.exists():
            proof['startup_to_ready_hdmi_seconds']=(first_ready_ns-read(request)['requested_ns'])/1e9
            proof['startup_240s_target_single_run_met']=proof['startup_to_ready_hdmi_seconds']<=240
        write(folder/'end_proof.json',proof)
        clean=not cleanup_errors and proof['all_owned_processes_exited'] and proof['desktop_restored'] and not remaining
        state('stopped' if clean and not failure else 'failed',ready_reached=ready,error=failure,cleanup_ok=clean)

def execute(action,session=None,check=False,user_ready=False):
    STATE.mkdir(exist_ok=True)
    if action=='doctor':return doctor(full=True)
    if action=='start':
        assert session and session.replace('_','').isalnum(),'Safe unique session required'
        probe=doctor();assert probe['ok'],probe
        # Check the supervisor lock before creating a fresh session.
        with (RELEASE.parent/'runtime.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            folder=STATE/session;folder.mkdir(exist_ok=False)
            write(folder/'start_request.json',dict(requested_ns=time.monotonic_ns(),check_only=check))
            with (folder/'supervisor.log').open('xb') as log:
                proc=subprocess.Popen(['python3','-B',str(RELEASE/'runtime.py'),'supervise','--session',session,
                     *(['--check'] if check else [])],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,
                     pass_fds=(lock.fileno(),),env=dict(os.environ,ARA_RUNTIME_LOCK_FD=str(lock.fileno())))
        write(STATE/'active.json',dict(session=session,release=str(RELEASE)))
        return dict(session=session,supervisor_pid=proc.pid,check_only=check,recording_requested=False)
    active=read(STATE/'active.json')
    assert active['release']==str(RELEASE),'Active release mismatch'
    if session:assert session==active['session'],'Active session mismatch'
    folder=STATE/active['session']
    if action=='supervise':return supervise(folder,check)
    state=read(folder/'state.json') if (folder/'state.json').exists() else {'stage':'starting'}
    if action=='stop':
        if state['stage'] not in ('stopped','failed'):
            owned=read(folder/'supervisor_identity.json');assert matches(owned),'Supervisor identity mismatch'
            (folder/'stop.request').touch()
        return dict(session=active['session'],stop_requested=True,state=state)
    if action=='record':
        assert user_ready,'--user-ready must follow a current explicit operator confirmation'
        assert not state['check_only'] and state['stage']=='ready','Live session must be ready'
        owned=read(folder/'pipeline_identity.json');assert matches(owned),'Pipeline identity mismatch'
        prep=read(folder/'live/preparation_state.json');assert prep.get('ready') and prep['state']=='ready'
        assert not list((folder/'live/windows').glob('*/capture_manifest.json')),'One window already recorded'
        command=dict(request_id='demo_utterance_01',not_before_ns=time.monotonic_ns(),
                     prompt='今日はいい天気ですね',operator_contract='only_after_explicit_user_ready_reply')
        receipt=folder/'recording_operator_ready.json'
        with receipt.open('x') as stream:json.dump(dict(session=active['session'],explicit_user_ready=True,command=command),stream)
        write(folder/'live/capture_command.json',command)
        return dict(recording_seconds=8,command=command)
    if action=='arm':
        assert user_ready,'--user-ready must follow a current explicit post-ready confirmation'
        assert not state['check_only'] and state['stage']=='ready','Live session must be ready'
        owned=read(folder/'pipeline_identity.json');assert matches(owned),'Pipeline identity mismatch'
        prep=read(folder/'live/preparation_state.json');assert prep.get('ready') and prep['state']=='ready'
        command_path=folder/'live/continuous_start_command.json'
        assert not command_path.exists(),'Continuous capture already armed'
        command=dict(request_id='continuous_post_ready_01',authorized_ns=time.monotonic_ns(),
                     explicit_user_ready=True,
                     operator_contract='current_confirmation_after_device_ready')
        write(folder/'post_ready_operator_confirmation.json',command)
        write(command_path,command)
        write(folder/'state.json',{**state,'recording_requested':True,'post_ready_capture_armed':True})
        return dict(session=active['session'],continuous_capture_armed=True,command=command)
    return dict(session=active['session'],state=state,health=health(),
                preparation=read(folder/'live/preparation_state.json') if (folder/'live/preparation_state.json').exists() else None,
                end=read(folder/'end_proof.json') if (folder/'end_proof.json').exists() else None)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['doctor','start','status','stop','record','arm','supervise'])
    p.add_argument('--session');p.add_argument('--check',action='store_true');p.add_argument('--user-ready',action='store_true')
    a=p.parse_args()
    if a.action=='supervise':supervise(STATE/a.session,a.check)
    else:print(json.dumps(execute(a.action,a.session,a.check,a.user_ready),ensure_ascii=False))
