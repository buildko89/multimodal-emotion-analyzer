"""Sample Linux process RSS and system headroom without interrupting capture."""
import os
from pathlib import Path
import time


def memory_snapshot(workspace, proc_root=Path('/proc')):
    if os.name=='nt':
        import psutil
        parent=psutil.Process()
        selected=[]
        for process in [parent,*parent.children(recursive=True)]:
            try:
                memory=process.memory_info()
                selected.append({'pid':process.pid,'ppid':process.ppid(),'argv':process.cmdline(),
                    'rss_bytes':memory.rss,'hwm_bytes':getattr(memory,'peak_wset',memory.rss),'role':'live_owned'})
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        return {'sampled_ns':time.monotonic_ns(),'processes':selected,
                'selected_rss_sum_bytes':sum(p['rss_bytes'] for p in selected),
                'sum_is_not_unique_physical_memory':True,'mem_available_bytes':psutil.virtual_memory().available,
                'swap_used_bytes':psutil.swap_memory().used}
    processes = []
    for directory in proc_root.glob('[0-9]*'):
        try:
            argv = (directory/'cmdline').read_bytes().decode(errors='replace').split('\0')
            status = (directory/'status').read_text()
            fields = {line.split(':', 1)[0]: line.split(':', 1)[1].strip() for line in status.splitlines() if ':' in line}
            parent = int(fields['PPid'])
            processes.append({'pid': int(directory.name), 'ppid': parent, 'argv': argv,
                'rss_bytes': int(fields.get('VmRSS', '0 kB').split()[0])*1024,
                'hwm_bytes': int(fields.get('VmHWM', '0 kB').split()[0])*1024})
        except (OSError, ValueError, KeyError):
            continue
    owned = {os.getpid()}
    while True:
        children = {p['pid'] for p in processes if p['ppid'] in owned}
        if children <= owned: break
        owned |= children
    selected = []
    for process in processes:
        command = ' '.join(process['argv'])
        if process['pid'] in owned or str(workspace) in command:
            process['role'] = 'live_owned'
        elif 'modules.ara_llm_service' in process['argv']:
            process['role'] = 'ara_service'
        else:
            continue
        selected.append(process)
    mem = {line.split(':')[0]: int(line.split(':')[1].strip().split()[0])*1024
           for line in (proc_root/'meminfo').read_text().splitlines()}
    return {'sampled_ns': time.monotonic_ns(), 'processes': selected,
            'selected_rss_sum_bytes': sum(p['rss_bytes'] for p in selected),
            'sum_is_not_unique_physical_memory': True,
            'mem_available_bytes': mem['MemAvailable'], 'swap_used_bytes': mem['SwapTotal']-mem['SwapFree']}
