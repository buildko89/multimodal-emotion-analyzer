"""Board-only resource admission/monitoring for overlapped startup."""
import json
from pathlib import Path
import shutil
import time

LIMITS={'ana-thermal':90,'a55-thermal':90,'pf09':130,'pf53_soc':130,'pf53_arm':130}

def snapshot():
    mem={l.split(':')[0]:int(l.split(':')[1].split()[0])*1024 for l in Path('/proc/meminfo').read_text().splitlines()}
    thermal={(p/'type').read_text().strip():int((p/'temp').read_text())/1000 for p in Path('/sys/class/thermal').glob('thermal_zone*')}
    return dict(sampled_ns=time.monotonic_ns(),available_bytes=mem['MemAvailable'],swap_used_bytes=mem['SwapTotal']-mem['SwapFree'],free_bytes=shutil.disk_usage('/').free,thermal=thermal)

def violations(row,baseline_swap,*,admission=False):
    errors=[]
    if row['available_bytes']<(2048 if admission else 400)*2**20:errors.append('mem_available')
    if row['swap_used_bytes']-baseline_swap>512*2**20:errors.append('swap_growth')
    if row['free_bytes']<150*2**20:errors.append('disk_free')
    for name,maximum in LIMITS.items():
        if name not in row['thermal']:errors.append('missing_thermal:'+name)
        elif not -40<=row['thermal'][name]<maximum:errors.append('thermal:'+name)
    return errors

class StartupGuard:
    def __init__(self,folder):
        self.folder=Path(folder);self.baseline=snapshot();self.rows=[]
        self.sample('admission',admission=True)

    def sample(self,phase,*,admission=False):
        row=snapshot();row['phase']=phase
        row['violations']=violations(row,self.baseline['swap_used_bytes'],admission=admission)
        self.rows.append(row)
        with (self.folder/'startup_resources.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        if row['violations']:raise RuntimeError('startup_resource_guard:'+','.join(row['violations']))
        return row

    def summary(self):
        return dict(baseline=self.baseline,samples=len(self.rows),
            minimum_available_bytes=min(r['available_bytes'] for r in self.rows),
            maximum_swap_growth_bytes=max(r['swap_used_bytes']-self.baseline['swap_used_bytes'] for r in self.rows),
            minimum_free_bytes=min(r['free_bytes'] for r in self.rows),
            all_samples_passed=all(not r['violations'] for r in self.rows),
            scope='Supervisor startup only; 0.3-0.5s nominal samples, no continuous or post-ready guarantee')
