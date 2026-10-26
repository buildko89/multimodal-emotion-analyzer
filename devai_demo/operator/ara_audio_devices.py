"""Resolve named ALSA hardware requirements without opening any PCM device."""
from pathlib import Path
import re
import stat

SCHEMA='ara-named-alsa-pcm-v1'


def requirements(config):
    live=config['live']
    values=[('speech_output',live['speech_output']['device'],'playback'),
            ('microphone',live['microphone'],'capture')]
    result=[]
    for role,selection,direction in values:
        match=re.fullmatch(r'hw:CARD=([A-Za-z0-9_]+),DEV=([0-9]+)',selection)
        if not match:raise ValueError('Named ALSA hw:CARD=...,DEV=... required: '+role)
        result.append(dict(role=role,alsa=selection,card_id=match[1],device=int(match[2]),direction=direction))
    return dict(schema=SCHEMA,requirements=result)


def check(spec,proc_asound=Path('/proc/asound'),dev_snd=Path('/dev/snd'),is_character=None):
    proc_asound,dev_snd=Path(proc_asound),Path(dev_snd)
    if spec.get('schema')!=SCHEMA:raise ValueError('Unsupported ALSA requirement schema')
    cards={};resolved=[];errors=[]
    try:
        pcm=(proc_asound/'pcm').read_text()
        for p in proc_asound.glob('card[0-9]*/id'):
            number=int(p.parent.name[4:]);name=p.read_text().strip()
            cards.setdefault(name,[]).append(number)
    except (OSError,ValueError) as error:
        return dict(ok=False,resolved=[],errors=[dict(reason='alsa_inventory_unavailable',detail=str(error))])
    endpoints={}
    for line in pcm.splitlines():
        match=re.match(r'^\s*([0-9]+)-([0-9]+):\s*(.*)$',line)
        if match:endpoints[(int(match[1]),int(match[2]))]=match[3]
    if is_character is None:
        def is_character(p):return stat.S_ISCHR(p.stat().st_mode)
    for item in spec['requirements']:
        role=item['role'];numbers=cards.get(item['card_id'],[])
        if len(numbers)!=1:
            errors.append(dict(role=role,reason='card_missing' if not numbers else 'card_ambiguous',card_id=item['card_id']))
            continue
        number=numbers[0];device=item['device'];direction=item['direction']
        if direction not in ('playback','capture') or isinstance(device,bool) or not isinstance(device,int) or device<0:
            errors.append(dict(role=role,reason='invalid_endpoint_requirement'));continue
        advertised=endpoints.get((number,device),'')
        if not re.search(r'\b'+direction+r'\s+[1-9][0-9]*\b',advertised):
            errors.append(dict(role=role,reason='pcm_direction_missing',card_id=item['card_id'],device=device,direction=direction));continue
        path=dev_snd/f"pcmC{number}D{device}{'p' if direction=='playback' else 'c'}"
        try:
            if not is_character(path):raise ValueError('PCM node is not a character device')
            if (proc_asound/f'card{number}'/'id').read_text().strip()!=item['card_id']:
                raise ValueError('ALSA card identity changed during check')
        except (OSError,ValueError) as error:
            errors.append(dict(role=role,reason='pcm_node_unavailable',path=str(path),detail=str(error)));continue
        resolved.append(dict(**item,card_number=number,path=str(path)))
    return dict(ok=not errors and len(resolved)==len(spec['requirements']),resolved=resolved,errors=errors)
