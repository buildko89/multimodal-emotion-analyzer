"""Application-scoped registration for the pinned two audio models; no venv edits.

The installed initializer is hash checked and executed with its single eager
package scan suppressed. Model implementation files and normal submodule import
loaders remain unchanged. This is deliberately limited to the pinned demo.
"""
import hashlib
import importlib
import importlib.metadata
import importlib.util
from pathlib import Path
import sys
import time

PINNED_INIT_SHA256='2e14bea49617caf22b22db64c51da367ee15f686d23cade0c905b1bf5c43d64c'
MODULES={
    'sensevoice':('funasr.models.sense_voice.model','funasr.frontends.wav_frontend',
                  'funasr.tokenizer.sentencepiece_tokenizer','funasr.models.specaug.specaug'),
    'emotion2vec':('funasr.models.emotion2vec.model','funasr.tokenizer.char_tokenizer'),
}
REQUIRED={
    'sensevoice':{'model_classes':('SenseVoiceSmall',),'encoder_classes':('SenseVoiceEncoderSmall',),
                  'frontend_classes':('WavFrontend',),'tokenizer_classes':('SentencepiecesTokenizer',),
                  'specaug_classes':('SpecAugLFR',)},
    'emotion2vec':{'model_classes':('Emotion2vec',),'tokenizer_classes':('CharTokenizer',)},
}


def prepare_funasr(backend):
    if backend not in MODULES:
        raise ValueError('Unsupported scoped audio backend')
    started=time.monotonic_ns()
    package=sys.modules.get('funasr')
    if package is not None and not hasattr(package,'_ara_scoped_policy'):
        raise RuntimeError('FunASR already initialized outside pinned scoped policy')
    if package is None:
        if importlib.metadata.version('funasr')!='1.4.16':
            raise RuntimeError('FunASR version differs from scoped policy pin')
        spec=importlib.util.find_spec('funasr')
        if spec is None or spec.origin is None or spec.submodule_search_locations is None:
            raise RuntimeError('Pinned FunASR package not found')
        source=Path(spec.origin).read_bytes()
        actual=hashlib.sha256(source).hexdigest()
        if actual!=PINNED_INIT_SHA256:
            raise RuntimeError('FunASR initializer differs from scoped policy pin')
        anchor=b'\nimport_submodules(__name__)\n'
        if source.count(anchor)!=1:
            raise RuntimeError('Expected one eager registration call')
        effective=source.replace(anchor,b'\n# Eager package scan replaced by the pinned demo registration list.\n')
        package=importlib.util.module_from_spec(spec)
        sys.modules['funasr']=package
        try:
            exec(compile(effective,spec.origin,'exec'),package.__dict__)
            package._ara_scoped_policy=dict(mode='pinned_demo_scoped_registration_v1',
                original_init_sha256=actual,effective_init_sha256=hashlib.sha256(effective).hexdigest(),
                installed_files_modified=False,backends=[],initialization_seconds={})
        except BaseException:
            sys.modules.pop('funasr',None)
            raise
    policy=package._ara_scoped_policy
    cached=backend in policy['backends']
    if not cached:
        for name in MODULES[backend]:importlib.import_module(name)
        # Load the normal exported constructor and its dependencies, not other models.
        if not callable(package.AutoModel):raise RuntimeError('AutoModel export unavailable')
        tables=importlib.import_module('funasr.register').tables
        for table,keys in REQUIRED[backend].items():
            if not all(key in getattr(tables,table) for key in keys):
                raise RuntimeError('Required pinned model registry entries missing')
        policy['backends'].append(backend)
        policy['initialization_seconds'][backend]=(time.monotonic_ns()-started)/1e9
    return dict(mode=policy['mode'],backend=backend,cached=cached,
        initialization_seconds=policy['initialization_seconds'][backend],
        original_init_sha256=policy['original_init_sha256'],effective_init_sha256=policy['effective_init_sha256'],
        modules=list(MODULES[backend]),installed_files_modified=False,
        scope='Pinned SenseVoiceSmall and Emotion2vec only; no generic FunASR model support claim')
