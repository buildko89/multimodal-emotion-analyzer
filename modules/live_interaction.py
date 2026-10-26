"""Explicit, revisioned session input. Sensor text never enters this command path."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import uuid

from modules.driver_response_policy import resolve_preferences, select_policy


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


class LiveInteraction:
    """Called while holding the application's shared-state lock."""
    def __init__(self, preferences, guidance, *, scope='actual', session_id=None):
        if scope not in {'actual', 'fixture'}:
            raise ValueError('invalid_interaction_scope')
        resolve_preferences(preferences)
        self.history = deepcopy(preferences)
        self.guidance = guidance
        self.scope = scope
        self.session_id = session_id or uuid.uuid4().hex
        self.revision = 0
        self.preference_revision = self.session_id+':0'
        self.preference_count = 0
        self.seen = set()
        self.asr = {}
        self.response = None
        self.events = []

    def snapshot(self):
        active = resolve_preferences(self.history)['active']
        names = {'minimal_guidance':'必要最小限', 'no_conversation':'会話を控える',
                 'rest_information':'休憩案内', 'repeat_guidance':'案内を繰り返す', None:'解除・待機'}
        prefix = '試験用の希望：' if self.scope == 'fixture' else '本人の明示希望：'
        value = {'interaction_revision':self.revision,'interaction_scope':self.scope,
                 'preference_revision':self.preference_revision,'preference_label':prefix+names[active],
                 'preference_history':deepcopy(self.history), **deepcopy(self.asr)}
        if self.response is not None:
            value['response'] = deepcopy(self.response)
        return value

    def observe_asr(self, text, window_id, source_end_ns):
        if not isinstance(text, str) or not isinstance(window_id, str):
            raise ValueError('invalid_asr_observation')
        self.asr = {'asr_text':text,'asr_window_id':window_id,'asr_source_end_ns':source_end_ns}
        # Receiving a different observation clears the previous window's correction.
        return {'event':'asr_text_available','window_id':window_id,'text':text,
                'speaker_verified':False,'changes_preferences':False}

    def accept_result(self, result):
        if result.get('preference_revision') != self.preference_revision:
            return False
        self.response = deepcopy(result)
        return True

    def apply(self, command, now_ns):
        ident = command.get('request_id')
        if not isinstance(ident, str) or not ident or len(ident)>80:
            raise ValueError('invalid_interaction_request_id')
        if ident in self.seen:
            raise ValueError('duplicate_interaction_request')
        channel = 'fixed_test_fixture' if self.scope == 'fixture' else 'conversation_user_reply'
        if (command.get('source') != 'explicit_user' or command.get('input_channel') != channel
                or command.get('scope') != self.scope or command.get('automatic_from_asr') is not False):
            raise ValueError('explicit_matching_scope_input_required')
        expected = command.get('expected_revision')
        if type(expected) is not int or expected != self.revision:
            raise ValueError('stale_interaction_revision')
        kind = command.get('kind')
        if kind in {'preference_set','preference_clear'}:
            event = {k:deepcopy(command[k]) for k in ('request_id','source','input_channel','scope','automatic_from_asr')}
            if kind == 'preference_set':
                event['preference'] = command.get('preference')
            else:
                if command.get('preference') is not None:
                    raise ValueError('clear_cannot_set_preference')
                event['action'] = 'clear'
            policy = select_policy({}, self.history+[event], self.guidance)
            self.history.append(event)
            self.preference_count += 1
            self.preference_revision = self.session_id+':'+str(self.preference_count)
            self.response = {'source_end_ns':now_ns,'available_ns':now_ns,'source_kind':'explicit_preference',
                'text':policy['text'],'status':'policy_output','label':'希望の設定確認（明示入力）',
                'preference_revision':self.preference_revision,'window_id':None}
        elif kind == 'asr_correction':
            if (command.get('window_id') != self.asr.get('asr_window_id')
                    or command.get('raw_text_sha256') != text_hash(self.asr.get('asr_text',''))):
                raise ValueError('asr_correction_target_mismatch')
            text = command.get('text')
            if not isinstance(text, str) or not text.strip() or len(text)>200:
                raise ValueError('invalid_explicit_asr_correction')
            self.asr['asr_correction'] = {'source':'explicit_user','text':text,
                'request_id':ident,'window_id':command['window_id'], 'changes_preferences':False}
        else:
            raise ValueError('unsupported_interaction_kind')
        self.seen.add(ident)
        self.revision += 1
        event = {'event':'explicit_interaction_applied','command':deepcopy(command),
                 'applied_ns':now_ns,'snapshot':self.snapshot()}
        self.events.append(event)
        return event


def poll_commands(state, output, now_ns, emit):
    """Immutable inbox + durable acknowledgements. Invalid commands never alter state."""
    output = Path(output)
    inbox, acks = output/'interaction_commands', output/'interaction_acks'
    inbox.mkdir(exist_ok=True)
    acks.mkdir(exist_ok=True)
    changed = False
    for path in sorted(inbox.glob('*.json')):
        ack = acks/path.name
        if ack.exists():
            continue
        try:
            command = json.loads(path.read_text(encoding='utf-8'))
            event = state.apply(command, now_ns)
            result = {'accepted':True,'event':event}
            emit(event)
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            result = {'accepted':False,'error':str(error),'revision':state.revision}
            emit({'event':'explicit_interaction_rejected','file':path.name, **result})
        atomic_json(ack, result)
        changed = True
    if changed or not (output/'interaction_state.json').exists():
        atomic_json(output/'interaction_state.json', state.snapshot())
