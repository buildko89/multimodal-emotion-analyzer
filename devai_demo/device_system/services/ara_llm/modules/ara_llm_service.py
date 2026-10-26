"""Loopback-only ARA structured generation bridge for LocalLLMClient."""
import argparse
from copy import deepcopy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import select
import socket
import threading
import time
import uuid

from modules.ara_json_logits import AraJSONLogitsProcessor, GenerationDeadline

MODEL_ID = 'ara-qwen2.5-7b-r2.0.4'
SCHEMA_KEYS = {'type', 'properties', 'required', 'additionalProperties', 'items', 'minItems',
               'maxItems', 'minLength', 'maxLength', 'minimum', 'maximum', 'enum', 'const',
               'anyOf', 'description', 'title', '$schema'}


def strict_json(text):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('duplicate_json_key')
            value[key] = item
        return value
    def nonfinite(value):
        raise ValueError('nonfinite_json')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)


def check_schema_subset(schema, depth=0):
    if not isinstance(schema, dict) or depth > 20:
        raise ValueError('schema_structure_or_depth')
    if set(schema) - SCHEMA_KEYS:
        raise ValueError('unsupported_schema_keywords')
    properties = schema.get('properties', {})
    alternatives = schema.get('anyOf', [])
    if not isinstance(properties, dict) or not isinstance(alternatives, list):
        raise ValueError('schema_structure')
    for child in properties.values():
        check_schema_subset(child, depth + 1)
    if 'items' in schema:
        check_schema_subset(schema['items'], depth + 1)
    for child in alternatives:
        check_schema_subset(child, depth + 1)


def validate_request(value, model_id):
    if not isinstance(value, dict) or value.get('model') != model_id:
        raise ValueError('unknown_model')
    if value.get('stream', False) or value.get('temperature', 0) != 0 or value.get('cache_prompt', False):
        raise ValueError('only_nonstreaming_greedy_uncached_supported')
    maximum = value.get('max_tokens', 512)
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 512:
        raise ValueError('max_tokens_range')
    messages = value.get('messages')
    if not isinstance(messages, list) or not 1 <= len(messages) <= 16:
        raise ValueError('messages_required')
    for message in messages:
        if not isinstance(message, dict) or message.get('role') not in {'system', 'user', 'assistant'} or not isinstance(message.get('content'), str):
            raise ValueError('message_structure')
    response_format = value.get('response_format', {})
    if not isinstance(response_format, dict):
        raise ValueError('response_format_structure')
    wrapped = response_format.get('json_schema', {})
    if not isinstance(wrapped, dict):
        raise ValueError('json_schema_structure')
    if response_format.get('type') != 'json_schema' or wrapped.get('strict') is not True:
        raise ValueError('strict_json_schema_required')
    schema = wrapped.get('schema')
    check_schema_subset(schema)
    return schema, maximum


class StructuredProcessorMixin:
    """BSP prompt forward replaces custom processors; restore before selection."""
    def _generate_first_token(self, *args, **kwargs):
        result = super()._generate_first_token(*args, **kwargs)
        self.logits_processor = self.structured_processors
        return result

    def _generate_next_token(self, *args, **kwargs):
        started = time.monotonic()
        try:
            return super()._generate_next_token(*args, **kwargs)
        finally:
            self.native_decode_seconds = getattr(self, 'native_decode_seconds', 0)+time.monotonic()-started
            self.native_decode_calls = getattr(self, 'native_decode_calls', 0)+1


def build_messages(messages, schema, prompt_format='legacy-v1'):
    """Compact only JSON serialization whitespace; retain keys, values and order."""
    messages = deepcopy(messages)
    options = {'ensure_ascii': False}
    if prompt_format == 'compact-json-v1':
        options['separators'] = (',', ':')
        for message in messages:
            if message['role'] == 'user':
                try:
                    value = strict_json(message['content'])
                except (ValueError, TypeError):
                    continue
                if isinstance(value, (dict, list)):
                    message['content'] = json.dumps(value, **options)
    elif prompt_format != 'legacy-v1':
        raise ValueError('unknown_prompt_format')
    instruction = ('Return exactly one JSON value matching the following schema. '
                   'No Markdown fences or explanation. Schema: ' + json.dumps(schema, **options))
    if messages[0]['role'] == 'system':
        messages[0]['content'] += '\n' + instruction
    else:
        messages.insert(0, {'role': 'system', 'content': instruction})
    return messages


def read_meminfo():
    """Extract key memory and swap metrics from /proc/meminfo."""
    metrics = {}
    try:
        with open('/proc/meminfo', 'r', encoding='utf-8') as stream:
            for line in stream:
                parts = line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip().split()[0]
                    if key in {'MemTotal', 'MemFree', 'MemAvailable', 'Buffers', 'Cached', 'SwapTotal', 'SwapFree'}:
                        metrics[key] = int(val)
    except OSError:
        pass
    return metrics


class CacheEvictor:
    """Evicts Linux page cache for large model files during and after load to prevent host RAM saturation."""
    def __init__(self, file_path, interval=0.5):
        self.file_path = Path(file_path)
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None
        self.evict_calls = 0

    def _worker(self):
        try:
            with open(self.file_path, 'rb') as stream:
                fd = stream.fileno()
                while not self.stop_event.is_set():
                    try:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                        self.evict_calls += 1
                    except (OSError, AttributeError):
                        pass
                    self.stop_event.wait(self.interval)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    self.evict_calls += 1
                except (OSError, AttributeError):
                    pass
        except (OSError, AttributeError):
            pass

    def start(self):
        if self.file_path.is_file() and hasattr(os, 'posix_fadvise'):
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        try:
            if self.file_path.is_file() and hasattr(os, 'posix_fadvise'):
                with open(self.file_path, 'rb') as stream:
                    os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                    self.evict_calls += 1
        except (OSError, AttributeError):
            pass


class NativeARABackend:
    def __init__(self, model_path, prompt_format='legacy-v1'):
        import xgrammar as xgr
        from transformers import AutoTokenizer
        from optimum.ara import AraQwenForCausalLM
        class StructuredAraQwenForCausalLM(StructuredProcessorMixin, AraQwenForCausalLM):
            pass
        started = time.monotonic()
        mem_before = read_meminfo()
        self.tokenizer = AutoTokenizer.from_pretrained(str(Path(model_path) / 'tokenizer'), local_files_only=True)
        
        # Approach 2: Evict page cache during and after DVM read to avoid swapping
        model_dvm = Path(model_path) / 'model.dvm'
        evictor = CacheEvictor(model_dvm, interval=0.5)
        evictor.start()
        try:
            self.model = StructuredAraQwenForCausalLM.from_pretrained(str(model_path))
        finally:
            evictor.stop()
            
        self.vocab_size = self.model.core.get_vocab_size()
        stop_ids = set(self.model.generation_config.eos_token_id)
        stop_ids.add(self.model.core.get_eos_token_id())
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(self.tokenizer,
            vocab_size=self.vocab_size, stop_token_ids=sorted(stop_ids))
        self.compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=2, cache_limit_bytes=32 * 1024**2)
        self.load_seconds = time.monotonic() - started
        mem_after = read_meminfo()
        self.load_metrics = {
            'load_seconds': self.load_seconds,
            'evict_calls': evictor.evict_calls,
            'mem_before_kb': mem_before,
            'mem_after_kb': mem_after,
        }
        self.model_path = str(model_path)
        self.prompt_format = prompt_format

    def generate(self, value, schema, maximum, deadline, cancelled):
        import jsonschema
        import torch
        from transformers import LogitsProcessorList
        jsonschema.Draft202012Validator.check_schema(schema)
        started = time.monotonic()
        compiled = self.compiler.compile_json_schema(schema, indent=None,
            separators=(',', ':'), any_whitespace=False)
        compile_seconds = time.monotonic() - started
        messages = build_messages(value['messages'], schema, self.prompt_format)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids'][0].tolist()
        if len(input_ids) + maximum >= self.model.core.get_max_num_tokens():
            raise ValueError('context_limit')
        if time.monotonic() >= deadline or cancelled():
            raise GenerationDeadline('deadline_before_native_generation')
        processor = AraJSONLogitsProcessor(compiled, self.vocab_size,
            deadline=deadline, max_tokens=maximum, cancelled=cancelled)
        config = deepcopy(self.model.generation_config)
        config.ara.target_prompt_post_mcp = 0
        config.ara.target_token_post_mcp = 0
        config.do_sample = True
        # HF rejects BSP's default temperature=0 with do_sample=True. The
        # grammar processor still leaves exactly one finite token, so this
        # transport setting does not introduce stochastic token selection.
        config.temperature = 1.0
        config.top_p = 1.0
        config.top_k = 0
        # BSP interprets max_length as generation budget and ignores HF's
        # max_new_tokens in this path. Keep the native loop bounded explicitly.
        config.max_length = maximum
        config.max_new_tokens = None
        generation_started = time.monotonic()
        self.model.structured_processors = LogitsProcessorList([processor])
        self.model.native_decode_seconds = 0.0
        self.model.native_decode_calls = 0
        self.last_generation_metrics = None
        try:
            output = self.model.generate(**inputs, generation_config=config,
                                        logits_processor=self.model.structured_processors)
        finally:
            self.last_generation_metrics = {
                'generation_seconds': time.monotonic()-generation_started,
                'ttft_seconds': getattr(self.model, '_ttft', None),
                'native_decode_seconds': self.model.native_decode_seconds,
                'native_decode_calls': self.model.native_decode_calls,
                'structured_processor_seconds': processor.processing_seconds,
                'mask_calls': processor.mask_calls, 'torch_num_threads': torch.get_num_threads(),
                'prompt_tokens': len(input_ids)}
        generation_seconds = time.monotonic() - generation_started
        ids = output[0].tolist()
        generated = ids[len(input_ids):] if ids[:len(input_ids)] == input_ids else ids
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        terminated = processor.matcher.is_terminated()
        if processor.mask_calls == 0:
            raise ValueError('structured_processor_was_not_called')
        if terminated:
            jsonschema.Draft202012Validator(schema).validate(strict_json(text))
        if time.monotonic() >= deadline or cancelled():
            raise GenerationDeadline('late_or_disconnected_native_result')
        return {'text': text, 'finish_reason': 'stop' if terminated else 'length',
                'prompt_tokens': len(input_ids), 'completion_tokens': len(generated),
                'input_token_ids': input_ids, 'output_token_ids': generated,
                'grammar_terminated': terminated, 'mask_calls': processor.mask_calls,
                'schema_sha256': hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
                'compile_seconds': compile_seconds, 'generation_seconds': generation_seconds,
                'prompt_format': self.prompt_format,
                'prompt_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                'ttft_seconds': self.model._ttft, 'model_class': type(self.model).__name__}


class ServiceRuntime:
    def __init__(self, backend=None, audit_path=None):
        self.backend = backend
        self.state = 'ready' if backend is not None else 'loading'
        self.error = None
        self.lock = threading.Lock()
        self.audit_lock = threading.Lock()
        self.audit_path = Path(audit_path) if audit_path else None
        if self.audit_path:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)

    def health(self):
        busy = self.lock.locked()
        return {'state': 'busy' if self.state == 'ready' and busy else self.state,
                'inference_active': busy, 'error': self.error,
                'model': MODEL_ID, 'backend': 'ARA', 'structured_mode': 'xgrammar-mask-v1',
                'prompt_format': getattr(self.backend, 'prompt_format', 'legacy-v1'),
                'load_seconds': getattr(self.backend, 'load_seconds', None),
                'fadvise_approach2': True,
                'load_metrics': getattr(self.backend, 'load_metrics', None)}

    def audit(self, value):
        if self.audit_path:
            with self.audit_lock, self.audit_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def handler_class(runtime, model_id=MODEL_ID):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, value):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
            try:
                self.send_response(status)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Connection', 'close')
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                pass

        def do_GET(self):
            if self.path == '/health':
                self.send_json(200, {**runtime.health(), 'model': model_id})
            else:
                self.send_json(404, {'error': 'not_found'})

        def do_POST(self):
            if self.path != '/v1/chat/completions':
                self.send_json(404, {'error': 'not_found'})
                return
            request_id = uuid.uuid4().hex
            started = time.monotonic()
            record = {'request_id': request_id, 'status': 'rejected', 'started_monotonic': started}
            acquired = False
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 1 <= length <= 1024**2:
                    raise ValueError('request_body_size')
                try:
                    budget = float(self.headers.get('X-ARA-Timeout-Seconds', '60'))
                except ValueError:
                    budget = float('nan')
                valid_budget = math.isfinite(budget) and 0 < budget <= 60
                self.connection.settimeout(min(budget, 10) if valid_budget else 1)
                # Drain a size-bounded body before sending 400. Closing with an
                # unread body can reset loopback TCP before the error is read.
                body = self.rfile.read(length)
                if not valid_budget:
                    raise ValueError('deadline_budget_range')
                deadline = started + budget
                value = strict_json(body.decode('utf-8'))
                record['request'] = value
                schema, maximum = validate_request(value, model_id)
                if runtime.state != 'ready':
                    record['error'] = 'backend_not_ready'
                    self.send_json(503, {'error': 'backend_not_ready', 'state': runtime.state})
                    return
                acquired = runtime.lock.acquire(blocking=False)
                if not acquired:
                    record['error'] = 'backend_busy'
                    self.send_json(503, {'error': 'backend_busy'})
                    return
                def cancelled():
                    try:
                        if not select.select([self.connection], [], [], 0)[0]:
                            return False
                        return self.connection.recv(1, socket.MSG_PEEK) == b''
                    except BlockingIOError:
                        return False
                    except (OSError, ValueError):
                        return True
                self.connection.setblocking(False)
                result = runtime.backend.generate(value, schema, maximum, deadline, cancelled)
                self.connection.setblocking(True)
                if time.monotonic() >= deadline:
                    raise GenerationDeadline('late_result')
                record.update(status='generated', result=result)
                self.send_json(200, {'id': request_id, 'object': 'chat.completion', 'model': model_id,
                    'choices': [{'index': 0, 'finish_reason': result['finish_reason'],
                                 'message': {'role': 'assistant', 'content': result['text']}}],
                    'usage': {'prompt_tokens': result['prompt_tokens'], 'completion_tokens': result['completion_tokens']},
                    'timings': {'generation_seconds': result.get('generation_seconds'), 'ttft_seconds': result.get('ttft_seconds')},
                    'ara': {'structured_mode': 'xgrammar-mask-v1', 'grammar_terminated': result.get('grammar_terminated'),
                            'mask_calls': result.get('mask_calls'), 'schema_sha256': result.get('schema_sha256')}})
            except GenerationDeadline as error:
                record.update(status='deadline', error=str(error))
                self.connection.setblocking(True)
                self.send_json(504, {'error': str(error), 'request_id': request_id})
            except (ValueError, TypeError, KeyError, UnicodeDecodeError) as error:
                record.update(status='invalid_request_or_output', error=str(error))
                self.connection.setblocking(True)
                self.send_json(400, {'error': str(error), 'request_id': request_id})
            except Exception as error:
                record.update(status='backend_failure', error=f'{type(error).__name__}: {error}')
                self.connection.setblocking(True)
                self.send_json(502, {'error': record['error'], 'request_id': request_id})
            finally:
                if acquired and getattr(runtime.backend, 'last_generation_metrics', None) is not None:
                    record['native_trace'] = runtime.backend.last_generation_metrics
                if acquired:
                    runtime.lock.release()
                record['wall_seconds'] = time.monotonic() - started
                runtime.audit(record)

        def log_message(self, *args):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', default='/usr/share/llm/Qwen2.5-7B-Instruct')
    parser.add_argument('--port', type=int, default=18081)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--prompt-format', choices=('legacy-v1', 'compact-json-v1'), default='legacy-v1')
    args = parser.parse_args()
    runtime = ServiceRuntime(audit_path=args.audit)
    def load():
        try:
            runtime.backend = NativeARABackend(args.model_path, args.prompt_format)
            runtime.state = 'ready'
            audit_entry = {'event': 'backend_ready', 'model': MODEL_ID, 'load_seconds': runtime.backend.load_seconds}
            if hasattr(runtime.backend, 'load_metrics'):
                audit_entry.update(runtime.backend.load_metrics)
            runtime.audit(audit_entry)
        except BaseException as error:
            runtime.state = 'failed'
            runtime.error = f'{type(error).__name__}: {error}'
            runtime.audit({'event': 'backend_load_failed', 'error': runtime.error})
            # A dead loader with a live HTTP thread used to report "loading"
            # forever.  Exit non-zero so systemd can perform its bounded retry.
            os._exit(1)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_class(runtime))
    threading.Thread(target=load, daemon=True).start()
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
