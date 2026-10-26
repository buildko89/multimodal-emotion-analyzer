"""Schema masking and deterministic token selection for BSP optimum-ara.

The BSP passes the original prompt IDs on every callback, so the matcher accepts
the selected token here rather than inferring progress from input_ids.
"""
import time


class GenerationDeadline(TimeoutError):
    pass


class AraJSONLogitsProcessor:
    def __init__(self, compiled_grammar, vocab_size, *, deadline, max_tokens, cancelled=None):
        import xgrammar as xgr
        self.xgr = xgr
        self.matcher = xgr.GrammarMatcher(compiled_grammar)
        self.bitmask = xgr.allocate_token_bitmask(1, vocab_size)
        self.deadline, self.max_tokens = deadline, max_tokens
        self.cancelled = cancelled or (lambda: False)
        self.selected_tokens = []
        self.mask_calls = 0
        self.processing_seconds = 0.0

    def __call__(self, input_ids, scores):
        started = time.monotonic()
        import torch
        if time.monotonic() >= self.deadline or self.cancelled():
            raise GenerationDeadline('deadline_or_client_disconnected')
        if len(self.selected_tokens) >= self.max_tokens:
            raise ValueError('generation_token_limit')
        if self.matcher.is_terminated():
            raise ValueError('generation_after_grammar_termination')
        if self.matcher.fill_next_token_bitmask(self.bitmask):
            self.xgr.apply_token_bitmask_inplace(scores, self.bitmask, backend='cpu')
        self.mask_calls += 1
        token = int(torch.argmax(scores[0]).item())
        if not torch.isfinite(scores[0, token]):
            raise ValueError('grammar_has_no_finite_token')
        if not self.matcher.accept_token(token):
            raise ValueError('selected_token_violates_grammar')
        self.selected_tokens.append(token)
        # BSP do_sample=True is needed to invoke processors. A single finite
        # score makes its final sampler deterministic, retaining greedy choice.
        scores.fill_(float('-inf'))
        scores[0, token] = 0.0
        self.processing_seconds += time.monotonic()-started
        return scores
