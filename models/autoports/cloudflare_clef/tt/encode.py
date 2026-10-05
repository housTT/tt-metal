import hashlib
import importlib.util
import struct
import sys
from pathlib import Path

RELEASE_MODULE_NAME = "clef_release_joint_schema_model"
RELEASE_FILE = "joint_schema_model.py"
_release = None


def release_module(snapshot=None):
    global _release
    if _release is not None and (snapshot is None or _release.__file__ == str(Path(snapshot) / RELEASE_FILE)):
        return _release
    if snapshot is None:
        from models.autoports.cloudflare_clef.tt.loader import resolve_snapshot

        snapshot = resolve_snapshot()
    path = Path(snapshot) / RELEASE_FILE
    spec = importlib.util.spec_from_file_location(RELEASE_MODULE_NAME, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[RELEASE_MODULE_NAME] = module
    spec.loader.exec_module(module)
    _release = module
    return module


def encode(tokenizer, record, processor=None, max_length=16384, max_state_tokens=None):
    return release_module().encode_record(
        tokenizer, record, max_length=max_length, max_state_tokens=max_state_tokens, processor=processor
    )


def schema_ids_and_question_starts(tokenizer, record):
    m = release_module()
    tokens = m._tokens
    schema_ids = tokens(tokenizer, "\n\nSCHEMA FIELDS:\n")
    question_starts = []
    for question_index, (question_id, question) in enumerate(record["questions"].items()):
        schema_ids.extend(
            tokens(
                tokenizer,
                f"\nFIELD {question_index + 1}\nID: {question_id}\nTYPE: {question['type']}\nINSTRUCTION: ",
            )
        )
        question_starts.append(len(schema_ids))
        instructions = question.get("instructions")
        if instructions is None or instructions == "":
            instructions = str(question_id)
        schema_ids.extend(tokens(tokenizer, m.render(instructions)))
        schema_ids.extend(tokens(tokenizer, "\nALLOWED OPTIONS:\n"))
        for option_index, (option_id, description) in enumerate(m.question_options(question)):
            schema_ids.extend(tokens(tokenizer, f"OPTION {option_index + 1}: "))
            semantics = {"option_id": option_id}
            if description is not None:
                semantics["description"] = description
            schema_ids.extend(tokens(tokenizer, m.render(semantics)))
            schema_ids.extend(tokens(tokenizer, "\n"))
        schema_ids.extend(tokens(tokenizer, "END FIELD\n"))
    return schema_ids, question_starts


def prefix_ids(tokenizer):
    m = release_module()
    return m._tokens(tokenizer, f"<|im_start|>system\n{m.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n")


def suffix_ids(tokenizer):
    m = release_module()
    return m._tokens(tokenizer, "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")


def split_for_cache(encoded, tokenizer, record):
    ids = list(encoded.input_ids)
    schema, question_starts = schema_ids_and_question_starts(tokenizer, record)
    suffix = suffix_ids(tokenizer)
    tail = schema + suffix
    split = len(ids) - len(tail)
    if split < 0 or ids[split:] != tail:
        raise ValueError("schema and suffix ids do not match the tail of the encoded record")
    head = prefix_ids(tokenizer)
    if ids[: len(head)] != head:
        raise ValueError("prefix ids do not match the head of the encoded record")
    if len(question_starts) != len(encoded.questions):
        raise ValueError("question count differs between the schema recomputation and the encoded record")
    for start, question in zip(question_starts, encoded.questions):
        if question.question_span[0] != split + start:
            raise ValueError(f"question {question.question_id} span does not start at the recomputed offset")
    return ids[:split], ids[split:], encoded.questions


def cache_key(token_ids):
    ids = [int(t) for t in token_ids]
    return hashlib.sha1(struct.pack(f"<{len(ids)}i", *ids)).hexdigest()


def load_tokenizer(snapshot=None):
    from transformers import AutoTokenizer

    if snapshot is None:
        from models.autoports.cloudflare_clef.tt.loader import resolve_snapshot

        snapshot = resolve_snapshot()
    return AutoTokenizer.from_pretrained(snapshot)


def load_processor(snapshot=None):
    from transformers import AutoProcessor

    if snapshot is None:
        from models.autoports.cloudflare_clef.tt.loader import resolve_snapshot

        snapshot = resolve_snapshot()
    return AutoProcessor.from_pretrained(snapshot)
