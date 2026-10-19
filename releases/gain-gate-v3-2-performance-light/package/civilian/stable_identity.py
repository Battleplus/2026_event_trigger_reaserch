"""Portable strict JSON identities extracted from the existing W1 contract.

`encoded` and `key_digest` preserve the original function implementation;
no historical simulator, fixed action layout, model, or NumPy import is used.
"""
import hashlib
import json
import math


def encoded(value):
    def check(x, path):
        if type(x) is dict:
            if any(type(k) is not str for k in x):
                raise ValueError('CANDIDATE_JSON_KEY_TYPE:' + path)
            for k, v in x.items():
                check(v, path + '.' + k)
        elif type(x) is list:
            for i, v in enumerate(x):
                check(v, path + '[' + str(i) + ']')
        elif x is None or type(x) in (str, bool, int):
            pass
        elif type(x) is float and math.isfinite(x):
            pass
        else:
            raise ValueError('CANDIDATE_JSON_TYPE:' + path + ':' + type(x).__name__)
    check(value, '$')
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')


def key_digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def bind_decision(observation, candidates):
    """Bind only public data; a hidden-state digest is deliberately excluded."""
    return {
        'schema': 'relief-public-decision/1',
        'observation_sha256': key_digest(observation),
        'candidate_catalog_sha256': key_digest(candidates),
    }
