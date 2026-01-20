import time

CACHE_TTL = 300 #seconds (5 minutes = 300 seconds)

retrieval_cache = {}
llm_cache = {}

def _is_valid(entry):
    return (time.time() - entry["time"]) < CACHE_TTL

def get_from_cache(cache, key):
    if key in cache and _is_valid(cache[key]):
        return cache[key]["value"]
    return None

def set_cache(cache, key, value):
    cache[key] = {
        "value": value,
        "time": time.time()
    }