import time
from typing import Any, Dict


def now_ms() -> int:

    return int(time.time() * 1000)


def make_message(*,source: str, data: Dict[str, Any]) -> Dict[str, Any]:
    
    return {
        "ts": now_ms(),
        "source": source,
        "data": data,
    }
