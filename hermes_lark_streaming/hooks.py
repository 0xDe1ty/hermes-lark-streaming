"""原生观察 Hook：只补充数据，不接管正文投递或 Gateway 控制流."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import OrderedDict
from typing import Any

_logger = logging.getLogger(__name__)
_requests: OrderedDict[tuple[str, str, str], str | None] = OrderedDict()
_requests_lock = threading.Lock()


def _request_key(payload: dict[str, Any]) -> tuple[str, str, str] | None:
    from gateway.session_context import get_session_env
    from hermes_constants import get_hermes_home

    from .plugin import PLATFORM

    source = get_session_env("HERMES_SESSION_MESSAGE_ID")
    request_id = payload.get("api_request_id")
    if get_session_env("HERMES_SESSION_PLATFORM") != PLATFORM or not source or not request_id:
        return None
    return str(get_hermes_home().resolve()), source, str(request_id)


def _request_effort(payload: dict[str, Any]) -> str | None:
    request = payload.get("request")
    body = request.get("body") if isinstance(request, dict) else None
    if not isinstance(body, dict):
        return None
    extra = body.get("extra_body")
    for item in (body, extra):
        if not isinstance(item, dict):
            continue
        effort = item.get("reasoning_effort")
        reasoning = item.get("reasoning")
        if not effort and isinstance(reasoning, dict):
            if reasoning.get("enabled") is False:
                return "none"
            effort = reasoning.get("effort")
        if isinstance(effort, str) and effort.strip().lower() in {
            "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra",
        }:
            return effort.strip().lower()
    return None


def observe_request(**kwargs: Any) -> None:
    key = _request_key(kwargs)
    if key is None:
        return
    with _requests_lock:
        _requests[key] = _request_effort(kwargs)
        _requests.move_to_end(key)
        while len(_requests) > 1024:
            _requests.popitem(last=False)


def _dispatch(kind: str, payload: dict[str, Any]) -> None:
    from gateway.session_context import get_session_env
    from hermes_constants import get_hermes_home

    from .adapter import ACTIVE
    from .plugin import PLATFORM

    if get_session_env("HERMES_SESSION_PLATFORM") != PLATFORM:
        return
    # 入站 ID 由官方 contextvars 随观察事件传递，不使用模型 turn_id 匹配 Gateway UUID
    source_id = get_session_env("HERMES_SESSION_MESSAGE_ID")
    if not source_id:
        return
    home = get_hermes_home().resolve()
    for adapter in list(ACTIVE):
        if adapter.cards._cfg._home != home:
            continue
        loop = adapter._card_loop
        if loop is None or loop.is_closed():
            continue

        def schedule(target: Any = adapter) -> None:
            task = asyncio.create_task(target.cards.observe(source_id, kind, payload))
            task.add_done_callback(_report)

        loop.call_soon_threadsafe(schedule)


def _report(task: asyncio.Task) -> None:
    if not task.cancelled() and task.exception():
        _logger.error("Card observer failed", exc_info=task.exception())


def observe_start(**kwargs: Any) -> None:
    _dispatch("start", kwargs)


def observe_reasoning(**kwargs: Any) -> None:
    if kwargs.get("kind") == "reasoning":
        _dispatch("reasoning", kwargs)


def observe_tool_start(**kwargs: Any) -> None:
    _dispatch("tool_start", kwargs)


def observe_tool_end(**kwargs: Any) -> None:
    _dispatch("tool_end", kwargs)


def observe_usage(**kwargs: Any) -> None:
    key = _request_key(kwargs)
    if key is not None:
        with _requests_lock:
            effort = _requests.pop(key, None)
        kwargs = {**kwargs, "reasoning_effort": effort}
    _dispatch("usage", kwargs)


def observe_end(**kwargs: Any) -> None:
    _dispatch("end", kwargs)
