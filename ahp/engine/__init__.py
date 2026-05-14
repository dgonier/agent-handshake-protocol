"""ahp.engine — protocol routing and thread management."""

from ahp.engine.errors import (
    IncompatibleTargetError,
    InvalidTargetTypeError,
    ProtocolError,
    UnauthorizedError,
)
from ahp.engine.router import DEFAULT_TIMEOUT, ProtocolEngine
from ahp.engine.scope import ScopePolicy, ScopeRule
from ahp.engine.thread_manager import Thread, ThreadManager

__all__ = [
    "DEFAULT_TIMEOUT",
    "IncompatibleTargetError",
    "InvalidTargetTypeError",
    "ProtocolEngine",
    "ProtocolError",
    "ScopePolicy",
    "ScopeRule",
    "Thread",
    "ThreadManager",
    "UnauthorizedError",
]
