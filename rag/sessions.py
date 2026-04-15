"""
Conversation session store.

Each session holds:
  - message history (last MAX_TURNS turns of user+assistant messages)
  - optional doc_id filter (pin query to a specific document)

Sessions are in-memory; they expire after EXPIRE_SECONDS of inactivity.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

MAX_TURNS     = 10       # keep last N user+assistant pairs
EXPIRE_SECS   = 3600     # 1 hour idle TTL


@dataclass
class Session:
    session_id: str
    doc_id: Optional[str]                      = None
    messages: List[Dict[str, str]]             = field(default_factory=list)
    last_active: float                         = field(default_factory=time.time)

    def add_turn(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        # Keep only last MAX_TURNS pairs (each pair = 2 messages)
        if len(self.messages) > MAX_TURNS * 2:
            self.messages = self.messages[-(MAX_TURNS * 2):]
        self.last_active = time.time()

    def history_for_llm(self) -> List[Dict[str, str]]:
        """Return messages in the format Anthropic SDK expects."""
        return list(self.messages)

    def touch(self):
        self.last_active = time.time()


class SessionStore:
    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def get_or_create(self, session_id: str, doc_id: Optional[str] = None) -> Session:
        self._evict_expired()
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id=session_id, doc_id=doc_id)
        session = self._sessions[session_id]
        if doc_id and session.doc_id is None:
            session.doc_id = doc_id
        session.touch()
        return session

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def delete(self, session_id: str):
        self._sessions.pop(session_id, None)

    def all_ids(self) -> List[str]:
        return list(self._sessions.keys())

    def _evict_expired(self):
        now = time.time()
        expired = [sid for sid, s in self._sessions.items()
                   if now - s.last_active > EXPIRE_SECS]
        for sid in expired:
            del self._sessions[sid]
