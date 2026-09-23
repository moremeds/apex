"""Transport-neutral lake queries shared by REST and MCP.

Callers pass a ``LakeServices`` and typed arguments; failures raise ``LakeError`` with
a stable code. Nothing here imports ``src.api`` or starts PG/streaming machinery.
"""
