"""Application layer - orchestration and workflow control.

No package-level re-exports: importing any ``src.application.*`` module executes this
file, and the lake queries (``src.application.lake``) must load without the
orchestrator, bootstrap container or PG repositories
(``tests/carve/test_lake_boundary.py``). Import from the defining module.
"""
