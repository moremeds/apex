# PostgreSQL read API fixture

`rows.jsonl` contains real rows captured on 2026-09-22 from the production
`option_wizard.uw_scan` database. Each line records the query name, parameters,
capture timestamp and at most two rows. PostgreSQL JSON numeric literals are
preserved verbatim; tests load them with `parse_float=Decimal`.

These are source-data excerpts for serialization and coverage tests, not a
claim that every registry output has precisely these columns. Missing provenance,
analysis and IV-rank matches are retained. The capture session used a read-only
transaction default and a 30-second statement timeout. Tests never access the
network. SQL compilation and route behavior against the database are verified
separately by `scripts/check_pg_read_api.py`.
