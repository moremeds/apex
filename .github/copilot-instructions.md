<!-- IMPORTANT: This file was created by an AI assistant after scanning the repository and finding no existing agent instruction files.
If you already have project-specific guidance, paste it into this file or tell the assistant where to look and it will merge intelligently. -->

# Copilot / AI agent instructions (short, focused)

Purpose: give an AI coding assistant the minimal, project-specific facts it needs to be productive immediately.

- Repo scan: No source or config files were found by the automated scan that produced this file. Replace the placeholders below with concrete paths/commands from this repo. If this is incorrect, point me to the directory to scan (for example `web/` or `src/`) and I will re-run the extraction and update this file.

What to prioritize
- Find the main entry points and build files: look for `package.json`, `pyproject.toml`, `go.mod`, `build.gradle`, `pom.xml`, `Cargo.toml`, or `src/` and `main.*` files. Add the paths here.
- Identify CI files: `.github/workflows/**`, `circleci/**`, `Jenkinsfile`.
- Tests: list test command(s) and test directories (examples: `npm test`, `pytest`, `go test ./...`).

Project-specific patterns (fill these in)
- Language(s): [e.g. TypeScript, Python, Go]
- Runtime / framework: [e.g. Next.js, Flask, Spring Boot]
- Key directories:
  - app code: <path>
  - infra / infra-as-code: <path>
  - tests: <path>
  - scripts: <path>

Build / run / test (replace with exact commands)
- Build: <command to build the project>
- Run locally: <command(s) to run dev server or app>
- Run tests: <command to run tests>
- Lint/typecheck: <commands>

Examples & idioms to follow
- Where to put new modules/features: add path and brief rationale.
- Typical service boundary examples: mention files that show service APIs or adapters (e.g. `services/*`, `api/*`).
- Database/migrations: point to migration tool and migration folder.

Integration points and external dependencies
- External services the app depends on (e.g. AWS S3, Postgres, Stripe). Add connection details needed for local dev if any (or point to `env.example`).
- Secrets and config: where are environment variables documented? (`.env.example`, `config/`)

Commits & PRs
- Preferred branch naming convention: <e.g. feature/..., fix/...>
- Testing requirement for PRs: list checks that must pass (CI jobs).

If you are an AI agent reading this repo
1. Re-run a file search for common build files listed above. If you find any, replace the top 'Repo scan' line with a short summary and update the sections below with concrete commands and paths.
2. When editing code, prefer small, focused changes and include or update tests near the changed code.
3. If you cannot find a build/test command, run discovery heuristics: look for `scripts/`, `Makefile`, `tox.ini`, `setup.cfg`, or `Procfile` and report back.

Where to update this file
- Keep a short, concrete list here. AI agents should update this file when they discover the build/test/run commands or important paths.

---
Last automated update: 2025-11-23. To improve: point the assistant to the project root or a subdirectory with the source code.
