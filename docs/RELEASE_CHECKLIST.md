# Release Checklist
- Run `ruff check .`
- Run `python -m pytest`
- Run CLI smoke tests (`virgo2 --help`, status/release-check/registry-validate/taxonomy-classify)
- Run `virgo2 release-check <vault_dir>`
- Run `virgo2 registry-validate <vault_dir>`
- Verify package metadata/version
- Update CHANGELOG
- Ensure no temp artifacts are committed
