# Release Process

1. Ensure the `dev` branch is up to date and all CI checks pass.
2. Update documentation and bump the version references in `package.json` or other manifests if required.
3. Merge `dev` into `main` using a merge commit or fast-forward.
4. Create a semantic version tag, e.g.:
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```
5. Draft a GitHub release summarising key changes. Attach build artifacts only if necessary.
6. Update `docs/CHANGELOG.md` (create if missing) with highlights for the release.

Automations:
- GitHub Actions workflow (`.github/workflows/ci.yml`) validates Python builds and optional Node builds on every push/pull request.
- Extend the workflow with publish steps (PyPI, Docker, etc.) once automated deployments are required.
