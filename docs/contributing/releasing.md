# Release checklist

Public releases are tag-driven. A signed or annotated `vX.Y.Z` tag starts the
GitHub release workflow, which verifies the source, builds the wheel and source
archive, validates package metadata, creates checksums, and publishes a GitHub
release with generated notes.

The package version is derived from Git tags by `uv-dynamic-versioning`; release
commits do not contain a hard-coded version bump.

## One-time GitHub setup

After the workflow files reach `main`, configure the repository once:

1. In **Settings → Pages**, choose **GitHub Actions** as the publishing source.
2. In the repository **About** panel, use
   `Dynamic graph runtime for building, adapting, and observing multi-agent LLM systems in Python.`
   as the description.
3. Set the website to `https://frontier-ai-next.github.io/gMAS/`.
4. Add the topics `agentic-ai`, `ai-agents`, `graph`,
   `graph-neural-networks`, `llm`, `mcp`, `multi-agent-systems`, `python`,
   `rustworkx`, and `workflow-automation`.
5. Enable private vulnerability reporting so `SECURITY.md` and the issue-form
   security link lead to the advisory form.

The Pages workflow will publish the site on the next documentation-related
push to `main`, or it can be started manually from the Actions tab.

## Preflight

Create the release from a clean `main` branch after CI and documentation checks
have passed:

```bash
uv sync --all-extras --group test --group lint --group typecheck --group docs
uv run tox -e lint,quick
uv run --group docs mkdocs build --strict
uv build
uvx twine check dist/*
```

Then verify:

- `gmas.__version__` will match the intended tag;
- the wheel and source archive contain `src/gmas`, package metadata, and the
  benchmark reproduction code expected for the release;
- no `.env`, credentials, benchmark logs, browser artifacts, or local build
  output is tracked;
- README and documentation links resolve;
- benchmark tables still match their fixed run artifacts and manuscript
  snapshot.

## Create the tag

Use semantic versioning. Final releases use tags such as `v0.1.0`; prereleases
may use tags such as `v0.2.0-rc.1`.

```bash
git switch main
git pull --ff-only
git tag -a v0.1.0 -m "gMAS v0.1.0"
git push origin v0.1.0
```

Pushing the tag is the publication trigger. Do not move or reuse a published
tag. If the release needs correction, fix `main` and publish a new patch tag.

## GitHub release behavior

`.github/workflows/release.yml` performs the following steps:

1. installs the release dependencies;
2. runs Ruff and the full test suite;
3. builds the wheel and source archive;
4. validates both distributions;
5. attaches the distributions and `SHA256SUMS.txt` to a generated GitHub
   release;
6. marks hyphenated versions such as `v0.2.0-rc.1` as prereleases.

Release-note categories are configured in `.github/release.yml`. Pull requests
should use the repository's `feature`, `enhancement`, `bug`, `fix`,
`documentation`, or `dependencies` labels when applicable.

## Optional PyPI publication

PyPI publication is disabled by default. To enable it, configure a PyPI Trusted
Publisher for this repository and `.github/workflows/release.yml`, then add the
repository Actions variable `PUBLISH_TO_PYPI=true`. The workflow uses OpenID
Connect and does not require a long-lived PyPI token.

Only final tags are uploaded to PyPI. Prerelease tags still create GitHub
prereleases with downloadable artifacts.

## After publication

1. Install the release wheel into fresh Python 3.12 and 3.13 environments.
2. Import `gmas` and print `gmas.__version__`.
3. Run a two-agent local smoke graph with a deterministic caller.
4. Confirm the release contains the wheel, source archive, checksum file,
   generated notes, and expected tag.
5. Confirm GitHub Pages still builds from the released source.
6. Update benchmark result pages only when a new complete matched run is
   available.
