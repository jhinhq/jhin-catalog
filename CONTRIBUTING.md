# Contributing to jhin-catalog

Thanks for helping keep this index honest. Most of the pipeline is
mechanical; the parts that need people are the curated overlays, the
denylist, and the source rules. Those are where contributions land.

Inbound contributions are licensed under [Apache-2.0](LICENSE) for code and
[CC BY 4.0](LICENSE-DATA) for catalogue content, and signed off under the
[Developer Certificate of Origin](https://developercertificate.org/)
(`git commit -s`). There is no separate contributor licence agreement.

## The one rule: `data/` is generated

**Never hand-edit `data/`, `sources.lock`, or `schema/catalog.schema.json`.**

Those are build artefacts. `jhin-catalog verify` re-derives each of them and
compares bytes, so a hand edit is not a stylistic disagreement — it is a red
CI run, and it stays red until the edit is reverted:

```
$ uv run jhin-catalog verify
data/mcp/3f.jsonl: file bytes differ from render_shard of its own contents
$ echo $?
5
```

This is deliberate. A downstream Jhin deployment trusts these files enough to
write them into its database. If a human could quietly edit one, the
provenance chain from `sources.lock` to a published entry would be a
suggestion rather than a fact.

If a generated entry is wrong, you fix the **input**, not the output:

| what is wrong | where you fix it |
|---|---|
| A field's value (name, category, icon, endpoint, note) | `curated/mcp.yaml` or `curated/skills.yaml` |
| An entry should not exist at all | `curated/denylist.yaml` |
| Two entries should be one, or one should be two | `aliases:` on a curated override, or a source rule in `src/` |
| Every entry of some shape is wrong | the pipeline in `src/jhin_catalog/` — that is a code change with tests |

Then run `uv run jhin-catalog build --from-cache …` (or `sync`) and commit
the regenerated files **in the same pull request** as the input change, so
the diff shows cause and effect side by side.

## Setup

```bash
git clone https://github.com/jhin-dev/jhin-catalog.git
cd jhin-catalog
uv sync --frozen
```

Python 3.13 and [uv](https://docs.astral.sh/uv/) are the only prerequisites.
No database, no Docker, no credentials. `GITHUB_TOKEN` is optional and only
raises GitHub's rate limit; the build works without it and never writes it
anywhere.

## Gates

Every pull request must pass these locally before review. CI runs the same
five commands.

| command | what it checks |
|---|---|
| `uv run ruff check .` | lint |
| `uv run ruff format --check .` | formatting |
| `uv run mypy` | `--strict` over `src` and `tests` |
| `uv run pytest -q` | the suite (integration tests deselected) |
| `uv run jhin-catalog verify` | that `data/**` and the schema are genuinely generated |

The test suite opens no sockets. Upstream behaviour is reproduced with
recorded payloads under `tests/data/` and `httpx.MockTransport`; the single
exception is `tests/test_live_upstream.py`, which is marked `integration`,
deselected by default, and run only by the scheduled sync workflow.

## Adding or correcting a curated entry

Curated overlays are field-level. They do not replace an entry — they set the
specific fields you can vouch for and leave the crawl to supply the rest.
Every field an override sets is recorded on the entry in `curated_fields`, so
a reader can always tell which values a human asserted.

Edit `curated/mcp.yaml` (or `curated/skills.yaml`) and add an entry to the
`entries:` list:

```yaml
entries:
  - key: mcp:repo:github.com/tavily-ai/tavily-mcp
    kind: mcp
    fields:
      name: Tavily
      description: Web search and extraction built for agents.
      category: Search & web
      icon: search
      mcp_url: https://mcp.tavily.com/mcp/
      transport: streamable_http
      auth_hint: bearer
      auth_note: Use a Tavily API key as the bearer token.
      docs_url: https://docs.tavily.com/documentation/mcp
```

Rules the loader enforces, each of which fails the build with exit code 6:

- `key` is a `canonical_key` **or** any `alias_key` of the entry you mean.
  Find it with `uv run jhin-catalog stats --json`, or by grepping `data/`.
- `key` must be unique across the file. A duplicate is an error, not a
  last-one-wins merge.
- Every name under `fields:` must be a real, settable field on `McpEntry` or
  `SkillEntry`. An unknown field is an error and the message names it.
- The result must validate. `category` must be one of the twelve
  `CATALOG_CATEGORIES`, `icon` one of the icon tokens, `connector_type` one
  of the connector types, `mcp_url` `https://` with no `{template}` segment,
  and `connector_config` within 10 pairs / 64-char keys / 500-char values.

Two things worth knowing:

- **An override whose `key` matches nothing creates a new entry** from its
  `fields` alone. That is how you add something no crawl can find — an
  endpoint published only in a blog post, say. Such an entry needs enough
  fields to validate on its own, including `slug`, `name`, and `sources`.
- **`aliases:` forces a merge.** Listing another entry's key under `aliases:`
  unions the two components before dedupe runs. Use it when the crawl split
  one project in two and no shared key exists to join them.

Say in the pull request *how you verified* an endpoint or a claim — the
provider's docs page, the response to a `GET`, the commit you read. Curated
values outrank crawled ones, so they should be the best-evidenced values in
the file.

## Reporting a malicious or unwanted entry

The denylist is the removal path, and it is a one-line pull request.

Open `curated/denylist.yaml` and add:

```yaml
entries:
  - key: mcp:npm:some-typosquatted-package
    reason: Typosquat of `some-package`; exfiltrates the API key it is given.
```

`reason` is **required** and must be at least eight characters. It is not
paperwork — it is the only record of why an entry vanished, and the next
person to wonder why will read exactly that line.

Denylisted keys are dropped before any projection, so the entry leaves both
`data/**` and the exported `catalog.json` on the next build.

Three cases, three routes:

- **Malicious or abusive entry** (typosquat, credential harvester, endpoint
  that is not what it claims): open a public pull request against
  `curated/denylist.yaml`. Public is fine and preferable here — the package
  is already public, and the reason is useful to everyone.
- **"That is my project, please remove it"**: open an issue or a pull request
  saying so. You do not need to argue a legal theory; ownership is enough,
  and the request is honoured. Include the entry's `canonical_key` if you
  can, or just the repository URL.
- **A vulnerability in this crawler** (an SSRF in the fetcher, a path escape
  in skill-path handling, anything that harms a person running the build):
  report it privately through the repository's security advisory form, not in
  a public issue.

One operational wrinkle to know about: **a denylist key that matches nothing
fails the build.** A stale denylist is treated as a bug, because a key that
silently stops matching is a removal that silently stops working. If an
upstream deletes the thing you denylisted, remove the denylist line in the
same pull request that regenerates `data/`.

No curated key may also be denylisted; the test suite checks for that
contradiction.

## Changing the pipeline

Code changes carry a higher bar, because the output is a database somebody
else writes to.

- **Determinism is a hard invariant.** Nothing in the path that produces
  `data/**` may read a clock, a random number, an environment variable, or an
  unordered iteration. Clocks enter only as an injected `now`, and only reach
  `sources.lock`. `tests/test_determinism.py` builds twice and compares all
  512 files byte for byte.
- **Pointer-not-payload is a hard invariant.** No `SKILL.md` body, no server
  source, no tool schema, and no `configSchema` may reach a `RawRecord` or an
  entry. Discard it at the parse boundary, not later.
- **New sources need a licence answer.** A new upstream needs a row in the
  README's source table saying what it is and on what basis it is read,
  written as plainly as the existing rows. "It has a public API" is a real
  answer; pretending it is open data is not.
- **New identity keys need a rank.** Anything added to `KEY_SPACE_RANK`
  changes which key wins an election, which can move entries between shards.
  Expect the diff gate to fire, and say in the pull request that the mass
  change is intended.
- **Fixtures stay small and clean.** Payloads under `tests/data/` are
  hand-trimmed to 8 KiB or less, contain only synthetic or already-public
  identifiers, and use `*.example` hostnames wherever the hostname is not
  itself the thing under test.

## Commits and pull requests

- [Conventional Commits](https://www.conventionalcommits.org/):
  `feat(sources): …`, `fix(dedupe): …`, `data: …`, `docs: …`, `test: …`.
- Sign off every commit (`git commit -s`).
- Keep the pull request focused, and put the regenerated `data/` in the same
  one as the change that caused it.
- If a build legitimately breaches the diff gate, say so explicitly and why.
  `--allow-breaking` is for a reviewed, intentional mass change — a schema
  bump, a source retirement — never for turning a red build green.
