# jhin-catalog

An open index of **MCP servers** and **Agent Skills**, published as sorted
JSONL shards that a [Jhin](https://github.com/Teachmetech/Jhin) deployment
syncs into Postgres.

The catalogue crawls the official MCP registry, Smithery, npm, GitHub topic
search, and Claude Code plugin marketplaces; normalises every record into one
canonical entry; merges duplicates by identity key rather than by name; scores
popularity on fixed anchors; and writes the result as 512 byte-stable files
under `data/`. Two builds of the same upstream bytes produce the same output
bytes, so a pull request that changes `data/` is reviewable as a diff instead
of as a blob.

## Pointers, not payload

This repository stores **identity, metadata, and a source reference** — and
nothing else. It never stores:

- the body of a `SKILL.md`,
- MCP server source code, or any package contents,
- tool schemas, `configSchema` objects, or prompt text,
- upstream README prose.

A skill entry records where the skill lives (`owner/repo/path/SKILL.md`), the
commit it was seen at, and the handful of frontmatter fields needed to display
and route it. Fetching the skill is the consumer's job, from the upstream
repository, under the upstream repository's licence.

That boundary is a build gate, not a convention: §6 of
[`BUILD-SPEC`](#specification) forbids payload in `data/**`, and
`jhin-catalog verify` re-derives every file from its own parsed contents.

## Layout

```text
jhin-catalog/
  src/jhin_catalog/          # the crawler, the pipeline, the CLI
    types.py                 # every model, constant, and canonical-form helper
    http.py                  # bounded, redirect-refusing, retrying GETs
    sources/                 # one module per upstream
    normalize.py             # raw record -> Candidate (keys, slug, category)
    dedupe.py                # union-find over identity keys, field precedence
    score.py                 # popularity on fixed logarithmic anchors
    diffgate.py              # the gate that stops a bad crawl from landing
    build.py                 # planning, sharding, writing, projection
    cli.py                   # jhin-catalog {sync,build,verify,export,diff,stats}
  curated/                   # hand-written overlays and the denylist
    mcp.yaml  skills.yaml  denylist.yaml
  data/                      # GENERATED. 256 shards per kind. Never hand-edit.
    mcp/00.jsonl … ff.jsonl
    skills/00.jsonl … ff.jsonl
  schema/catalog.schema.json # GENERATED from the models
  sources.lock               # GENERATED. Per-source URL, hash, count, time.
  tests/                     # every gate; no socket is opened outside
                             # tests/test_live_upstream.py
```

`data/**`, `schema/catalog.schema.json`, and `sources.lock` are **build
artefacts**. A hand edit to any of them fails CI, because
`jhin-catalog verify` re-derives them and compares bytes.

## The entry model

Two kinds share one shape: `mcp` and `skill`. Every entry carries a
`canonical_key` of the form `{kind}:{space}:{value}` — for example
`mcp:repo:github.com/tavily-ai/tavily-mcp` or
`skill:skill:github.com/salesforceairesearch/agentforce-adlc/skills/design-review`.

Identity is the hard part of a catalogue like this, so it is done explicitly.
Each candidate emits every key it can justify (repo, endpoint URL, registry
name, npm name, PyPI name, Smithery qualified name), those keys are unioned
with a disjoint set, and the component's canonical key is the highest-ranked
key it holds. Repo keys claimed by many unrelated candidates — the monorepo
case, where forty servers live in `modelcontextprotocol/servers` — are
demoted before the union so a monorepo does not collapse into one entry.
Their GitHub stars still reach every sibling, because the signal join and the
identity join are deliberately separate.

An entry's shard is `sha256(canonical_key)[:2]`. Because the key is stable,
an entry never moves file unless its identity genuinely changed, and a
one-line diff stays a one-line diff.

Trust is a four-tier ladder, strongest first: `curated`,
`registry_verified`, `smithery_verified`, `indexed`. Only the top three are
publishable; `indexed` entries are indexed for search and reconciliation but
never offered to a user as a connect-me-now endpoint.

## Upstream sources — and their licence status

Be clear-eyed about this: **most of what this catalogue reads is not open
data.** It is factual metadata served by public APIs, gathered for
discovery and attribution, stored as pointers. The table below says exactly
what each source is and on what basis it is used.

| source | endpoint | auth | basis for use | notes |
|---|---|---|---|---|
| `registry` | `registry.modelcontextprotocol.io/v0.1/servers` | none | Public, documented, unauthenticated API of the Model Context Protocol project's server registry. The registry *software* is open source; the records are publisher-submitted metadata offered for discovery. | Strongest source. Supplies remotes, packages, and the `registry_verified` tier. |
| `smithery` | `registry.smithery.ai/servers` | none | **Not open data.** A commercially operated registry with a public read API, used under its published terms. Read-only, paginated politely, no credential presented. | Supplies `useCount`, a `verified` flag, and deployment URLs. Its `configSchema` is read to derive an auth hint and then discarded. |
| `npm` | `registry.npmjs.org/-/v1/search` | none | Public package-registry metadata. Package metadata is factual; each package's own licence is recorded verbatim in the entry's `license` field and is not altered by anything here. | Download and dependent counts, keywords, repository links. Never a source of endpoints. |
| `github_topics` | `api.github.com/search/repositories` | optional `GITHUB_TOKEN` | GitHub's REST API under GitHub's Terms of Service and rate limits. Repository *metadata* only — stars, forks, topics, SPDX id, description. **No repository content is fetched by this source.** | Throttled at 30 requests/minute client-side. A token raises the ceiling; it is never written to any artefact. |
| `marketplaces` | `raw.githubusercontent.com` + `api.github.com/repos/*/git/trees` | optional `GITHUB_TOKEN` | Public repositories that publish a `.claude-plugin/marketplace.json`. Each repository stays under **its own licence**, recorded per entry where the upstream declares one. | The only source that reads file contents. It reads a `SKILL.md`, parses the leading frontmatter block, and **discards the body before the record is constructed**. |
| `curated` | this repo, `curated/*.yaml` | — | Written here, licensed with the rest of the catalogue. | Field-level overlays applied after the merge. Human judgement, versioned. |

Two honest caveats:

1. **Descriptions are upstream text.** An entry's `name` and `description`
   are short factual strings authored by the upstream publisher, collapsed to
   a single line and truncated. They are stored so a person can recognise the
   thing; they are attribution, not republication. If you own an upstream
   project and want its record removed, see
   [Reporting an entry](CONTRIBUTING.md#reporting-a-malicious-or-unwanted-entry)
   — the denylist is a one-line pull request and it is honoured.
2. **A listing is not an endorsement.** `indexed` means "we saw it". It does
   not mean the endpoint was reached, the code was read, or the publisher was
   verified. Nothing in this repository ever dials an MCP endpoint:
   `url_unverified: false` means only that a source this index trusts named
   that URL, never that a handshake succeeded. Entries carrying
   `url_unverified: true` are the ones where not even that much is true, and
   every consumer is expected to respect the flag.
3. **Free text in a record is written by strangers.** `description`,
   `category`, `marketplace` and `plugin` come from whoever published the
   upstream. They are length-bounded and stripped of control characters; they
   are not screened for meaning, and no `trust_tier` claims otherwise. An
   agent that reads them must treat every one as untrusted data.

The crawler is a well-behaved client: a single identifying User-Agent
(`jhin-catalog/0.1.0 (+https://github.com/jhin-dev/jhin-catalog)`), bounded
response sizes, refused redirects, exponential backoff that honours
`Retry-After`, and a client-side token bucket on GitHub search. It never
authenticates to Smithery or npm, and it opens no socket during the test
suite.

## Building

```bash
uv sync --frozen

uv run jhin-catalog sync            # crawl, build, gate, write data/ + sources.lock
uv run jhin-catalog verify          # re-derive and compare; no network, no writes
uv run jhin-catalog export --out catalog.json
```

| command | what it does |
|---|---|
| `sync` | Fetches every source, runs the pipeline, checks the gate, and writes `data/**` and `sources.lock`. Nothing is written until every gate passes. |
| `build --from-cache DIR` | The same pipeline over recorded payloads, with no network. |
| `verify` | Asserts every line parses, every key lands in the shard its hash demands, keys are globally unique, each file equals `render_shard` of its own contents, the schema matches the models, every curated key resolves, and every denylist entry has a reason. |
| `export --limit N` | Writes the Jhin `catalog.json` projection. |
| `diff --against PATH` | Prints a `DiffReport` per kind against another root. Exits 3 if a threshold would fail. |
| `stats` | Counts per kind, trust tier, category, and source, plus a popularity decile histogram. |

Exit codes are part of the contract: `0` success, `1` internal error, `2`
usage, `3` diff gate, `4` fetch or source fault, `5` verify violation or a
normalise/dedupe error, `6` bad curated input. `--json` makes any command
print exactly one canonical JSON object on stdout and nothing else.

A scheduled workflow (`.github/workflows/sync.yml`) runs `sync` daily and
opens a pull request. Nothing lands unattended without review: `commit` mode,
which pushes to the default branch and cuts a release, is reachable only from
`workflow_dispatch`, where a person chose it. That matters most on the first
build of an empty tree, where there is no baseline and the diff gate is
skipped for want of one.

Additions never fail the gate — growth is the normal case — so what stops an
unreviewed repository from putting text into the index is the reviewed
allowlist in `curated/skills.yaml`, not the gate.

### Determinism

Nothing in the build reads a clock, a random number, an environment variable,
or an unordered iteration. Identical input bytes produce identical output
bytes. The only timestamp anywhere in the output is `fetched_at` in
`sources.lock`, and it arrives as an injected `now` parameter so tests pin it.

Records are `json.dumps(..., ensure_ascii=False, sort_keys=True,
separators=(",", ":"), allow_nan=False)` plus `\n`, written in binary so no
platform can translate a line ending. No key is ever omitted from `data/**` —
defaults are materialised, `None` is `null`, empty tuples are `[]`. All 256
shards per kind are written on every build; an empty shard is a zero-byte
file, so a deletion is visible in git rather than hidden by a vanishing path.

## The diff gate

This is the part that protects consumers, so it gets its own section.

Upstream registries serve empty pages. Smithery's list endpoint silently
truncates at 500 results unless a `seed` parameter is present, and returns
`200 OK` with an empty array beyond that. A naive crawler reads that as *the
servers are gone*, rewrites `data/`, and a downstream Jhin deployment
faithfully deletes its catalogue. The gate exists because that failure is
indistinguishable from success at the HTTP layer.

Before a single byte is written, `sync` compares the new build against the
committed one and refuses to proceed when:

- more than **5 %** of baseline entries would disappear (`max_drop_fraction`),
- more than **20 %** of shared entries would change (`max_change_fraction`),
- or any source that previously returned records now returns **zero**
  (`check_source_counts` — treated as a fetch fault, never as a real
  deletion).

Additions never fail the gate; growth is the normal case. The gate is
skipped below 100 baseline entries so a bootstrap or a deliberately small
catalogue is not permanently un-buildable. `--allow-breaking` overrides it,
and is meant for a reviewed, intentional mass change — a schema bump, a
source retirement — not for getting a red build green.

When the gate fires, the workflow fails and no pull request is opened. That
is the whole point.

## How Jhin consumes this

Two paths, both read-only from Jhin's side.

**1. The full index → Postgres.** A Jhin deployment syncs `data/**` into its
own tables. Each line is a complete, self-describing record validated against
`schema/catalog.schema.json`; `canonical_key` is the natural primary key and
`alias_keys` gives the reconciliation path when an upstream renames itself
(a plugin rename becomes an alias, never a new row and a dead one). Shards
make an incremental sync cheap: only the files whose bytes changed need
re-reading.

**2. The publishable slice → `catalog.json`.** `jhin-catalog export` projects
publishable MCP entries into exactly the fifteen keys of Jhin's `CatalogApp`
model (`packages/connectors/src/jhin_connectors/catalog.json`), with the
same omit-when-default convention the hand-written file uses, ordered by
`rank_score` — trust tier first, then popularity, then key — and capped at
`--limit`. The result is drop-in: same slug rules, same twelve categories,
same icon token set, same `auth_hint` / `url_unverified` / `stdio_only`
semantics.

An entry is publishable only when its trust tier is above `indexed`, it is
not deprecated, and it has a description. A crawled entry must additionally
offer *something* to connect to — an endpoint, a native connector type, or an
honest `stdio_only` flag with a setup note. A `curated` entry is exempt from
that last test: half of `curated/mcp.yaml` describes a server a person
self-hosts, and those records deliberately carry a `setup_note` and
`url_unverified: true` in place of an endpoint. Everything else stays in the
index and out of the UI.

## Provenance and attribution

Every entry names its own sources. `sources[]` holds one `SourceRef` per
contributing upstream, each with the source id, that source's own primary id
for the record, and the canonical human-readable page for it. An entry
merged from the registry, npm, and GitHub topics carries all three, so a
reader can check any claim at its origin.

Build-level provenance lives in `sources.lock`:

```json
{
  "entry_count": 25492,
  "fetched_at": "2026-08-29T04:11:07Z",
  "page_count": 255,
  "sha256": "…",
  "source_id": "registry",
  "url": "https://registry.modelcontextprotocol.io/v0.1/servers"
}
```

`sha256` is a rolling hash over every raw page body in fetch order, so it
changes when upstream changes and only then. Together with the commit that
produced it, that pins any published shard to the exact crawl behind it.

If you redistribute the catalogue, attribute it:

> jhin-catalog — https://github.com/jhin-dev/jhin-catalog — CC BY 4.0

Attributing this catalogue does not discharge any obligation you have to the
upstream projects it points at. Each of those keeps its own licence, and the
entry records it where the upstream publishes one.

## Contributing

Curated entries, corrections, and denylist reports are the contributions that
matter most — they are where human judgement enters an otherwise mechanical
pipeline. See [CONTRIBUTING.md](CONTRIBUTING.md). The one rule worth
repeating here: **`data/` is generated and must never be hand-edited.**

## Licence

- **Code** — Apache License 2.0, see [`LICENSE`](LICENSE).
- **The compiled catalogue** — `data/`, `curated/`, `sources.lock`, and
  `schema/catalog.schema.json` — Creative Commons Attribution 4.0
  International, see [`LICENSE-DATA`](LICENSE-DATA).

<a id="specification"></a>The normative build specification — every constant,
formula, and byte-level rule this implementation is held to — is the
`BUILD SPECIFICATION v1` document tracked with the project.
