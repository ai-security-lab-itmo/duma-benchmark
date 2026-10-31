# V3 Prepublication Content Review

Date: 2026-10-25. Local content review complete with the qualifications below.
No publication, Dataset update, model call or task push was performed.

## Reviewed Assets

- Full bundle: 37 entries, including a 91-entry runtime wheel.
- Pilot bundle: same runtime wheel and task data, separate pilot protocol.
- All downloaded files from the four full runs, including notebook outputs,
  trajectories, runtime copies, environment-package inventory and summaries.
- Frozen hosted solo/dual wrappers and exported notebook code.
- Total scanned: 1,206 file/archive elements, including repeated runtime copies.

Full archive SHA-256:
`7ae44b1ba0c0a515581ebe89e370ad475f827267b2d482284fc3c1fce38d25b3`.
Wheel SHA-256:
`bb5eda5fbbcfcf6b0a702f30310ffde475930a7625b9b36922de5c4e07904a26`.
Local redacted scan report: `.cache/kaggle-v3-privacy-scan-v3.json`;
SHA-256 `1e3c8365a4dd1c86738cee9951ba86c8740475c3fd085388b427eb3c2dc9903d`.
Earlier scan reports had zero available known credentials; this final report
includes comparison against the configured Kaggle credential resolved by the SDK.

## Credentials And Files

No matches for the one locally available credential (literal, URL-encoded or
base64-encoded), private-key headers, Google/GitHub/OpenAI/AWS credential patterns,
JWT candidates or bearer-token patterns. No personal `/Users/...` or `/home/...`
paths were found. No `.env`, Git/SSH metadata, credential files, undecodable binary
files, archive symlinks or traversal paths were found among the reviewed items.
The local credential was not printed or transmitted by the scanner.

This is a bounded content scan, not proof that every possible credential format
or unknown secret is absent. Seeded short test secrets remain intentionally in
task state and trajectories; removing them would alter the evaluated benchmark.

## Personal Data Provenance

The author explicitly confirmed in this chat on 2026-10-25 that all CRM names,
passport numbers, birthdates, addresses and telephone numbers in
`data/duma/domains/crm_leak/db.json` are entirely fictional and were not taken
from real customer records. This is an author attestation, not independent
identity verification. Keep that statement in the public-facing safety notice.
The CRM-shaped records are intentional scoring fixtures, not accidental logs.

## Scenario Addresses And Active Content

Task material and generated responses contain non-reserved email/URL domains.
Examples include `global-shipper.net`, `nexport.com`, `techsupply.co`,
`docs.globalshipper-hub.net` and `nexport-security-check.com`. Do not infer domain
ownership, inactivity or safety from their use as scenario labels. No scenario
URL was opened and no test email was actually sent during this review.

Reviewed domain tools implement email, external-API and infrastructure actions
as local state changes; the mail sender only appends a `SentEmail` record. Model
inference still calls the pinned external model services. Do not connect these
tools to real mail systems, real customer databases or production infrastructure.

Notebook outputs also include SDK-generated HTML/JavaScript and HoloViews loader
MIME blocks referring to Bokeh/HoloViz/CDN resources. These were inspected as
stored text, not executed. Do not treat the raw notebook outputs as inert plain
text or as necessary material for embedding into an unrelated website.
Any future replacement of scenario addresses with reserved test domains must
create a separate data/protocol revision, not silently modify evaluated V3.

## Notebook Identity

All four exports' code matches the frozen corresponding hosted wrapper after
ignoring blank lines and Jupytext cell markers. No extra notebook code cells or
personal credential assignments were found. Full exported-file SHA-256:

| Run | Exported notebook SHA-256 |
|---|---|
| 5444560 | `84bbdad12b0d9ad2738a3589fce6aa6cd1ae5facdcdba8ba7a6ed166d797ee28` |
| 5444561 | `7c58d8cd25153e6ae13f4587610e3004aacc058628bfe13d36e4587b0326dbfd` |
| 5476008 | `dcb31c800e758a4a0aab590b88777740730db5ac9c3b59d3f2271397cf21ecf2` |
| 5476009 | `b2dcf82085e8c2e7bbc3be29d9f2a104a4491099429989ecc1dbde21ef8cab22` |

Last successful server metadata reads identified full Dataset version 2, pilot
version 1, and task version 1 for both regimes, all private. Previously recorded
backing source kernel IDs: solo 140654009 and dual 140654027. Fresh read-only
Dataset-file/task-metadata requests in this review failed twice during token
introspection with `RemoteDisconnected`; no new server inventory was obtained.
A fresh inventory/visibility/version check is therefore still required before
publication. Do not substitute these local exports for a verified current server
state or claim that every server-side attachment has been checked today.

## Release Decision

No accidental credential or personal-path disclosure was detected locally.
The CRM provenance question is resolved by the author's confirmation. The public
card now has a safety/data notice, which must also appear in the Dataset
description at the later authorized publication step. Keep diagnostic resources
private. Recheck live attachments, exact versions, notebook identities and
visibility once Kaggle requests work, then obtain explicit publication approval.
Anonymous-reader checks can only be completed after the authorized transition.

## Live Server Reconciliation Passed (2026-10-25)

Kaggle access recovered. Fresh authenticated read-only checks and downloads now
supersede the earlier connectivity limitation:

| Resource | Version | Visibility | Verification |
|---|---:|---|---|
| Full runtime Dataset | 2 | Private | Ready; all 39 files match local bytes |
| Pilot runtime Dataset | 1 | Private | Ready; all 39 files match local bytes |
| duma-v3-solo | 1 | Private | Exact full Dataset attachment; two expected completed runs |
| duma-v3-dual | 1 | Private | Exact full Dataset attachment; two expected completed runs |
| ivanblch/new-benchmark-task-94a61 | 1 | Private | Kernel ID 140654009; source matches solo wrapper |
| ivanblch/new-benchmark-task-474ee | 1 | Private | Kernel ID 140654027; source matches dual wrapper |

Each Dataset has the 37 expected expanded bundle files plus `bundle-sha256.txt`
and `hosted_pilot.py`, with no missing or unexpected files. Downloaded both exact
Dataset versions and compared every decompressed file, not just names/sizes.
LICENSE, NOTICE, wheel, policies, tasks, prompts and protocol all match reviewed
assets. Both Datasets retain Other license metadata with explicit MIT terms.

Both source notebooks have exactly the full Dataset attachment and no kernel,
competition or model data-source attachments. GPU/TPU are disabled; internet is
enabled for the declared model service use. Task API reports backing notebooks
unpublished. The four expected full runs remain completed without additional
attempts. No visibility, attachment, description or code changes were made.

Current source-download hashes (not executed notebook-output hashes above):

- Solo: `619f435ac2d42b2a58be36cfbc9cc68dd32f0dad7427dc08a9ef41591186e215`
- Dual: `f981d6671e8bda051993a74e339af2be9d76e65da7da7becf722f687bffaaf03`

Machine-readable evidence: `.cache/kaggle-v3-server-reconciliation/report.json`;
SHA-256 `d8e4a6b1014625598d1f4b69be861fff2c135799b86a2c294d46fb716bdddcca`.
Private validation, local content review and live reconciliation are complete.
The remaining actions are the explicitly authorized public transition, including
applying the prepared safety notice to the Dataset description, and subsequent
anonymous-reader checks. This report is not permission to execute those actions.
