# ReproLLM usage: local ResNet18 checkpoint inventory

## Scope

This project used ReproLLM to inspect the identity of a local PyTorch `.pth`
checkpoint used by the tongue-coating classifier. This is an artifact-recording
and reproducibility metadata case, not an evaluation of diagnosis quality.

## Evidence table (version-scoped)

| Field | Actual value |
|---|---|
| Using project | `LiangRichard13/LingshuSmartLink` |
| Verified module | Local `.pth` identity / directory-inventory scan of the ResNet18 tongue-coating classifier checkpoint |
| Upstream issue | [`ReproLLM#15`](https://github.com/EnumaElish123/ReproLLM/issues/15) (filed 2026-11-06 by this account) |
| Original discovery | 2026-11-06, on ReproLLM source `dfb72eda57abb47cc0fc972982e770c6199c5bd2` (unreleased 0.7.0 line; wheel `reprollm-0.7.0-py3-none-any.whl`, sha256 `ca65c2b2f99487746ffc6222ae75d14a793e243deaa4ab5c7799b605b397ef09`, built from that commit) — **pre-fix version** |
| Post-fix version | ReproLLM source `d73f45f2b5334cd05e382b86b137e37d9aabda12` containing fix [`b4d6c713f`](https://github.com/EnumaElish123/ReproLLM/commit/b4d6c713fe9cf16c1d97364fa07dda5f4ef24a60); wheel `reprollm-0.7.0-py3-none-any.whl` sha256 `6c1fccde646c25cb67d5bdfa0f4ff4a10f614048b42c82336f3e535baff2d28f`, built from that commit — **post-fix version, see §Post-fix check** |
| Lingshu commit | `b09becc18b1b9cd0e4d6ac04ed61a7dcbb63554b` (branch `codex/single-case-replay`, clean tree; weights are untracked local assets) |
| Environment | Windows 10 (10.0.26200), CPython 3.12.13 (uv-managed), isolated venv per check |
| Real classifier file | `trial/assets-local/classifier/tongue_coating_resnet18.pth` (repo-relative; untracked, stays local) — sha256 `b8ee07a926b12bc4322b81860fd09edbea6bb2343215fec680b67ca8b796deaa`, 44,799,262 bytes |
| Original-run command (pre-fix, fixture) | `reprollm lock . --offline --hash-large-files` inside a minimal project declaring `models.primary: {provider: local, id: weights}` containing only `.pth` files |
| Post-fix command (real weights) | `reprollm lock . --offline --hash-large-files` then `reprollm audit . --format json --fail-on never`, run in an isolated project containing a **local copy of the real classifier checkpoint** |
| Pre-fix observed result | Directory declaration of `.pth`-only folders: `weights.total_size_bytes = 0`, `hashed = true`, `sha256` = empty-set digest (`sha256:e3b0c442…`), **unchanged when `.pth` bytes changed**; explicit-file declaration hashed normally. Full CLI matrix in Issue #15. |
| Post-fix observed result (real weights) | lock exit 0; `models.primary.local.weights`: `total_size_bytes = 44799262`, `hashed = true`, `sha256 = sha256:ee4e394994303c14a0c408e5a8ce71b45c31ae912c73fb3b3c367ea19fbbdc21` (directory aggregate), `revision = null` (no `config.json` present; revision is not the signal this check relies on) |
| Result scope | Real application classifier file, copied into an isolated check project; plus a synthetic A/B fixture matrix (Issue #15). **No model inference was executed in the isolated check** — it verifies lock inventory only. The broader application trial (fixture/local/live runs) is documented separately in [`docs/reprollm-single-case-trial.md`](../reprollm-single-case-trial.md). |
| Conclusions NOT supported | Does not certify medical/diagnostic effectiveness; does not make ReproLLM verify the whole multi-agent system; does not extend to end-to-end reproducibility |
| Public evidence | Issue #15 (CLI matrix) + this document + follow-up comment on Issue #15 |

## ReproLLM versions

- Original observed source: `dfb72eda57abb47cc0fc972982e770c6199c5bd2` (unreleased
  0.7.0 line, installed from a wheel built from that commit — not PyPI 0.6.1).
- Upstream `.pth` directory support: [`b4d6c713f`](https://github.com/EnumaElish123/ReproLLM/commit/b4d6c713fe9cf16c1d97364fa07dda5f4ef24a60)
- Post-fix application test: **established** — see the evidence table above; executed
  2026-11-09 on a wheel built from `d73f45f2b5334cd05e382b86b137e37d9aabda12`.
- Public PyPI 0.6.1 does **not** include the above change.

## What we ran

- Lingshu source revision: `b09becc18b1b9cd0e4d6ac04ed61a7dcbb63554b`
- OS / Python: Windows 10 (10.0.26200) / CPython 3.12.13
- ReproLLM command: `reprollm lock . --offline --hash-large-files` and
  `reprollm audit . --format json --fail-on never` (executed by
  `LiangRichard13` on 2026-11-09)
- Declared local weight scope: `weights/` in an isolated check project holding a
  copy of the real classifier checkpoint
- Inputs and privacy: real model weights stay local; no patient material or
  secrets published. The weight copy was deleted after the check; the original
  file was never moved.

## Observed findings

1. **Original (pre-fix) behavior**: directory declarations whose inventory
   consisted only of `.pth` files recorded `total_size_bytes = 0` and the
   empty-set digest, invariant under content changes; explicit-file declarations
   hashed normally. Reproduced with a four-project CLI matrix (Issue #15).
2. **Explicit-file declaration** already worked pre-fix and post-fix.
3. **Post-fix (real weights)**: the directory declaration reports
   `total_size_bytes = 44799262`, `hashed = true`, and a content-sensitive
   aggregate `sha256` (`ee4e3949…`). The aggregate digest is ReproLLM's
   directory-inventory hash, not the raw file hash (`b8ee07a9…`); both are
   recorded here because they answer different questions. With a fixed
   `config.json` present, `config_sha256` stays independent of weight bytes
   (verified separately on a synthetic A/B fixture).

Audit context: `reprollm audit --fail-on never` on the bare isolated project
exits 0 with 34 findings, of which 3 CRITICAL relate to the bare project setup
(`code.git_repo`, `env.dependency_manifest_present`,
`model.revision_pinned` — the last one is expected without a `config.json`),
not to `.pth` recognition. Audit exit 0 only means the process did not exit on
findings; it does not mean all rules passed.

## Contribution to ReproLLM

- Report: https://github.com/EnumaElish123/ReproLLM/issues/15
- Upstream fix: https://github.com/EnumaElish123/ReproLLM/commit/b4d6c713fe9cf16c1d97364fa07dda5f4ef24a60
- Follow-up: (direct Issue comment URL — added with this document)

## Limitations

The case covers only the recorded checkpoint-inventory behavior. It does not
validate medical accuracy, complete model reproducibility, or end-to-end agent
behavior. All conclusions are tied to the exact software/input conditions above.
