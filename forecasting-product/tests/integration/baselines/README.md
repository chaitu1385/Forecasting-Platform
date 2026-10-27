# M5 numerical references and empirical gates

## What exists, and what does not

`m5_daily_reference.json` is the **input-only, library-independent seasonal-naive
oracle**, not an empirical learned-model baseline. It covers all 105 series,
76,755 input rows, two 28-period folds, and 5,880 observations per model. The
exact naive WMAPE is `7921/7945`; signed bias is `479/7945`.

No authoritative learned-model baseline was found in the repository or available
history of `m5_daily_baseline.json`. The original gates were introduced without
that file. Captured runs are therefore candidates for a **first empirical
reference under `raw-observations-v1`**, not proof of historical no-regression.
Capture, a successful CI run, and agreement with the naive oracle are not approval.

The two original tests retain their original numerical requirements:

- `test_baseline_no_regression`: champion WMAPE ≤ reference × **1.05**;
  **every** model WMAPE ≤ its reference × **1.10**.
- `test_baseline_fva_preserved`: naive minus champion WMAPE must be **positive**
  and at least **80%** of the approved reference improvement.

Missing/unapproved references, missing scores/models, nonfinite values, wrong
fixtures, wrong scoring policy, incomplete populations and incompatible native
profiles now **fail**, never skip. The naive-only oracle replaces neither gate.
The optional calibration helper is not a new empirical-coverage qualification.

## Capture in a dedicated native qualification job

Use the integrated numerical/gate code and the independently qualified dependency
profile. Do not run these installation commands in a shared developer environment.
Use native Linux/Windows x64, the exact Python patch recorded in the corresponding
`requirements/manifest.json` entry, and the matching `py311`, `py312` or `py313`
**dev** lock. No package upgrade, fixture reduction, model pruning, seed override,
or hyperparameter change is part of baseline capture.

Inputs to the existing full M5 fixture:

| Environment variable | Required evidence |
|---|---|
| `M5_PROFILE_LOCK` | Actual `requirements/py<minor>/dev.txt`; raw bytes must match its manifest entry |
| `M5_PROFILE_INSTALL_REPORT` | Native pip `--report` JSON from the hash-locked restore, including the selected numerical/runtime wheels |
| `M5_PROFILE_QUALIFICATION` | Successful `scripts/check_dependency_profile.py --profile dev` JSON from the same environment |
| `M5_CANDIDATE_OUTPUT` | New JSON output path; the matching `.predictions.parquet` contains all 17,640 keyed model observations |
| `M5_BASELINE_PATH` | Explicit approved reference for that native profile when available; otherwise the canonical `m5_daily_baseline.json` is required |

Example in PowerShell, **only in a clean, dedicated native CI environment**,
from `forecasting-product` after dependency-profile integration:

```powershell
$tag = python -c "import sys; print(f'py{sys.version_info.major}{sys.version_info.minor}')"
$out = "artifacts\m5-$tag"
New-Item -ItemType Directory -Force "$out\runtime" | Out-Null
$env:TEMP = (Resolve-Path "$out\runtime").Path
$env:TMP = $env:TEMP
$env:M5_PROFILE_LOCK = "requirements\$tag\dev.txt"
$env:M5_PROFILE_INSTALL_REPORT = "$out\install.json"
$env:M5_PROFILE_QUALIFICATION = "$out\native-dev.json"
$env:M5_CANDIDATE_OUTPUT = "$out\candidate.json"

python scripts\lock_dependencies.py --check
python -m pip install --require-hashes --only-binary=:all: --force-reinstall `
  --report $env:M5_PROFILE_INSTALL_REPORT -r $env:M5_PROFILE_LOCK
python -m pip check
python scripts\check_dependency_profile.py --profile dev --output $env:M5_PROFILE_QUALIFICATION
python -m pytest tests\integration\test_m5_daily_backend.py --no-cov `
  --junitxml="$out\m5.xml" --basetemp="$out\pytest"
```

Check each command's exit status in automation. The force-reinstall is confined
to that clean qualification environment so pip records every selected wheel,
including otherwise-satisfied packages. Preserve the existing native CI thread
settings. Equivalent Linux CI commands may set the same environment variables.
GitHub's actual `GITHUB_RUN_ID`, `GITHUB_SHA` and `RUNNER_ARCH` are captured;
never supply invented run/source identities.

The fixture runs the unchanged full workload once, validates every actual/key
and naive prediction, independently recomputes every model score, and retains
its shared run identity and deferred-publication arguments. It records an
**unapproved** snapshot in JUnit and optionally writes the JSON plus raw Parquet.
Existing output paths are not overwritten. With no approved reference yet, the
two empirical tests are expected to fail explicitly; that capture job is **not**
a green release gate. Do not suppress those failures.

## Review and approval

Archive the native JUnit, candidate JSON, raw prediction Parquet, exact lock and
manifest, pip install/check evidence, and native profile report together with
their SHA-256 identities and actual source/run IDs. An independent reviewer must
verify completeness, hand-oracle agreement, per-model WMAPE/FVA, the selected
winner, approved stable/non-yanked profile provenance, and the raw artifact hash.

The legacy script now prepares a draft from those **existing** observations; it
does not run another fit, round scores, claim seed 42, or approve its own output:

```powershell
python scripts\bless_m5_baseline.py --frequency daily `
  --candidate artifacts\m5-py312\candidate.json `
  --output artifacts\m5-py312\review-draft.json
```

It refuses to overwrite a file or write into the approved-reference directory.
The draft always has `approved_baseline: false`, `approval: null`, and `seed: null`
(no seed override was captured). After independent review, the reference owner
may commit a schema-v2 `kind: m5_empirical_reference` record with:

- `approved_baseline: true`;
- `approval.reviewed_by`, an actual `review_reference`, and timezone-qualified
  `approved_at`;
- `approval.candidate_sha256`: the printed SHA-256 of the entire captured
  `provenance` object, serialized with sorted keys, compact separators and no
  NaN values. Keep that captured object's `approved_baseline: false` unchanged.

The approval record is a review/audit assertion, not a cryptographic signature;
the script and tests cannot authenticate who performed the external review.
Do not fill it with placeholder identities or approve merely because a run passes.
Review the initial reference as such; there is no missing historical learned-model
golden against which to claim backward no-regression.

Each approved record is specific to its Python patch, OS descriptor/architecture,
interpreter platform, complete installed profile, and selected wheel hashes.
Use separate reviewed files and `M5_BASELINE_PATH` for different native profiles.
Source commits/run IDs may change in a subsequent regression run; fixture,
policy, population and environment identities must still match. Kernel/image,
wheel or profile changes require explicit requalification, not relaxed tolerances.
There is no claim of cross-OS bitwise model identity.

Finally rerun the **same complete** integration suite against the approved
reference with fresh candidate output paths. Both original empirical gates
must pass. Keep archived runs/reference files immutable; record intentional
updates as new reviewed evidence rather than retroactively relabeling old runs.
