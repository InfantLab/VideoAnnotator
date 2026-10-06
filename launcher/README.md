# The launcher (for maintainers)

`videoannotator-start` is how researchers start VideoAnnotator (spec 024): it finds Docker or
Podman, shares the folders they choose read-only at their real paths, starts the image on
127.0.0.1 and opens the viewer, connected. Researchers' instructions are in
[docs/installation/INSTALLATION.md](../docs/installation/INSTALLATION.md); what it must do is in
[specs/024-container-feels-local/contracts/launcher.md](../specs/024-container-feels-local/contracts/launcher.md).

| File | What it is |
| --- | --- |
| `videoannotator-start` | POSIX `sh`, for Linux and macOS (no bashisms; shellcheck clean) |
| `videoannotator-start.ps1` | The same behaviour in Windows PowerShell 5.1+ |
| `videoannotator-start.cmd` | Double-click wrapper for the `.ps1` (`-ExecutionPolicy Bypass`) |
| `install.sh`, `install.ps1` | One-line installers, with a "Start VideoAnnotator" shortcut |

## Structure

Both scripts are small named functions plus a `main`:

- **sh**: functions are `va_*`. Sourcing the script with `VA_SOURCE_ONLY=1` defines them without
  running `main`, which is how the tests call them one at a time.
- **PowerShell**: functions are `Verb-Va*` (`Get-VaSettingsPath`, `ConvertTo-VaContainerPath`,
  ...). Dot-sourcing with `$env:VA_SOURCE_ONLY = '1'` defines them without running `Main`.
- Every call to the engine goes through one wrapper (`va_engine` / `Invoke-VaEngine`), so tests
  replace it and record what would have run.
- Messages are worded exactly as in the contract's Messages table: one line, plus the next step.
  Exit codes: 0 started or already running, 1 a problem, 2 the researcher cancelled.

`VA_VERSION` is the placeholder `@VERSION@` in the repository. Release CI replaces it with the
release version, which pins the image (`ghcr.io/infantlab/videoannotator:<version>`). From a
source checkout the launcher runs `:latest` (or `--image`) and says "development launcher".

## Shared test cases

`tests/launcher/cases.json` is the single description of expected behaviour, read by both test
suites, so the two scripts can't drift apart. Change behaviour by changing a row here first.

It is an object of named groups, each a list of rows:

```json
{ "name": "what this row shows", "os": "any|linux|macos|windows",
  "input": { ... }, "expect": { ... } }
```

| Group | Function under test | `input` | `expect` |
| --- | --- | --- | --- |
| `paths` | `va_container_path` / `ConvertTo-VaContainerPath` | `path`, `n` (share number, or `results`) | `container` |
| `normalise` | `va_normalise` / `ConvertTo-VaNormalPath` | `path`, `home` | `path` |
| `classify` | `va_classify` / `Get-VaClassification` | `path`, `home`, `shares`, `results` | `result` (`ok`, `broad <kind>`, `refused`, `duplicate`, then flags `replaces`, `nested_results`) |
| `settings` | `va_settings_path` / `Get-VaSettingsPath` | `home`, `xdg` (`appdata` on Windows) | `path` |
| `run` | `va_build_run` / `New-VaRunArgs` | engine, OS, shares, missing shares, results, port, image, uid/gid, GPU flags, Docker Desktop | `args`, one per element |
| `engine` | `va_engine_detect` / `Get-VaEngine` | each engine `running`/`stopped`/`absent`, `saved`, `option` | `engine` (and `announced`), or `message` |
| `gpu` | `va_gpu_flags` / `Get-VaGpuFlags` | `engine`, `nvidia`, `runtimes`, `cdi` | `flags`, `note` |
| `messages` | `va_explain_error` / `Get-VaErrorMessage` | `engine`, `stderr`, `code` | `message` (first line), `status`, optional `detail` |

Rows with `"os": "windows"` run only under Pester; the others run under both (`any`) or under
`bats` with that OS set. Whole-command behaviour (first start, restart, share, update) is tested
in the `.bats` files with a pretend engine and server (`flow_setup` in `helpers.bash`).

## Running the tests

```bash
shellcheck launcher/videoannotator-start launcher/install.sh
bats tests/launcher                       # needs bats and jq
```

```powershell
Invoke-ScriptAnalyzer -Path launcher -Recurse -Severity Warning
Invoke-Pester tests/launcher
```

The end-to-end test, `tests/launcher/test_launcher_e2e.sh`, starts a real container with Docker or
Podman; CI runs it on Linux with both (the `launcher-e2e` job), using an image built from
`tests/launcher/Dockerfile.e2e` that adds the test-only `stub_pipeline`.
