# Building the JOSS paper locally

JOSS papers are compiled with the Open Journals toolchain (Pandoc via `openjournals/inara`).

## Expected files

- `paper/paper.md`
- `paper/paper.bib`

## Option A: Docker (recommended)

From the repository root:

```bash
docker run --rm \
  --volume $PWD/paper:/data \
  --env JOURNAL=joss \
  openjournals/inara
```

On success, `paper/paper.pdf` will be created next to `paper/paper.md`.

## Option B: GitHub Action

Open Journals provides a GitHub Action for PDF generation. If we add it later, the output PDF will appear as a workflow artifact in the Actions tab.

## Notes

- If you are on Windows PowerShell, `$PWD` is still supported, but Docker volume path handling may differ depending on your Docker installation.
- The compilation step is sensitive to YAML metadata formatting and missing citations; if compilation fails, check the frontmatter in `paper/paper.md` and the citekeys in `paper/paper.bib`.
