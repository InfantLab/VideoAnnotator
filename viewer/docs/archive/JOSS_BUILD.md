# Building the JOSS paper locally (Archived)

> **Archived 2026-07-08.** Video Annotation Viewer no longer has its own JOSS paper — this repo
> and [VideoAnnotator](https://github.com/InfantLab/VideoAnnotator) are now evaluated jointly as a
> single combined paper hosted in the VideoAnnotator repository
> ([joss-reviews#10182](https://github.com/openjournals/joss-reviews/issues/10182); vav's own
> submission, [joss-reviews#10183](https://github.com/openjournals/joss-reviews/issues/10183),
> remains open only as a pointer to the combined review). The paper files this doc describes now
> live at [`docs/archive/paper/`](./paper/) for historical reference only — they are not maintained
> and the `draft-pdf` CI workflow that built them has been removed.

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
