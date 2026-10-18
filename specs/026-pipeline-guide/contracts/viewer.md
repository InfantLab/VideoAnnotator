# Contract: viewer

## `PipelineGuide` (new, `viewer/src/components/PipelineGuide.tsx`)

Used on the Pipelines page and in the job wizard's pipeline choice (`PipelineSelectionStep`).

- Header: `title` (fallback `display_name`), availability/readiness badge (existing), `summary`
  (fallback `description`). Deprecated notice first when present.
- Settings, via `DynamicPipelineParameters`:
  - **What to measure**: behavioural parameters, open.
  - **Technical settings (n · m changed)**: technical parameters, collapsed; the heading counts
    parameters and those differing from their default; "all at defaults" when m = 0.
- **About** (collapsed on the wizard, open on Pipelines): homepage link; each benchmark as
  "name: measure figure" linking to `source`, with `note`; "No published benchmark" for `[]`;
  `quality_note` when present; citation keys with **Copy BibTeX** (all entries, joined as the
  methods paragraph does).

## `DynamicPipelineParameters` (changed)

- Groups by `group` (absent → technical); label from `label`; shows `unit`, `min`, `max` (already
  supported).
- `appliesWhen`: hides a parameter whose conditions don't match the current values, unless its value
  differs from its default, then shows it with "Not used with these settings".
- No change to the config object it produces (same keys, same values).

## Tests (Vitest)

- `PipelineGuide.test.tsx`: title/summary fallbacks; About with benchmarks, empty benchmarks, quality
  note; Copy BibTeX copies every entry.
- `DynamicPipelineParameters.test.tsx`: grouping and counts; changed count updates on edit;
  `appliesWhen` hide/show and the changed-but-not-applicable case.
- Contract: a fixture of `GET /pipelines/face_analysis/schema` with the new fields parses into the
  viewer types.
