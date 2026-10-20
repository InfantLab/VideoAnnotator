/**
 * Where each page's "More in the docs" goes (spec 029 R16): a section of the
 * published docs. Links, never loaded (constitution I). `docsLinks.test.ts`
 * checks every file exists under `docs/` and every anchor is a heading.
 */
export const DOCS_BASE = 'https://infantlab.github.io/VideoAnnotator/';

export interface DocsTarget {
  /** Under `docs/`. */
  file: string;
  /** A heading's slug in that file, or none for the top. */
  anchor?: string;
}

export const DOCS_LINKS = {
  home: { file: 'usage/GETTING_STARTED.md', anchor: 'three-ways-to-start' },
  help: { file: 'usage/GETTING_STARTED.md', anchor: 'using-the-viewer' },
  words: { file: 'usage/GETTING_STARTED.md', anchor: 'words' },
  demos: { file: 'usage/GETTING_STARTED.md', anchor: 'demos' },
  firstRun: { file: 'usage/GETTING_STARTED.md', anchor: 'your-first-run' },
  videos: { file: 'usage/videos_and_projects.md', anchor: 'the-videos-page' },
  folders: { file: 'usage/videos_and_projects.md', anchor: 'folders' },
  projects: { file: 'usage/videos_and_projects.md', anchor: 'projects' },
  projectSettings: { file: 'usage/videos_and_projects.md', anchor: 'settings-a-project-remembers' },
  workflows: { file: 'usage/videos_and_projects.md', anchor: 'workflows' },
  newRun: { file: 'usage/videos_and_projects.md', anchor: 'new-run' },
  runs: { file: 'usage/videos_and_projects.md', anchor: 'runs' },
  exporting: { file: 'usage/videos_and_projects.md', anchor: 'exporting-results' },
  noServer: { file: 'usage/videos_and_projects.md', anchor: 'without-a-server' },
  table: { file: 'usage/exporting_results.md', anchor: 'the-table' },
  videoPage: { file: 'usage/video_page.md', anchor: 'tracks' },
  videoRuns: { file: 'usage/video_page.md', anchor: 'data-about-runs' },
  ask: { file: 'usage/ask_ai.md', anchor: 'trying-a-question' },
  compare: { file: 'usage/ask_ai.md', anchor: 'running-on-the-whole-video' },
  pipelines: { file: 'usage/pipeline_specs.md', anchor: 'overview' },
  settings: { file: 'usage/CLIENT_TOKEN_SETUP_GUIDE.md', anchor: 'overview' },
} satisfies Record<string, DocsTarget>;

export type DocsPage = keyof typeof DOCS_LINKS;

/** MkDocs with directory URLs: `usage/x.md#a` is served at `usage/x/#a`. */
export function docsUrl(page: DocsPage): string {
  const { file, anchor } = DOCS_LINKS[page] as DocsTarget;
  const path = file.replace(/(^|\/)README\.md$/, '$1').replace(/\.md$/, '/');
  return `${DOCS_BASE}${path}${anchor ? `#${anchor}` : ''}`;
}
