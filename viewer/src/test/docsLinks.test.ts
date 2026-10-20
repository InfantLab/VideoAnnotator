/** Every "More in the docs" link lands on a real section (spec 029 SC-007). */
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

import { DOCS_BASE, DOCS_LINKS, docsUrl, type DocsPage } from '@/lib/docsLinks';

const DOCS = path.resolve(__dirname, '../../../docs');
const SRC = path.resolve(__dirname, '..');

/** Python-Markdown's toc slug, which MkDocs uses for heading ids. */
function slug(heading: string): string {
  return heading
    .normalize('NFKD')
    .replace(/[^ -~\s]/g, '')
    .replace(/[^\w\s-]/g, '')
    .trim()
    .toLowerCase()
    .replace(/[-\s]+/g, '-');
}

function headings(file: string): string[] {
  let fenced = false;
  const found: string[] = [];
  for (const line of readFileSync(file, 'utf8').split('\n')) {
    if (line.startsWith('```')) fenced = !fenced;
    const match = !fenced && /^#{1,6}\s+(.*?)\s*#*\s*$/.exec(line);
    if (match) found.push(slug(match[1].replace(/\*\*/g, '').replace(/`/g, '')));
  }
  return found;
}

function sources(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) return entry.name === 'test' ? [] : sources(full);
    return /\.(ts|tsx)$/.test(entry.name) ? [full] : [];
  });
}

describe('docs links', () => {
  it.each(Object.entries(DOCS_LINKS))('%s goes to a heading that exists', (_page, target) => {
    const file = path.join(DOCS, target.file);
    expect(existsSync(file), target.file).toBe(true);
    if ('anchor' in target && target.anchor) expect(headings(file)).toContain(target.anchor);
  });

  it('builds MkDocs directory URLs', () => {
    expect(docsUrl('projects')).toBe(`${DOCS_BASE}usage/videos_and_projects/#projects`);
  });

  it('slugs as MkDocs does', () => {
    expect(slug('Data, About, Runs')).toBe('data-about-runs');
    expect(slug('🧠 Overview')).toBe('overview');
  });

  it('every page key used in the viewer is in the map', () => {
    const used = new Set<string>();
    for (const file of sources(SRC)) {
      for (const match of readFileSync(file, 'utf8').matchAll(/page="(\w+)"|docsUrl\('(\w+)'\)/g)) {
        used.add(match[1] ?? match[2]);
      }
    }
    for (const page of used) expect(Object.keys(DOCS_LINKS)).toContain(page as DocsPage);
  });

  it('links to no wiki', () => {
    for (const file of sources(SRC)) {
      expect(readFileSync(file, 'utf8'), file).not.toMatch(/deepwiki|wiki\.github/i);
    }
  });
});
