/**
 * The viewer's table rows match the exporter's (spec 028 R5).
 *
 * `tests/fixtures/viewer_contract/expected_table.csv` is written by the real
 * exporter (`tests/unit/test_viewer_table_fixture.py`) from the same files.
 */
import { readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

import { COLUMNS, NOT_RECORDED, classify, readFiles, rowsFromFiles, type Row } from '@/lib/table/rows';

const FIXTURES = path.resolve(__dirname, '../../../../tests/fixtures/viewer_contract');

const files = (dir: string) =>
  readdirSync(dir)
    .filter((name) => statSync(path.join(dir, name)).isFile())
    .map((name) => ({ name, text: readFileSync(path.join(dir, name), 'utf8') }));

/** RFC 4180, as Python's csv module writes it. */
function parseCsv(source: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = '';
  let quoted = false;
  for (let i = 0; i < source.length; i++) {
    const c = source[i];
    if (quoted) {
      if (c === '"' && source[i + 1] === '"') {
        cell += '"';
        i++;
      } else if (c === '"') quoted = false;
      else cell += c;
    } else if (c === '"') quoted = true;
    else if (c === ',') {
      row.push(cell);
      cell = '';
    } else if (c === '\n') {
      row.push(cell);
      rows.push(row);
      row = [];
      cell = '';
    } else cell += c;
  }
  if (cell || row.length) rows.push([...row, cell]);
  return rows;
}

const NUMBERS = new Set(['event', 'start', 'end', 'frame', 'value', 'confidence', 'x', 'y', 'width', 'height']);

// Python writes 1.0 where JavaScript has 1; compare numbers as numbers.
function same(column: string, expected: string, actual: Row[keyof Row]): boolean {
  if (actual === null || actual === undefined) return expected === '';
  if (NUMBERS.has(column)) return expected !== '' && Number(expected) === actual;
  return String(actual) === expected;
}

describe('the table twin (lib/table/rows.ts)', () => {
  const [header, ...expected] = parseCsv(readFileSync(path.join(FIXTURES, 'expected_table.csv'), 'utf8'));
  const rows = rowsFromFiles(files(FIXTURES), 'demo_clip');

  it('has the exporter\'s columns, author last', () => {
    // The fixture gains `author` with table format version 2 (spec 028 T047).
    expect([...COLUMNS].filter((c) => c !== 'author')).toEqual(header);
    expect(COLUMNS[COLUMNS.length - 1]).toBe('author');
  });

  it('makes the same rows as the exporter, cell by cell', () => {
    expect(rows.length).toBe(expected.length);
    for (let i = 0; i < expected.length; i++) {
      header.forEach((column, j) => {
        const actual = rows[i][column as keyof Row];
        if (!same(column, expected[i][j], actual)) {
          throw new Error(
            `row ${i + 2} (${rows[i].pipeline} ${rows[i].measure}), column ${column}: ` +
              `expected ${JSON.stringify(expected[i][j])}, got ${JSON.stringify(actual)}`,
          );
        }
      });
    }
  });

  it('leaves author empty on pipeline rows', () => {
    expect(rows.every((r) => r.author === '')).toBe(true);
  });

  it('reads older files without provenance as not recorded', () => {
    const legacy = rowsFromFiles(files(path.join(FIXTURES, 'legacy')), 'demo_clip');
    expect(legacy.length).toBeGreaterThan(0);
    const versions = new Set(legacy.filter((r) => r.pipeline !== 'vlm_annotation').map((r) => r.videoannotator_version));
    expect(versions).toEqual(new Set([NOT_RECORDED]));
  });

  it('skips a file it can\'t read, saying why', () => {
    const { read, skipped } = readFiles([
      { name: 'x_scene_detection.json', text: '{not json' },
      { name: 'x_speech_recognition.vtt', text: 'WEBVTT\n\n00:00.000 --> 00:01.000\nhi\n' },
      { name: 'notes.txt', text: '' },
    ]);
    expect(read.map((r) => r.name)).toEqual(['x_speech_recognition.vtt']);
    expect(skipped.map((s) => s.file).sort()).toEqual(['notes.txt', 'x_scene_detection.json']);
  });

  it('classifies as the exporter does', () => {
    expect(classify('a_person_tracks.json')).toBeNull();
    expect(classify('a.mp4')).toBeNull();
    expect(classify('a_speaker_diarization.rttm.provenance.json')).toBeNull();
    expect(classify('scene_detection.json')).toBeNull();
    expect(classify('a_scene_detection.json')?.stem).toBe('a');
    expect(classify('a_unknown.json')).toEqual({ kind: null, stem: '' });
  });
});
