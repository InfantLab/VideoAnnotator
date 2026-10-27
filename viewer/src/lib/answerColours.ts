/**
 * Ask AI answers' colours (UAT #23, #50; Caspar 2026-10-27): every answer is
 * coloured, the same answer the same colour across every try and run of one
 * question, remembered in this browser. 16 distinct colours, none red (red
 * means failed); past 16 answers, each new one takes a shade of the colour of
 * the answer it reads most like, so similar answers look alike.
 */
import type { AskAnswer } from '@/types/ask';

/** Eight hues clear of red, each at two lightnesses: 16 colours. */
const HUES = [210, 35, 150, 280, 55, 185, 320, 100];
export const PALETTE_SIZE = HUES.length * 2;

const paletteColour = (slot: number) => `hsl(${HUES[slot % HUES.length]}, 70%, ${slot < HUES.length ? 50 : 34}%)`;
const shadeColour = (slot: number, shade: number) =>
  `hsl(${HUES[slot % HUES.length]}, ${slot < HUES.length ? 55 : 45}%, ${[66, 76, 24, 58][shade % 4]}%)`;

/** The value an answer is coloured by, as the model wrote it. */
export function answerText(answer: Pick<AskAnswer, 'label' | 'fields'>): string {
  const fields = answer.fields ?? readJson(answer.label);
  const value = fields ? Object.values(fields)[0] : answer.label;
  return value === null || value === undefined ? '' : String(value);
}

/** What an answer is coloured by: its first field's value, or its text, compared loosely. */
export function answerKey(answer: Pick<AskAnswer, 'label' | 'fields'>): string {
  const fields = answer.fields ?? readJson(answer.label);
  const value = fields ? Object.values(fields)[0] : answer.label;
  return normalise(value === null || value === undefined ? '' : String(value));
}

const normalise = (text: string) =>
  text
    .trim()
    .toLowerCase()
    .replace(/[\s_]+/g, ' ')
    .replace(/[.!]+$/, '');

/**
 * A reply that is JSON, inside ```json fences or not, read as fields for showing
 * (UAT #27, D4); the stored reply is left as it was. null when it isn't an object.
 */
export function readJson(text: string | null | undefined): Record<string, unknown> | null {
  if (!text) return null;
  const unfenced = text.trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '');
  if (!unfenced.startsWith('{')) return null;
  try {
    const value = JSON.parse(unfenced);
    return value && typeof value === 'object' && !Array.isArray(value) ? value : null;
  } catch {
    return null;
  }
}

function trigrams(text: string): Map<string, number> {
  const padded = `  ${text} `;
  const counts = new Map<string, number>();
  for (let i = 0; i + 3 <= padded.length; i++) {
    const g = padded.slice(i, i + 3);
    counts.set(g, (counts.get(g) ?? 0) + 1);
  }
  return counts;
}

/** How alike two answers read, 0 to 1: the cosine of their letter-triple counts. */
export function similarity(a: string, b: string): number {
  const x = trigrams(a);
  const y = trigrams(b);
  let dot = 0;
  for (const [g, n] of x) dot += n * (y.get(g) ?? 0);
  const norm = (m: Map<string, number>) => Math.sqrt([...m.values()].reduce((s, n) => s + n * n, 0));
  const d = norm(x) * norm(y);
  return d ? dot / d : 0;
}

/** Slot in the palette, or a shade of a slot for answers past the 16th. */
type Assigned = { slot: number; shade?: number };
type Assignments = Record<string, Assigned>;

const STORE_PREFIX = 'vav.ask.colours.';

function storeKey(question: string): string {
  let h = 0;
  for (let i = 0; i < question.length; i++) h = (Math.imul(31, h) + question.charCodeAt(i)) | 0;
  return `${STORE_PREFIX}${(h >>> 0).toString(36)}`;
}

function load(question: string): Assignments {
  try {
    return JSON.parse(localStorage.getItem(storeKey(question)) ?? '{}') as Assignments;
  } catch {
    return {};
  }
}

function save(question: string, assigned: Assignments) {
  try {
    localStorage.setItem(storeKey(question), JSON.stringify(assigned));
  } catch {
    /* private window: consistent for this visit only */
  }
}

/** Give each new key a colour: free palette slots first (most common first), then shades of the nearest. */
export function assign(assigned: Assignments, keys: string[]): Assignments {
  const counts = new Map<string, number>();
  for (const k of keys) counts.set(k, (counts.get(k) ?? 0) + 1);
  const fresh = [...counts.keys()].filter((k) => !(k in assigned)).sort((a, b) => counts.get(b)! - counts.get(a)! || a.localeCompare(b));
  const next: Assignments = { ...assigned };
  for (const key of fresh) {
    const used = new Set(Object.values(next).filter((a) => a.shade === undefined).map((a) => a.slot));
    const free = [...Array(PALETTE_SIZE).keys()].find((s) => !used.has(s));
    if (free !== undefined) {
      next[key] = { slot: free };
      continue;
    }
    let best = { key: '', score: -1 };
    for (const [other, a] of Object.entries(next)) {
      if (a.shade !== undefined) continue;
      const score = similarity(key, other);
      if (score > best.score) best = { key: other, score };
    }
    const slot = next[best.key].slot;
    const shades = Object.values(next).filter((a) => a.slot === slot && a.shade !== undefined).length;
    next[key] = { slot, shade: shades };
  }
  return next;
}

export const colourOf = (a: Assigned | undefined): string =>
  !a ? 'hsl(220, 9%, 60%)' : a.shade === undefined ? paletteColour(a.slot) : shadeColour(a.slot, a.shade);

/** Colours for every answer of `question`'s entries, assigning (and remembering) new ones. */
export function answerColours(question: string, answers: Pick<AskAnswer, 'label' | 'fields' | 'status' | 'fit'>[]) {
  const answered = answers.filter((a) => a.status === 'answered' && a.fit !== false);
  const keys = answered.map(answerKey);
  const shown = new Map<string, string>();
  answered.forEach((a, i) => {
    if (!shown.has(keys[i])) shown.set(keys[i], answerText(a));
  });
  const before = load(question);
  const assigned = assign(before, keys);
  if (Object.keys(assigned).length !== Object.keys(before).length) save(question, assigned);
  return {
    colour: (answer: Pick<AskAnswer, 'label' | 'fields'>) => colourOf(assigned[answerKey(answer)]),
    /** The key under the timeline: each answer shown, with its colour, most common first. */
    legend: [...new Set(keys)]
      .map((key) => ({ key, label: shown.get(key) ?? key, colour: colourOf(assigned[key]), count: keys.filter((k) => k === key).length }))
      .sort((a, b) => b.count - a.count),
  };
}

/** The model's reasoning, unless it only repeats the answer or is the JSON already shown as fields. */
export function reasoningOf(answer: Pick<AskAnswer, 'reasoning' | 'label'>): string {
  const text = answer.reasoning?.trim() ?? '';
  if (!text || text === answer.label?.trim() || readJson(text)) return '';
  return text;
}
