/**
 * Answer shapes in the viewer: the twin of validate_shape and suggest_by_rule
 * in src/videoannotator/pipelines/vlm_annotation/answer_shape.py (spec 033).
 * Both are pinned by tests/fixtures/answer_shape/cases.json. Replies are
 * checked only on the server.
 */
import type { AnswerShape, AnswerShapeField } from '@/types/ask';

export const KINDS = ['choice', 'yes_no', 'number', 'text'] as const;
export const KIND_WORDS: Record<AnswerShapeField['kind'], string> = {
  choice: 'One of a list',
  yes_no: 'Yes or no',
  number: 'A number',
  text: 'Short text',
};
export const DIDNT_FIT = "DIDN'T FIT";

const RESERVED = new Set(['label', 'reasoning', 'fit', 'fit_problem', 't']);
const MAX_FIELDS = 12;
const MAX_VALUES = 30;
const MAX_VALUE_LENGTH = 60;
const NAME = /^[a-z][a-z0-9_]{0,39}$/;

const isNumber = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

/** The first problem with a shape, in the server's words, or null. */
export function shapeProblem(raw: unknown): string | null {
  if (!Array.isArray(raw)) return 'The answer shape must be a list of fields.';
  if (raw.length === 0) return 'Add at least one field.';
  if (raw.length > MAX_FIELDS) return `An answer shape has at most ${MAX_FIELDS} fields.`;
  const seen = new Set<string>();
  for (const [index, item] of raw.entries()) {
    const i = index + 1;
    if (!item || typeof item !== 'object' || Array.isArray(item)) return `Field ${i}: not a field.`;
    const field = item as Record<string, unknown>;
    const name = field.name;
    if (typeof name !== 'string' || !NAME.test(name)) {
      return `Field ${i}: '${String(name)}' isn't a usable name: start with a lower-case letter, then lower-case letters, digits or _ (at most 40).`;
    }
    if (RESERVED.has(name)) return `Field ${i}: '${name}' is reserved; choose another name.`;
    if (seen.has(name)) return `Field ${i}: '${name}' is used twice.`;
    seen.add(name);
    const where = `Field ${i} (${name})`;
    const kind = field.kind;
    if (!KINDS.includes(kind as AnswerShapeField['kind'])) {
      return `${where}: the kind must be choice, yes_no, number or text.`;
    }
    if (kind === 'choice') {
      const values = field.values;
      if (!Array.isArray(values) || !values.every((v) => typeof v === 'string')) {
        return `${where}: a choice needs 2 to ${MAX_VALUES} different values.`;
      }
      const trimmed = (values as string[]).map((v) => v.trim());
      if (trimmed.some((v) => v.length < 1 || v.length > MAX_VALUE_LENGTH)) {
        return `${where}: each value must be 1 to ${MAX_VALUE_LENGTH} characters.`;
      }
      const distinct = new Set(trimmed).size;
      if (distinct !== trimmed.length || distinct < 2 || distinct > MAX_VALUES) {
        return `${where}: a choice needs 2 to ${MAX_VALUES} different values.`;
      }
    } else if (kind === 'number') {
      const { min, max } = field;
      if ((min != null && !isNumber(min)) || (max != null && !isNumber(max))) {
        return `${where}: the lowest and highest must be numbers.`;
      }
      if (isNumber(min) && isNumber(max) && min > max) return `${where}: the lowest number is more than the highest.`;
    }
  }
  return null;
}

/** The shape as the server stores it: only the keys each kind uses. */
export function cleanShape(shape: AnswerShape): AnswerShape {
  return shape.map((f) => {
    const out: AnswerShapeField = { name: f.name, kind: f.kind };
    if (f.kind === 'choice') out.values = (f.values ?? []).map((v) => v.trim());
    if (f.kind === 'number') {
      if (isNumber(f.min)) out.min = f.min;
      if (isNumber(f.max)) out.max = f.max;
    }
    return out;
  });
}

// ---------------------------------------------------------------------------
// Suggesting a shape from the question's text, by rule (research R11)

const TRIGGER =
  String.raw`\b(?:return\s+only|(?:answer|reply|respond)\s+with(?:\s+only)?(?:\s+one\s+(?:word|of))?|choose\s+(?:one\s+)?(?:of|from)|one\s+of)\s*:?\s*`;
const LIST_AFTER_TRIGGER = new RegExp(TRIGGER + String.raw`([^\n.?!;()]+)`, 'gi');
const PARENTHESISED = /\(([^()\n]+)\)/g;
const SCALE = /\bscale\s+(?:of|from)\s+(-?\d+(?:\.\d+)?)\s*(?:to|-)\s*(-?\d+(?:\.\d+)?)/gi;
const YES_OR_NO = /\byes\s+or\s+no\b/gi;
const A_NUMBER = /\b(?:answer|reply|respond)\s+with\s+(?:a|one)\s+(?:whole\s+)?number\b/gi;
const ITEM = /^[A-Za-z0-9][A-Za-z0-9_-]*$/;
const SPLIT = /\s*(?:,|\/|\bor\b)\s*/i;

type Spec = Omit<AnswerShapeField, 'name'>;

function items(text: string): string[] | null {
  const parts = text
    .split(new RegExp(SPLIT.source, 'gi'))
    .map((s) => s.trim())
    .filter(Boolean);
  if (parts.length < 2 || parts.length > MAX_VALUES || new Set(parts).size !== parts.length) return null;
  return parts.every((p) => ITEM.test(p)) ? parts : null;
}

function listField(values: string[]): Spec {
  const lower = new Set(values.map((v) => v.toLowerCase()));
  if (values.length === 2 && lower.has('yes') && lower.has('no')) return { kind: 'yes_no' };
  return { kind: 'choice', values };
}

const parseNumber = (text: string): number => Number(text);

/** Fields for the lists of options the question itself gives, or null. */
export function suggestByRule(prompt: string): AnswerShape | null {
  const found: Array<{ start: number; end: number; rank: number; spec: Spec }> = [];
  for (const m of prompt.matchAll(LIST_AFTER_TRIGGER)) {
    const values = items(m[1]);
    if (values) found.push({ start: m.index!, end: m.index! + m[0].length, rank: 0, spec: listField(values) });
  }
  for (const m of prompt.matchAll(PARENTHESISED)) {
    if (!m[1].includes('/') && !/\bor\b/i.test(m[1])) continue;
    const values = items(m[1]);
    if (values) found.push({ start: m.index!, end: m.index! + m[0].length, rank: 1, spec: listField(values) });
  }
  for (const m of prompt.matchAll(SCALE)) {
    const lo = parseNumber(m[1]);
    const hi = parseNumber(m[2]);
    if (lo <= hi) found.push({ start: m.index!, end: m.index! + m[0].length, rank: 2, spec: { kind: 'number', min: lo, max: hi } });
  }
  for (const m of prompt.matchAll(YES_OR_NO)) {
    found.push({ start: m.index!, end: m.index! + m[0].length, rank: 3, spec: { kind: 'yes_no' } });
  }
  for (const m of prompt.matchAll(A_NUMBER)) {
    found.push({ start: m.index!, end: m.index! + m[0].length, rank: 4, spec: { kind: 'number' } });
  }
  found.sort((a, b) => a.start - b.start || a.rank - b.rank);
  const kept: Spec[] = [];
  let end = -1;
  for (const f of found) {
    if (f.start < end) continue;
    kept.push(f.spec);
    end = f.end;
  }
  if (!kept.length) return null;
  return kept.slice(0, MAX_FIELDS).map((spec, i) => ({ name: i === 0 ? 'answer' : `answer_${i + 1}`, ...spec }));
}
