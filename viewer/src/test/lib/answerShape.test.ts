import fs from 'fs';
import path from 'path';
import { describe, expect, it } from 'vitest';
import { cleanShape, shapeProblem, shapeSentence, suggestByRule, withShapeSentence } from '@/lib/answerShape';
import type { AnswerShape } from '@/types/ask';

interface Cases {
  shapes: {
    valid: Array<{ name: string; shape: AnswerShape }>;
    invalid: Array<{ name: string; shape: unknown; problem: string }>;
  };
  suggest: Array<{ name: string; prompt: string; expect: AnswerShape | null }>;
}

const CASES: Cases = JSON.parse(
  fs.readFileSync(path.resolve(__dirname, '../../../../tests/fixtures/answer_shape/cases.json'), 'utf-8'),
);

describe('answer shape twin of the pipeline', () => {
  it.each(CASES.shapes.valid.map((c) => [c.name, c] as const))('valid: %s', (_, c) => {
    expect(shapeProblem(c.shape)).toBeNull();
    expect(cleanShape(c.shape)).toEqual(c.shape);
  });

  it.each(CASES.shapes.invalid.map((c) => [c.name, c] as const))('invalid: %s', (_, c) => {
    expect(shapeProblem(c.shape)).toBe(c.problem);
  });

  it.each(CASES.suggest.map((c) => [c.name, c] as const))('suggests: %s', (_, c) => {
    expect(suggestByRule(c.prompt)).toEqual(c.expect);
  });
});

describe('a question of several parts (UAT #57)', () => {
  it('gets a field per part, named in its words', () => {
    expect(suggestByRule('How many adults are in the room, and is the child smiling?')).toEqual([
      { name: 'Adults', kind: 'number' },
      { name: 'Child smiling', kind: 'yes_no' },
    ]);
    expect(suggestByRule('What is the mother doing with her hands?')).toBeNull();
  });
});

describe('the sentence Fields writes into the question (D5)', () => {
  const shape = [
    { name: 'Adults', kind: 'number' as const },
    { name: 'Child smiling', kind: 'yes_no' as const },
  ];
  it('says each field in words', () => {
    expect(shapeSentence(shape)).toBe('Answer as JSON with "Adults": a number, "Child smiling": "yes" or "no".');
  });
  it('is added, replaced as the fields change, and removed for free text', () => {
    const q = 'How many adults are in the room, and is the child smiling?';
    const added = withShapeSentence(q, null, shape);
    expect(added).toBe(`${q}\n\n${shapeSentence(shape)}`);
    const changed = withShapeSentence(added, shape, [shape[0]]);
    expect(changed).toBe(`${q}\n\nAnswer as JSON with "Adults": a number.`);
    expect(withShapeSentence(changed, [shape[0]], null)).toBe(q);
  });
  it('leaves a sentence the researcher edited alone, adding the new one', () => {
    const edited = 'Count them. Answer as JSON please.';
    expect(withShapeSentence(edited, shape, [shape[0]])).toBe(`${edited}\n\nAnswer as JSON with "Adults": a number.`);
  });
});
