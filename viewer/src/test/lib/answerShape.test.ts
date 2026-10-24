import fs from 'fs';
import path from 'path';
import { describe, expect, it } from 'vitest';
import { cleanShape, shapeProblem, suggestByRule } from '@/lib/answerShape';
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
