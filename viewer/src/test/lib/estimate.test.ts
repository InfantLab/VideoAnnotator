import { describe, expect, it } from 'vitest';
import { estimate, formatEstimate } from '@/lib/estimate';
import type { AskAnswer, AskEntry } from '@/types/ask';

const answer = (t: number, total: number, frames = 1, status: AskAnswer['status'] = 'answered'): AskAnswer => ({
  t,
  status,
  frame_numbers: Array.from({ length: frames }, (_, i) => i),
  frame_times: [],
  total_time: total,
});

const entry = (model: string, answers: AskAnswer[], frames = 1): AskEntry =>
  ({ model, answers, sampling: { interval: 5, frames } }) as unknown as AskEntry;

describe('estimate', () => {
  it('is a rough guess before any answer', () => {
    const e = estimate(40, 1, 'qwen3.5:9b', []);
    expect(e).toEqual({ seconds: 240, rough: true, overHour: false });
  });

  it("uses the median of this model's answers, per frame", () => {
    const history = [entry('qwen3.5:9b', [answer(0, 2), answer(5, 4), answer(10, 100)]), entry('other', [answer(0, 50)])];
    const e = estimate(10, 3, 'qwen3.5:9b', history);
    expect(e.rough).toBe(false);
    expect(e.seconds).toBe(10 * 3 * 4);
  });

  it('scales by the frames those answers were sent', () => {
    const history = [entry('m', [answer(0, 6, 3), answer(5, 6, 3)], 3)];
    expect(estimate(1, 1, 'm', history).seconds).toBe(2);
  });

  it('leaves errors out', () => {
    const history = [entry('m', [answer(0, 99, 1, 'error'), answer(5, 1)])];
    expect(estimate(1, 1, 'm', history).seconds).toBe(1);
  });

  it('warns past an hour', () => {
    expect(estimate(1000, 1, 'm', []).overHour).toBe(true);
  });

  it('reads like a person would say it', () => {
    expect(formatEstimate(40)).toBe('about 40 s');
    expect(formatEstimate(180)).toBe('about 3 min');
    expect(formatEstimate(4800)).toBe('about 1 h 20 min');
  });
});
