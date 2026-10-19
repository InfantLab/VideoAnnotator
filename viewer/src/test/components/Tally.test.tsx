import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Tally } from '@/components/ask/Tally';
import type { AskAnswer } from '@/types/ask';

const a = (label: string | null, status: AskAnswer['status'] = 'answered'): AskAnswer => ({
  t: 0,
  label,
  status,
  frame_numbers: [],
  frame_times: [],
});

describe('Tally', () => {
  it('counts each answer, and errors and skipped apart', () => {
    render(<Tally answers={[a('TOUCH'), a('TOUCH'), a('NO_TOUCH'), a(null, 'error'), a(null, 'skipped')]} />);
    expect(screen.getAllByTestId('tally-item').map((i) => i.textContent)).toEqual(['TOUCH 2', 'NO_TOUCH 1']);
    expect(screen.getByTestId('tally-errors')).toHaveTextContent('no answer 1');
    expect(screen.getByTestId('tally-skipped')).toHaveTextContent('skipped 1');
    expect(screen.queryByTestId('tally-hint')).not.toBeInTheDocument();
  });

  it('suggests saying which answers are allowed past ten different ones', () => {
    const { rerender } = render(<Tally answers={Array.from({ length: 10 }, (_, i) => a(`answer ${i}`))} />);
    expect(screen.queryByTestId('tally-hint')).not.toBeInTheDocument();
    rerender(<Tally answers={Array.from({ length: 11 }, (_, i) => a(`answer ${i}`))} />);
    expect(screen.getByTestId('tally-hint')).toHaveTextContent('Say in the question which answers are allowed');
  });

  it("doesn't count errors as different answers", () => {
    render(<Tally answers={[a('X'), ...Array.from({ length: 12 }, () => a(null, 'error'))]} />);
    expect(screen.queryByTestId('tally-hint')).not.toBeInTheDocument();
  });
});
