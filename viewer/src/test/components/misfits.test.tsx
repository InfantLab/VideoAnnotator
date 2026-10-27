/** A shaped answer that didn't fit is shown as such, never as a value (spec 033, Principle VI). */
import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { AnswerLane } from '@/components/lanes/AnswerLane';
import { AnswersTable } from '@/components/ask/AnswersTable';
import { Tally } from '@/components/ask/Tally';
import type { AskAnswer } from '@/types/ask';

const fits: AskAnswer = {
  t: 0, status: 'answered', label: 'yes', fields: { touch: 'yes', initiator: 'adult' }, fit: true,
  frame_numbers: [0], frame_times: [0], raw_response: '{"touch":"yes","initiator":"adult"}',
};  // prettier-ignore
const misfit: AskAnswer = {
  t: 5, status: 'answered', label: "DIDN'T FIT", fields: {}, fit: false, fit_problem: "The reply isn't JSON.",
  frame_numbers: [125], frame_times: [5], raw_response: 'TOUCH',
};  // prettier-ignore

describe('misfits', () => {
  it('the table shows fields, and a misfit with its problem and the reply', () => {
    render(<AnswersTable columns={[{ key: 'e', title: 'q', subtitle: '', answers: [fits, misfit], frames: 1 }]} highlight={null} onSeek={() => undefined} />);
    expect(screen.getByTestId('answer-fields')).toHaveTextContent('touch yes');
    expect(screen.getByTestId('answer-fields')).toHaveTextContent('initiator adult');
    expect(screen.getByTestId('misfit')).toHaveTextContent("Didn't fit: The reply isn't JSON.");
    expect(screen.getByText('TOUCH')).toBeInTheDocument();
  });

  it('the tally counts misfits apart from the values', () => {
    render(<Tally answers={[fits, misfit, misfit]} />);
    // Field by field (UAT #24).
    expect(screen.getAllByTestId('tally-field').map((f) => f.textContent)).toEqual(['touchyes 1', 'initiatoradult 1']);
    expect(screen.getByTestId('tally-misfits')).toHaveTextContent("didn't fit 2");
  });

  it('the lane draws a misfit as its own mark', () => {
    render(<AnswerLane markers={[fits, misfit]} duration={10} colourOf={() => 'hsl(120, 70%, 50%)'} />);
    const [a, b] = screen.getAllByTestId('answer-marker');
    expect(a).toHaveAttribute('data-status', 'answered');
    expect(b).toHaveAttribute('data-status', 'misfit');
    expect(b).toHaveAttribute('aria-label', "0:05.0: didn't fit the answer shape");
  });
});
