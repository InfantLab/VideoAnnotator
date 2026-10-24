import { describe, expect, it, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { AnswerShapeEditor } from '@/components/ask/AnswerShapeEditor';
import type { AnswerShape, SuggestShapeResponse } from '@/types/ask';

function Harness({ initial = null, onSuggest, spy }: { initial?: AnswerShape | null; onSuggest?: () => Promise<SuggestShapeResponse>; spy?: (s: AnswerShape | null) => void }) {
  const [shape, setShape] = useState<AnswerShape | null>(initial);
  return (
    <AnswerShapeEditor
      value={shape}
      onChange={(s) => {
        setShape(s);
        spy?.(s);
      }}
      onSuggest={onSuggest}
      cantEnforce="lab-gpu"
    />
  );
}

const TWO: AnswerShape = [
  { name: 'touch', kind: 'yes_no' },
  { name: 'initiator', kind: 'choice', values: ['adult', 'child'] },
];

describe('AnswerShapeEditor', () => {
  it('starts as free text and adds fields', async () => {
    const spy = vi.fn();
    render(<Harness spy={spy} />);
    expect(screen.queryByTestId('shape-field')).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Fields' }));
    expect(screen.getAllByTestId('shape-field')).toHaveLength(1);
    expect(spy).toHaveBeenLastCalledWith([{ name: 'answer', kind: 'yes_no' }]);
    await userEvent.click(screen.getByRole('button', { name: /Add a field/ }));
    expect(screen.getAllByTestId('shape-field')).toHaveLength(2);
    expect(screen.getByText("lab-gpu can't enforce the answer shape; answers are checked after.")).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Free text' }));
    expect(spy).toHaveBeenLastCalledWith(null);
  });

  it('says what is wrong, in the server words', async () => {
    render(<Harness initial={[{ name: 'answer', kind: 'yes_no' }]} />);
    const name = screen.getByLabelText('Field 1 name');
    await userEvent.clear(name);
    await userEvent.type(name, 'label');
    expect(screen.getByRole('alert')).toHaveTextContent("Field 1: 'label' is reserved; choose another name.");
    await userEvent.selectOptions(screen.getByLabelText('Field 1 kind'), 'choice');
    await userEvent.clear(name);
    await userEvent.type(name, 'who');
    expect(screen.getByRole('alert')).toHaveTextContent('Field 1 (who): a choice needs 2 to 30 different values.');
    const values = screen.getByLabelText('Field 1 values');
    await userEvent.type(values, 'adult, child{Enter}');
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(within(screen.getByTestId('shape-field')).getByText('adult')).toBeInTheDocument();
  });

  it('reorders and removes fields', async () => {
    const spy = vi.fn();
    render(<Harness initial={TWO} spy={spy} />);
    await userEvent.click(screen.getByRole('button', { name: 'Move field 2 up' }));
    expect(spy).toHaveBeenLastCalledWith([TWO[1], TWO[0]]);
    await userEvent.click(screen.getByRole('button', { name: 'Remove field 1' }));
    expect(spy).toHaveBeenLastCalledWith([TWO[0]]);
  });

  it('never applies a suggestion without the user', async () => {
    const spy = vi.fn();
    const onSuggest = vi.fn().mockResolvedValue({ shape: TWO, made_by: 'model', message: 'Suggested by the model.' });
    render(<Harness onSuggest={onSuggest} spy={spy} />);
    await userEvent.click(screen.getByRole('button', { name: /Suggest/ }));
    expect(await screen.findByText(/Suggested by the model/)).toBeInTheDocument();
    expect(spy).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole('button', { name: 'Use these fields' }));
    expect(spy).toHaveBeenLastCalledWith(TWO);
  });

  it('says when a suggestion came from the question, and can be discarded', async () => {
    const spy = vi.fn();
    const onSuggest = vi.fn().mockResolvedValue({ shape: TWO, made_by: 'rule', message: '' });
    render(<Harness onSuggest={onSuggest} spy={spy} />);
    await userEvent.click(screen.getByRole('button', { name: /Suggest/ }));
    expect(await screen.findByText(/Suggested from your question/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Discard' }));
    expect(spy).not.toHaveBeenCalled();
    expect(screen.queryByTestId('shape-proposal')).not.toBeInTheDocument();
  });
});
