import { useState } from 'react';
import { Lightbulb } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { EXAMPLE_QUESTIONS, type ExampleQuestion } from '@/lib/askExamples';

interface ExamplesProps {
  current: string;
  onInsert: (text: string) => void;
}

/** Example questions; replacing a question you've written asks first. */
export function Examples({ current, onInsert }: ExamplesProps) {
  const [open, setOpen] = useState(false);
  const [pending, setPending] = useState<ExampleQuestion | null>(null);

  const choose = (example: ExampleQuestion) => {
    setOpen(false);
    if (current.trim() && current.trim() !== example.text) setPending(example);
    else onInsert(example.text);
  };

  return (
    <>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button variant="outline" size="sm" className="gap-1">
            <Lightbulb className="h-4 w-4" /> Examples
          </Button>
        </PopoverTrigger>
        <PopoverContent align="start" className="w-80 p-1">
          <ul>
            {EXAMPLE_QUESTIONS.map((example) => (
              <li key={example.title}>
                <button
                  type="button"
                  className="w-full rounded px-2 py-1.5 text-left text-sm hover:bg-muted"
                  onClick={() => choose(example)}
                >
                  <span className="font-medium">{example.title}</span>
                  <span className="block truncate text-xs text-muted-foreground">{example.text.split('\n').pop()}</span>
                </button>
              </li>
            ))}
          </ul>
        </PopoverContent>
      </Popover>
      <AlertDialog open={pending !== null} onOpenChange={(o) => !o && setPending(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Replace your question?</AlertDialogTitle>
            <AlertDialogDescription>
              The example “{pending?.title}” will replace what you've written. Your earlier tries stay in the history.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Keep mine</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (pending) onInsert(pending.text);
                setPending(null);
              }}
            >
              Replace
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
