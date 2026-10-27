import type React from 'react';
import { Crosshair, ListChecks, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { VlmModelPicker } from '@/components/VlmModelPicker';
import type { ModelServer } from '@/types/ask';
import { Examples } from './Examples';
import { ServerPicker } from './ServerPicker';

interface QuestionPanelProps {
  question: string;
  onQuestion: (text: string) => void;
  model: string;
  onModel: (model: string) => void;
  servers: ModelServer[];
  server: ModelServer | undefined;
  onServer: (id: string) => void;
  /** Under the question: the answer shape editor (spec 033). */
  shapeEditor?: React.ReactNode;
  /** Where the chosen model runs, beside the picker (UAT #25). */
  whereItRuns?: React.ReactNode;
  onTryHere: () => void;
  /** The frame "this frame" is: "0:12.4, frame 186" (UAT #21). */
  here: string;
  onTryMoments: () => void;
  tryCount: number;
  /** The moments the try picks from, in words: "every 5 s, 0:00 to the end" (UAT #22). */
  momentsFrom: string;
  busy: boolean;
  disabled: boolean;
}

export function QuestionPanel(props: QuestionPanelProps) {
  const { question, model, busy, disabled } = props;
  const blocked = disabled || busy || !question.trim() || !model;
  return (
    <div className="space-y-3">
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <Label htmlFor="ask-question">Question</Label>
          <Examples current={question} onInsert={props.onQuestion} />
        </div>
        <Textarea
          id="ask-question"
          rows={6}
          placeholder="What do you want to know about each moment? Say what form the answer should take."
          value={question}
          onChange={(e) => props.onQuestion(e.target.value)}
        />
      </div>
      {props.shapeEditor}
      <ServerPicker servers={props.servers} value={props.server?.id ?? 'default'} onChange={props.onServer} fieldId="ask-server" />
      <VlmModelPicker
        value={model}
        onChange={props.onModel}
        fieldId="ask-model"
        server={props.server?.id}
        serverName={props.server?.name}
      />
      {props.whereItRuns}
      <div className="space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={props.onTryHere} disabled={blocked} className="gap-1" data-testid="try-here">
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Crosshair className="h-4 w-4" />}
            Try on this frame
          </Button>
          <span className="text-xs text-muted-foreground tabular-nums">{props.here}</span>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button variant="outline" onClick={props.onTryMoments} disabled={blocked || props.tryCount === 0} className="gap-1" data-testid="try-moments">
            <ListChecks className="h-4 w-4" /> Try on {props.tryCount} moment{props.tryCount === 1 ? '' : 's'}
          </Button>
          <span className="text-xs text-muted-foreground">spread over the moments set below: {props.momentsFrom}</span>
        </div>
      </div>
    </div>
  );
}
