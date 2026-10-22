import type React from 'react';
import { Crosshair, Loader2, Shuffle } from 'lucide-react';
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
  onTryHere: () => void;
  onTryMoments: () => void;
  tryCount: number;
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
      <div className="flex flex-wrap gap-2">
        <Button onClick={props.onTryHere} disabled={blocked} className="gap-1">
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Crosshair className="h-4 w-4" />}
          Try on this frame
        </Button>
        <Button variant="outline" onClick={props.onTryMoments} disabled={blocked || props.tryCount === 0} className="gap-1">
          <Shuffle className="h-4 w-4" /> Try on {props.tryCount} moment{props.tryCount === 1 ? '' : 's'}
        </Button>
      </div>
    </div>
  );
}
