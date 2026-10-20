import { Label } from '@/components/ui/label';
import { Term } from '@/components/Term';
import { useWorkflows } from '@/hooks/useWorkflows';

/** Start New run from a saved workflow (spec 029 FR-014); none means the usual order. */
export function WorkflowPicker({ value, onChange }: { value: string | null; onChange: (id: string | null) => void }) {
  const { data: workflows = [] } = useWorkflows();
  if (workflows.length === 0) return null;
  return (
    <div className="flex flex-wrap items-center gap-2 text-sm">
      <Label htmlFor="workflow-picker">
        <Term word="workflow">Workflow</Term>
      </Label>
      <select
        id="workflow-picker"
        className="h-9 rounded-md border border-input bg-background px-2 text-sm"
        value={value ?? ''}
        onChange={(e) => onChange(e.target.value || null)}
      >
        <option value="">None: the usual settings</option>
        {workflows.map((w) => (
          <option key={w.id} value={w.id}>
            {w.name}
          </option>
        ))}
      </select>
    </div>
  );
}
