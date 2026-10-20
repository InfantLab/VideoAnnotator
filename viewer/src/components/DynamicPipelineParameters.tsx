import { useMemo, type ReactNode } from 'react';
import { Switch } from '@/components/ui/switch';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select';
import { Checkbox } from '@/components/ui/checkbox';
import { Textarea } from '@/components/ui/textarea';
import { VlmModelPicker } from '@/components/VlmModelPicker';
import { OpenInAskAI } from '@/components/ask/OpenInAskAI';
import { isUrlParameter, urlValueError } from '@/lib/pipelineUrlFields';
import { pipelineSummary } from '@/lib/pipelineGuide';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { ChevronRight } from 'lucide-react';

import type { PipelineDescriptor, PipelineParameterSchema } from '@/types/pipelines';

interface DynamicPipelineParametersProps {
  pipelines: PipelineDescriptor[];
  selectedPipelineIds: string[];
  config: Record<string, unknown>;
  onConfigChange: (
    updater: (prev: Record<string, unknown>) => Record<string, unknown>
  ) => void;
}

const getPipelineConfig = (
  config: Record<string, unknown>,
  pipelineId: string
): Record<string, unknown> => {
  const value = config[pipelineId];
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return {};
};

const coerceNumber = (value: string, fallback: number | undefined) => {
  if (value === '') {
    return undefined;
  }
  const parsed = Number(value);
  if (Number.isNaN(parsed)) {
    return fallback;
  }
  return parsed;
};

const coerceBoolean = (value: unknown, fallback: boolean | undefined) => {
  if (typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'string') {
    if (value === 'true') return true;
    if (value === 'false') return false;
  }
  return fallback ?? false;
};

const getParameterDefault = (parameter: PipelineParameterSchema) => {
  if (parameter.default !== undefined) {
    return parameter.default;
  }
  if (parameter.type === 'boolean') {
    return false;
  }
  if (parameter.type === 'multiselect') {
    return [];
  }
  return '';
};

const normalizeValue = (parameter: PipelineParameterSchema, value: unknown) => {
  if (value === undefined || value === null) {
    return getParameterDefault(parameter);
  }

  switch (parameter.type) {
    case 'boolean':
      return coerceBoolean(value, !!parameter.default);
    case 'integer':
    case 'number':
      return typeof value === 'number' ? value : Number(value);
    case 'multiselect':
      return Array.isArray(value) ? value : [];
    default:
      return value;
  }
};

/**
 * The same normalized/defaulted value a field's widget currently displays
 * (schema default until the user actually edits it), by name — since raw
 * `pipelineConfig[name]` is `undefined` until then, a consumer outside the
 * parameter-render loop (e.g. the Open in Ask AI link) that read the raw config
 * directly would see an empty value even while the form visibly shows a
 * default like the pipeline's own DEFAULT_PROMPT or default model.
 */
const resolveParameterValue = (
  parameters: PipelineParameterSchema[] | undefined,
  pipelineConfig: Record<string, unknown>,
  name: string
): unknown => {
  const parameter = parameters?.find((p) => p.name === name);
  if (!parameter) return pipelineConfig[name];
  return normalizeValue(parameter, pipelineConfig[name]);
};

/** Whether a setting differs from its default: an untouched field is never changed. */
const isChanged = (parameter: PipelineParameterSchema, pipelineConfig: Record<string, unknown>) => {
  const raw = pipelineConfig[parameter.name];
  if (raw === undefined) return false;
  return JSON.stringify(normalizeValue(parameter, raw)) !== JSON.stringify(getParameterDefault(parameter));
};

/**
 * Spec 026 `applies_when`: hidden when another setting rules it out, but a
 * value the user changed is never hidden, only marked as not used.
 */
const parameterState = (
  parameter: PipelineParameterSchema,
  parameters: PipelineParameterSchema[],
  pipelineConfig: Record<string, unknown>
): 'shown' | 'hidden' | 'not-applicable' => {
  if (!parameter.appliesWhen) return 'shown';
  const applies = Object.entries(parameter.appliesWhen).every(([name, values]) =>
    values.includes(String(resolveParameterValue(parameters, pipelineConfig, name)))
  );
  if (applies) return 'shown';
  return isChanged(parameter, pipelineConfig) ? 'not-applicable' : 'hidden';
};

/** Required, and the allowed range; the unit is shown beside the field instead. */
const renderFieldDescription = (parameter: PipelineParameterSchema) => {
  const meta: string[] = [];
  if (parameter.required) meta.push('required');
  const { min, max } = parameter;
  if (min !== undefined && max !== undefined) meta.push(`${min} to ${max}`);
  else if (min !== undefined) meta.push(`at least ${min}`);
  else if (max !== undefined) meta.push(`at most ${max}`);
  return meta.join(' · ');
};

/**
 * One setting: its name and explanation on the left, the control on the
 * right at the width its value needs. Long text (prompts, JSON, option
 * lists) goes full width underneath instead.
 */
const SettingRow = ({
  label,
  htmlFor,
  description,
  hint,
  wide = false,
  children,
}: {
  label: string;
  htmlFor?: string;
  description?: string;
  hint?: string;
  wide?: boolean;
  children: ReactNode;
}) => (
  <div
    className={
      wide
        ? 'space-y-2 py-3'
        : 'grid items-center gap-x-6 gap-y-2 py-3 sm:grid-cols-[minmax(0,1fr)_auto]'
    }
  >
    <div className="min-w-0 space-y-0.5">
      <Label htmlFor={htmlFor} className="text-sm font-medium">
        {label}
      </Label>
      {description && <p className="text-xs text-muted-foreground">{description}</p>}
      {hint && <p className="text-[11px] text-muted-foreground/80">{hint}</p>}
    </div>
    <div className={wide ? '' : 'flex items-center gap-2 sm:justify-end'}>{children}</div>
  </div>
);

export const DynamicPipelineParameters = ({
  pipelines,
  selectedPipelineIds,
  config,
  onConfigChange
}: DynamicPipelineParametersProps) => {
  const selectedPipelines = useMemo(
    () => pipelines.filter((pipeline) => selectedPipelineIds.includes(pipeline.id)),
    [pipelines, selectedPipelineIds]
  );

  const handleValueChange = (
    pipelineId: string,
    parameter: PipelineParameterSchema,
    value: unknown
  ) => {
    onConfigChange((prevConfig) => {
      const nextConfig: Record<string, unknown> = { ...prevConfig };
      const pipelineConfig = {
        ...getPipelineConfig(prevConfig, pipelineId)
      };

      pipelineConfig[parameter.name] = value;
      nextConfig[pipelineId] = pipelineConfig;
      return nextConfig;
    });
  };

  if (selectedPipelines.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        Select pipelines in the previous step to configure their parameters.
      </p>
    );
  }

  return (
    <div className="space-y-6">
      {selectedPipelines.map((pipeline) => {
        const pipelineConfig = getPipelineConfig(config, pipeline.id);
        const renderParameter = (parameter: PipelineParameterSchema) => {
          const currentValue = normalizeValue(
            parameter,
            pipelineConfig[parameter.name]
          );
          const fieldHint = renderFieldDescription(parameter);

          // vlm_annotation's `model` field: the one deliberate,
          // narrowly-scoped pipeline-specific UI exception (spec
          // 009 / viewer-handoff) — a live-populated picker instead
          // of the generic free-text default. Every other
          // pipeline/field still goes through the switch below.
          if (pipeline.id === 'vlm_annotation' && parameter.name === 'model') {
            return (
              <VlmModelPicker
                key={parameter.name}
                value={typeof currentValue === 'string' ? currentValue : ''}
                onChange={(value) => handleValueChange(pipeline.id, parameter, value)}
                fieldId={`${pipeline.id}-${parameter.name}`}
              />
            );
          }

          const id = `${pipeline.id}-${parameter.name}`;
          const label = parameter.label || parameter.name;
          const rowProps = { label, description: parameter.description, hint: fieldHint || undefined };

          switch (parameter.type) {
            case 'boolean':
              return (
                <SettingRow key={parameter.name} htmlFor={id} {...rowProps}>
                  <Switch
                    id={id}
                    checked={Boolean(currentValue)}
                    onCheckedChange={(checked) => handleValueChange(pipeline.id, parameter, checked)}
                  />
                </SettingRow>
              );

            case 'text':
              return (
                <SettingRow key={parameter.name} htmlFor={id} wide {...rowProps}>
                  <Textarea
                    id={id}
                    value={typeof currentValue === 'string' ? currentValue : ''}
                    rows={6}
                    onChange={(event) => handleValueChange(pipeline.id, parameter, event.target.value)}
                  />
                </SettingRow>
              );

            case 'integer':
            case 'number':
              return (
                <SettingRow key={parameter.name} htmlFor={id} {...rowProps}>
                  <Input
                    id={id}
                    type="number"
                    className="h-9 w-28 text-right tabular-nums"
                    value={typeof currentValue === 'number' || typeof currentValue === 'string' ? currentValue : ''}
                    onChange={(event) => {
                      const newValue = coerceNumber(
                        event.target.value,
                        typeof currentValue === 'number' ? currentValue : undefined
                      );
                      handleValueChange(pipeline.id, parameter, newValue);
                    }}
                    min={parameter.min}
                    max={parameter.max}
                    step={parameter.step}
                  />
                  <span className="w-12 text-xs text-muted-foreground">{parameter.unit ?? ''}</span>
                </SettingRow>
              );

            case 'enum':
              return (
                <SettingRow key={parameter.name} {...rowProps} hint={undefined}>
                  <Select
                    value={currentValue != null ? String(currentValue) : undefined}
                    onValueChange={(value) => {
                      const matched = (parameter.enum ?? []).find((option) => String(option.value) === value);
                      handleValueChange(pipeline.id, parameter, matched ? matched.value : value);
                    }}
                  >
                    <SelectTrigger className="h-9 w-56" aria-label={label}>
                      <SelectValue placeholder="Select option" />
                    </SelectTrigger>
                    <SelectContent>
                      {(parameter.enum ?? []).map((option) => (
                        <SelectItem key={String(option.value)} value={String(option.value)}>
                          {option.label || String(option.value)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </SettingRow>
              );

            case 'multiselect':
              return (
                <SettingRow key={parameter.name} wide {...rowProps} hint={undefined}>
                  <div className="flex flex-wrap gap-x-4 gap-y-2">
                    {(parameter.enum ?? []).map((option) => {
                      const valueArray = Array.isArray(currentValue) ? (currentValue as unknown[]) : [];
                      const checked = valueArray.includes(option.value);
                      return (
                        <label key={String(option.value)} className="flex items-center gap-2 text-sm">
                          <Checkbox
                            checked={checked}
                            onCheckedChange={(checkboxValue) => {
                              const nextValues = new Set(valueArray);
                              if (checkboxValue === true) {
                                nextValues.add(option.value);
                              } else if (checkboxValue === false) {
                                nextValues.delete(option.value);
                              }
                              handleValueChange(pipeline.id, parameter, Array.from(nextValues));
                            }}
                          />
                          <span>{option.label || String(option.value)}</span>
                        </label>
                      );
                    })}
                  </div>
                </SettingRow>
              );

            case 'object':
              return (
                <SettingRow key={parameter.name} wide {...rowProps} hint={undefined}>
                  <Textarea
                    aria-label={label}
                    value={JSON.stringify(currentValue ?? parameter.default ?? {}, null, 2)}
                    className="font-mono text-xs"
                    rows={6}
                    onChange={(event) => {
                      try {
                        const parsed = JSON.parse(event.target.value);
                        handleValueChange(pipeline.id, parameter, parsed);
                      } catch (error) {
                        // noop – keep previous value until valid JSON is provided
                      }
                    }}
                  />
                </SettingRow>
              );

            default: {
              const urlError = isUrlParameter(parameter) ? urlValueError(currentValue) : null;
              return (
                <SettingRow key={parameter.name} htmlFor={id} {...rowProps} hint={undefined}>
                  <div className="w-72 max-w-full space-y-1">
                    <Input
                      id={id}
                      className="h-9"
                      value={String(currentValue ?? '')}
                      onChange={(event) => handleValueChange(pipeline.id, parameter, event.target.value)}
                      {...(isUrlParameter(parameter) && {
                        type: 'url',
                        placeholder: 'Leave empty for the server default',
                        'aria-invalid': urlError !== null
                      })}
                    />
                    {urlError && <p className="text-xs text-destructive">{urlError}</p>}
                  </div>
                </SettingRow>
              );
            }
          }
        };

        const parameters = pipeline.parameters ?? [];
        // Servers before spec 026 send no groups: show everything, as before.
        const grouped = parameters.some((p) => p.group !== undefined);
        const states = new Map(
          parameters.map((p) => [p.name, parameterState(p, parameters, pipelineConfig)] as const)
        );
        const visible = parameters.filter((p) => states.get(p.name) !== 'hidden');
        const behavioural = grouped ? visible.filter((p) => p.group === 'behavioural') : visible;
        const technical = grouped ? visible.filter((p) => p.group !== 'behavioural') : [];
        const changedCount = technical.filter((p) => isChanged(p, pipelineConfig)).length;
        const renderWithNote = (parameter: PipelineParameterSchema) =>
          states.get(parameter.name) === 'not-applicable' ? (
            <div key={parameter.name} className="opacity-70">
              {renderParameter(parameter)}
              <p className="-mt-2 pb-3 text-[11px] text-amber-400">Not used with these settings</p>
            </div>
          ) : (
            renderParameter(parameter)
          );

        return (
          <section key={pipeline.id} className="rounded-lg border bg-card" aria-labelledby={`${pipeline.id}-title`}>
            <header className="flex flex-wrap items-start justify-between gap-x-4 gap-y-1 border-b px-5 py-4">
              <div className="min-w-0">
                <h4 id={`${pipeline.id}-title`} className="text-base font-semibold text-foreground">
                  {pipeline.title ?? pipeline.name}
                </h4>
                <p className="mt-0.5 text-sm text-muted-foreground">{pipelineSummary(pipeline)}</p>
              </div>
              <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-muted-foreground">
                {pipeline.model && <span className="rounded bg-muted px-1.5 py-0.5 font-mono">{pipeline.model}</span>}
                {pipeline.version && <span>v{pipeline.version}</span>}
              </div>
            </header>

            <div className="px-5 pb-3">
            {parameters.length > 0 ? (
              <>
                <div className="divide-y">{behavioural.map(renderWithNote)}</div>
                {technical.length > 0 && (
                  <Collapsible className="border-t pt-3">
                    <CollapsibleTrigger className="flex w-full items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground">
                      <ChevronRight className="h-3 w-3 transition-transform [[data-state=open]_&]:rotate-90" />
                      Technical settings ({technical.length} ·{' '}
                      {changedCount === 0 ? 'all at defaults' : `${changedCount} changed`})
                    </CollapsibleTrigger>
                    <CollapsibleContent className="divide-y">{technical.map(renderWithNote)}</CollapsibleContent>
                  </Collapsible>
                )}
              </>
            ) : (
              <p className="py-3 text-xs text-muted-foreground">No settings to change for this pipeline.</p>
            )}

            {pipeline.id === 'vlm_annotation' && (
              <OpenInAskAI
                config={Object.fromEntries(
                  (pipeline.parameters ?? []).map((p) => [
                    p.name,
                    resolveParameterValue(pipeline.parameters, pipelineConfig, p.name),
                  ])
                )}
              />
            )}
            </div>
          </section>
        );
      })}
    </div>
  );
};

export type { DynamicPipelineParametersProps };
