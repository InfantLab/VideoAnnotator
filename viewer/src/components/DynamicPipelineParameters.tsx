import { useMemo } from 'react';
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

const renderFieldDescription = (parameter: PipelineParameterSchema) => {
  const meta: string[] = [];
  if (parameter.required) meta.push('required');
  if (parameter.min !== undefined) meta.push(`min ${parameter.min}`);
  if (parameter.max !== undefined) meta.push(`max ${parameter.max}`);
  if (parameter.step !== undefined) meta.push(`step ${parameter.step}`);
  if (parameter.unit) meta.push(parameter.unit);
  return meta.join(' · ');
};

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

        return (
          <div key={pipeline.id} className="rounded-lg border bg-card p-4 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <h4 className="text-sm font-semibold text-foreground">{pipeline.name}</h4>
                {pipeline.description && (
                  <p className="text-xs text-muted-foreground">{pipeline.description}</p>
                )}
              </div>
              <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
                {pipeline.version && <span>v{pipeline.version}</span>}
                {pipeline.model && <span>{pipeline.model}</span>}
              </div>
            </div>

            {pipeline.parameters && pipeline.parameters.length > 0 ? (
              <div className="space-y-3">
                {pipeline.parameters.map((parameter) => {
                  const currentValue = normalizeValue(
                    parameter,
                    pipelineConfig[parameter.name]
                  );
                  const fieldHint = renderFieldDescription(parameter);

                  switch (parameter.type) {
                    case 'boolean':
                      return (
                        <div
                          key={parameter.name}
                          className="flex items-center justify-between rounded border bg-muted/40 px-3 py-2"
                        >
                          <div className="space-y-1">
                            <Label htmlFor={`${pipeline.id}-${parameter.name}`} className="text-sm">
                              {parameter.label || parameter.name}
                            </Label>
                            {parameter.description && (
                              <p className="text-xs text-muted-foreground">
                                {parameter.description}
                              </p>
                            )}
                            {fieldHint && (
                              <p className="text-[11px] text-muted-foreground/80">{fieldHint}</p>
                            )}
                          </div>
                          <Switch
                            id={`${pipeline.id}-${parameter.name}`}
                            checked={Boolean(currentValue)}
                            onCheckedChange={(checked) =>
                              handleValueChange(pipeline.id, parameter, checked)
                            }
                          />
                        </div>
                      );

                    case 'integer':
                    case 'number':
                      return (
                        <div key={parameter.name} className="space-y-1">
                          <Label htmlFor={`${pipeline.id}-${parameter.name}`} className="text-sm">
                            {parameter.label || parameter.name}
                          </Label>
                          <Input
                            id={`${pipeline.id}-${parameter.name}`}
                            type="number"
                            value={currentValue ?? ''}
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
                          {parameter.description && (
                            <p className="text-xs text-muted-foreground">
                              {parameter.description}
                            </p>
                          )}
                          {fieldHint && (
                            <p className="text-[11px] text-muted-foreground/80">{fieldHint}</p>
                          )}
                        </div>
                      );

                    case 'enum':
                      return (
                        <div key={parameter.name} className="space-y-1">
                          <Label className="text-sm">{parameter.label || parameter.name}</Label>
                          <Select
                            value={currentValue != null ? String(currentValue) : undefined}
                            onValueChange={(value) => {
                              const matched = (parameter.enum ?? []).find(
                                (option) => String(option.value) === value
                              );
                              handleValueChange(
                                pipeline.id,
                                parameter,
                                matched ? matched.value : value
                              );
                            }}
                          >
                            <SelectTrigger>
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
                          {parameter.description && (
                            <p className="text-xs text-muted-foreground">
                              {parameter.description}
                            </p>
                          )}
                        </div>
                      );

                    case 'multiselect':
                      return (
                        <div key={parameter.name} className="space-y-1">
                          <Label className="text-sm">{parameter.label || parameter.name}</Label>
                          <div className="flex flex-wrap gap-3 rounded border bg-muted/40 p-3">
                            {(parameter.enum ?? []).map((option) => {
                              const valueArray = Array.isArray(currentValue)
                                ? (currentValue as unknown[])
                                : [];
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
                          {parameter.description && (
                            <p className="text-xs text-muted-foreground">
                              {parameter.description}
                            </p>
                          )}
                        </div>
                      );

                    case 'object':
                      return (
                        <div key={parameter.name} className="space-y-1">
                          <Label className="text-sm">{parameter.label || parameter.name}</Label>
                          <Textarea
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
                          {parameter.description && (
                            <p className="text-xs text-muted-foreground">
                              {parameter.description}
                            </p>
                          )}
                        </div>
                      );

                    default:
                      return (
                        <div key={parameter.name} className="space-y-1">
                          <Label className="text-sm">{parameter.label || parameter.name}</Label>
                          <Input
                            value={String(currentValue ?? '')}
                            onChange={(event) =>
                              handleValueChange(pipeline.id, parameter, event.target.value)
                            }
                          />
                          {parameter.description && (
                            <p className="text-xs text-muted-foreground">
                              {parameter.description}
                            </p>
                          )}
                        </div>
                      );
                  }
                })}
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">No configurable parameters for this pipeline.</p>
            )}
          </div>
        );
      })}
    </div>
  );
};

export type { DynamicPipelineParametersProps };
