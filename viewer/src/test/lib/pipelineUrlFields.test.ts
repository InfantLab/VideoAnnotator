import { describe, it, expect } from 'vitest';
import { findUrlFieldErrors, urlValueError } from '@/lib/pipelineUrlFields';
import type { PipelineDescriptor } from '@/types/pipelines';

describe('urlValueError', () => {
  it('accepts empty and http(s) URLs', () => {
    expect(urlValueError('')).toBeNull();
    expect(urlValueError(undefined)).toBeNull();
    expect(urlValueError('http://localhost:11434')).toBeNull();
    expect(urlValueError('https://host.docker.internal:11434/')).toBeNull();
  });

  it('rejects a prompt pasted into the URL field', () => {
    expect(
      urlValueError('Count the number of people in the scene. Return JSON count: 2, child: True')
    ).not.toBeNull();
    expect(urlValueError('localhost:11434')).not.toBeNull();
    expect(urlValueError('ftp://example.com')).not.toBeNull();
  });
});

describe('findUrlFieldErrors', () => {
  const vlm = {
    id: 'vlm_annotation',
    name: 'VLM',
    parameters: [
      { name: 'prompt', type: 'text' },
      { name: 'base_url', type: 'string' },
    ],
  } as unknown as PipelineDescriptor;

  it('flags only selected pipelines with a bad URL', () => {
    const config = { vlm_annotation: { prompt: 'hi there', base_url: 'hi there' } };
    expect(findUrlFieldErrors([vlm], ['vlm_annotation'], config)).toEqual([
      'vlm_annotation.base_url',
    ]);
    expect(findUrlFieldErrors([vlm], [], config)).toEqual([]);
    expect(
      findUrlFieldErrors([vlm], ['vlm_annotation'], { vlm_annotation: { prompt: 'x y' } })
    ).toEqual([]);
  });
});
