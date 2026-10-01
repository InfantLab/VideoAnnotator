import { describe, expect, it } from 'vitest';

import { apiClient } from '@/api/client';
import { normalizeApiUrl, tokenFormatProblem } from '@/lib/apiConnection';

const KEY = `va_${'a'.repeat(43)}`;

describe('normalizeApiUrl', () => {
  it('keeps this page\'s own origin relative, whatever its host is called', () => {
    // Rewriting it to 127.0.0.1 made a viewer opened at localhost call another
    // origin: blocked by CORS, and without the key saved under this origin.
    expect(normalizeApiUrl(window.location.origin)).toBe('');
    expect(normalizeApiUrl(`${window.location.origin}/`)).toBe('');
  });

  it('sends another server at localhost to 127.0.0.1 (the server binds IPv4)', () => {
    expect(normalizeApiUrl('http://localhost:18011')).toBe('http://127.0.0.1:18011');
  });

  it('leaves other URLs alone apart from trailing slashes', () => {
    expect(normalizeApiUrl(' http://lab-server:18011/ ')).toBe('http://lab-server:18011');
    expect(normalizeApiUrl('')).toBe('');
  });
});

describe('tokenFormatProblem', () => {
  it('accepts a key, a JWT, or nothing', () => {
    expect(tokenFormatProblem(KEY)).toBeNull();
    expect(tokenFormatProblem(` ${KEY} `)).toBeNull();
    expect(tokenFormatProblem('eyJhbGciOiJIUzI1NiJ9.e30.sig')).toBeNull();
    expect(tokenFormatProblem('')).toBeNull();
  });

  it('catches what people paste by mistake', () => {
    expect(tokenFormatProblem(`Bearer ${KEY}`)).toMatch(/without "Bearer "/);
    expect(tokenFormatProblem('dev-token')).toMatch(/no longer accepts/);
    expect(tokenFormatProblem(KEY.slice(0, 20))).toMatch(/this one has 17/);
    expect(tokenFormatProblem('hunter2')).toMatch(/starts with "va_"/);
  });
});

describe('apiClient.updateConfig', () => {
  it('treats an empty token as "no token", not "keep the old one"', () => {
    // Settings tests "anonymous" by passing ''; it used to test the saved key.
    const { baseURL, token } = apiClient.getConfig();
    try {
      apiClient.updateConfig(undefined, KEY);
      apiClient.updateConfig(undefined, '');
      expect(apiClient.getConfig().token).toBe('');
      apiClient.updateConfig('', undefined);
      expect(apiClient.getConfig().baseURL).toBe('');
    } finally {
      apiClient.updateConfig(baseURL, token);
    }
  });
});
