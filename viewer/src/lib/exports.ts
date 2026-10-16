// Results out (spec 025): turning the server's export responses into text.

import type { ExportSummary, MethodsDescription } from '@/api/client';

/** What the export left out, in sentences; empty when it covered everything. */
export function summaryLines(summary: ExportSummary | null): string[] {
  if (!summary) return [];
  const lines = [
    ...summary.missing.map((m) => `No results for ${m.video}: ${m.reason}`),
    ...summary.skipped.map((s) => `Skipped ${s.file} (${s.video}): ${s.reason}`),
    ...summary.notes,
    ...summary.left_out,
  ];
  if (summary.truncated) lines.push('…and more (the full list is in the CLI export).');
  return lines;
}

/** The methods as one Markdown file, laid out as `videoannotator export methods` writes it. */
export function methodsMarkdown(methods: MethodsDescription): string {
  const parts = ['# Methods', '', methods.prose, ''];
  if (methods.revisions.length > 0) {
    parts.push('## Model revisions', '', ...methods.revisions.map((r) => `- ${r}`), '');
  }
  parts.push('## References (BibTeX)', '', '```bibtex', methods.bibtex, '```', '');
  for (const prompt of methods.prompts) {
    parts.push(`## Prompt ${prompt.sha256.slice(0, 12)}`, '', `sha256 \`${prompt.sha256}\``, '', '```text', prompt.text, '```', '');
  }
  return parts.join('\n');
}
