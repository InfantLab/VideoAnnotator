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

/** Hand `blob` to the browser as a download called `name`. */
export function saveBlob(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

/** Several files as one zip (Ask AI's tables, one per shown entry). */
export async function zipBlobs(files: Array<{ name: string; blob: Blob }>): Promise<Blob> {
  const zip = await import('@zip.js/zip.js');
  const writer = new zip.ZipWriter(new zip.BlobWriter('application/zip'));
  const used = new Set<string>();
  for (const { name, blob } of files) {
    let unique = name;
    for (let n = 2; used.has(unique); n++) unique = name.replace(/(\.[^.]+)?$/, ` ${n}$1`);
    used.add(unique);
    await writer.add(unique, new zip.BlobReader(blob));
  }
  return writer.close();
}
