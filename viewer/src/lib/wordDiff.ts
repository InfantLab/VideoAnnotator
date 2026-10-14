/** Word-by-word differences between two prompts (spec 020). */

export type DiffPart = { kind: 'same' | 'added' | 'removed'; text: string };

const tokens = (text: string) => text.split(/(\s+)/).filter((t) => t !== '');
const isSpace = (t: string) => /^\s+$/.test(t);

/** Longest-common-subsequence diff over words; whitespace kept as written. */
export function wordDiff(a: string, b: string): DiffPart[] {
  const x = tokens(a);
  const y = tokens(b);
  const lcs: number[][] = Array.from({ length: x.length + 1 }, () => new Array(y.length + 1).fill(0));
  for (let i = x.length - 1; i >= 0; i--) {
    for (let j = y.length - 1; j >= 0; j--) {
      lcs[i][j] = x[i] === y[j] ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
    }
  }
  const parts: DiffPart[] = [];
  const push = (kind: DiffPart['kind'], text: string) => {
    const last = parts[parts.length - 1];
    if (last?.kind === kind) last.text += text;
    else parts.push({ kind, text });
  };
  let i = 0;
  let j = 0;
  while (i < x.length && j < y.length) {
    if (x[i] === y[j]) {
      push('same', x[i]);
      i++;
      j++;
    } else if (lcs[i + 1][j] >= lcs[i][j + 1]) {
      push('removed', x[i++]);
    } else {
      push('added', y[j++]);
    }
  }
  while (i < x.length) push('removed', x[i++]);
  while (j < y.length) push('added', y[j++]);
  return parts;
}

/** True when the two differ only in spaces and line breaks. */
export function whitespaceOnly(a: string, b: string): boolean {
  return a !== b && tokens(a).filter((t) => !isSpace(t)).join(' ') === tokens(b).filter((t) => !isSpace(t)).join(' ');
}
