/**
 * Comparing two VLM runs on the same video (spec 021). Samples are paired by
 * time; nothing is dropped silently: every sample is paired, unpaired or an
 * error, and the summary counts all three. Labels are compared as recorded
 * unless the caller asks to ignore case (and says so on screen).
 */
import type { JobResponse } from '@/api/client';
import type { ElanTierAnnotation, ProvenanceInfo, VLMFrameAnnotation } from '@/types/annotations';
import { getElanFourwayAtTime } from '@/lib/parsers/elan';
import { isPositiveLabel } from '@/lib/vlmLabels';

export interface PairedMoment {
  time: number;
  a: VLMFrameAnnotation | null;
  b: VLMFrameAnnotation | null;
  status: 'agree' | 'disagree' | 'unpaired' | 'error';
  /** ELAN four-way category at this time, when ground truth is loaded. */
  truth?: string;
}

const isError = (s: VLMFrameAnnotation) => s.label.startsWith('ERROR');

/** The median gap between samples: the run's sampling interval. */
export function samplingInterval(samples: VLMFrameAnnotation[]): number {
  const gaps = samples.slice(1).map((s, i) => s.timestamp_sec - samples[i].timestamp_sec).filter((g) => g > 0).sort((x, y) => x - y);
  return gaps.length ? gaps[Math.floor(gaps.length / 2)] : 0;
}

export function pairRuns(
  a: VLMFrameAnnotation[],
  b: VLMFrameAnnotation[],
  options: { ignoreCase?: boolean; intervalA?: number; intervalB?: number } = {},
): PairedMoment[] {
  const sortedA = [...a].sort((x, y) => x.timestamp_sec - y.timestamp_sec);
  const sortedB = [...b].sort((x, y) => x.timestamp_sec - y.timestamp_sec);
  const tolerance =
    Math.max(options.intervalA ?? samplingInterval(sortedA), options.intervalB ?? samplingInterval(sortedB)) / 2;
  const same = (x: string, y: string) => (options.ignoreCase ? x.toLowerCase() === y.toLowerCase() : x === y);

  const usedB = new Set<number>();
  const pairs: PairedMoment[] = [];
  for (const sample of sortedA) {
    let best = -1;
    let bestDelta = Infinity;
    sortedB.forEach((other, j) => {
      const delta = Math.abs(other.timestamp_sec - sample.timestamp_sec);
      if (!usedB.has(j) && delta < bestDelta) {
        best = j;
        bestDelta = delta;
      }
    });
    if (best >= 0 && (bestDelta === 0 || bestDelta <= tolerance)) {
      usedB.add(best);
      const other = sortedB[best];
      pairs.push({
        time: sample.timestamp_sec,
        a: sample,
        b: other,
        status: isError(sample) || isError(other) ? 'error' : same(sample.label, other.label) ? 'agree' : 'disagree',
      });
    } else {
      pairs.push({ time: sample.timestamp_sec, a: sample, b: null, status: 'unpaired' });
    }
  }
  sortedB.forEach((other, j) => {
    if (!usedB.has(j)) pairs.push({ time: other.timestamp_sec, a: null, b: other, status: 'unpaired' });
  });
  return pairs.sort((x, y) => x.time - y.time);
}

export function withTruth(pairs: PairedMoment[], truth: ElanTierAnnotation[]): PairedMoment[] {
  return pairs.map((p) => ({ ...p, truth: getElanFourwayAtTime(truth, p.time).category }));
}

/** Whether a run's label agrees with the ELAN category (touch vs no touch). */
export const matchesTruth = (sample: VLMFrameAnnotation | null, truth?: string) =>
  sample !== null && truth !== undefined && !isError(sample) && isPositiveLabel(sample.label) === (truth !== 'NO_TOUCH');

export interface ComparisonSummary {
  samplesA: number;
  samplesB: number;
  compared: number;
  agree: number;
  disagree: number;
  unpaired: number;
  errors: number;
  agreementRate: number | null;
  /** "A label → B label" counts over compared moments. */
  labelPairs: Array<{ a: string; b: string; count: number }>;
  truth?: { compared: number; agreeA: number; agreeB: number };
}

export function summarize(pairs: PairedMoment[], samplesA: number, samplesB: number): ComparisonSummary {
  const compared = pairs.filter((p) => p.status === 'agree' || p.status === 'disagree');
  const agree = compared.filter((p) => p.status === 'agree').length;
  const counts = new Map<string, number>();
  for (const p of compared) {
    const key = `${p.a!.label}\u0000${p.b!.label}`;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  const withTruthPairs = compared.filter((p) => p.truth !== undefined);
  return {
    samplesA,
    samplesB,
    compared: compared.length,
    agree,
    disagree: compared.length - agree,
    unpaired: pairs.filter((p) => p.status === 'unpaired').length,
    errors: pairs.filter((p) => p.status === 'error').length,
    agreementRate: compared.length ? agree / compared.length : null,
    labelPairs: [...counts.entries()]
      .map(([key, count]) => {
        const [a, b] = key.split('\u0000');
        return { a, b, count };
      })
      .sort((x, y) => y.count - x.count),
    truth: withTruthPairs.length
      ? {
          compared: withTruthPairs.length,
          agreeA: withTruthPairs.filter((p) => matchesTruth(p.a, p.truth)).length,
          agreeB: withTruthPairs.filter((p) => matchesTruth(p.b, p.truth)).length,
        }
      : undefined,
  };
}

const csvCell = (value: unknown) => {
  const text = value === null || value === undefined ? '' : String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
};

export function comparisonCsv(pairs: PairedMoment[], jobA: string, jobB: string): string {
  const header = ['time_sec', 'status', `label_${jobA}`, `label_${jobB}`, 'ground_truth', `reasoning_${jobA}`, `reasoning_${jobB}`];
  const rows = pairs.map((p) => [p.time, p.status, p.a?.label, p.b?.label, p.truth, p.a?.reasoning, p.b?.reasoning]);
  return [header, ...rows].map((r) => r.map(csvCell).join(',')).join('\n') + '\n';
}

/** One job's VLM run, as the comparison page loads it. */
export interface Run {
  job: JobResponse;
  samples: VLMFrameAnnotation[];
  provenance: ProvenanceInfo;
}

const videoOf = (job: JobResponse) => job as JobResponse & { video_filename?: string; video_size_bytes?: number };

/** Why two runs can't be compared, or null when they are of the same video. */
export function differentVideos(a: Run, b: Run): string | null {
  const hashA = a.provenance.kind === 'recorded' ? a.provenance.record.input?.sha256 : null;
  const hashB = b.provenance.kind === 'recorded' ? b.provenance.record.input?.sha256 : null;
  if (hashA && hashB) return hashA === hashB ? null : 'The two jobs ran on different videos (their contents differ).';
  const va = videoOf(a.job);
  const vb = videoOf(b.job);
  if (va.video_filename && va.video_filename === vb.video_filename && va.video_size_bytes === vb.video_size_bytes) return null;
  return 'These jobs ran on different videos (name or size differ), so their labels can’t be compared moment by moment.';
}

