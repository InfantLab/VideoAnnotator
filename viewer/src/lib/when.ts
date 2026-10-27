/**
 * Dates and times, one way everywhere (viewer design P8, UAT #2): "26 Oct 2026, 17:50";
 * today just "17:50"; this year drops the year.
 */

const parse = (value: string | Date | null | undefined): Date | null => {
  if (!value) return null;
  const date = value instanceof Date ? value : new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
};

const time = (d: Date) => `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
const dayMonth = (d: Date) => d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' });

/** "17:50", "26 Oct, 17:50" or "26 Oct 2025, 17:50"; '' when there's no date. */
export function formatWhen(value: string | Date | null | undefined, now: Date = new Date()): string {
  const d = parse(value);
  if (!d) return '';
  if (d.toDateString() === now.toDateString()) return time(d);
  return `${formatDay(d, now)}, ${time(d)}`;
}

/** "26 Oct", or "26 Oct 2025" in another year; '' when there's no date. */
export function formatDay(value: string | Date | null | undefined, now: Date = new Date()): string {
  const d = parse(value);
  if (!d) return '';
  return d.getFullYear() === now.getFullYear() ? dayMonth(d) : `${dayMonth(d)} ${d.getFullYear()}`;
}

/** "24 s", "4 min", "1 h 12 min" (viewer design "Words, numbers and time"). */
export function formatDuration(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s} s`;
  const minutes = Math.round(s / 60);
  if (minutes < 60) return `${minutes} min`;
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return m ? `${h} h ${m} min` : `${h} h`;
}
