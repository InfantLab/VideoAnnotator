/** Helpers for the prompt workbench (spec 020). */

/** Seconds from "2, 5.5, 9"; nothing for anything that isn't a number. */
export function parseMoments(text: string): number[] {
  return text
    .split(/[,\s]+/)
    .map((t) => Number(t))
    .filter((n) => Number.isFinite(n) && n >= 0);
}

/** `count` moments spread through a video, away from its very start and end. */
export function evenlySpaced(count: number, duration: number): number[] {
  if (count < 1 || duration <= 0) return [];
  return Array.from({ length: count }, (_, i) => Math.round(((i + 1) * duration * 10) / (count + 1)) / 10);
}
