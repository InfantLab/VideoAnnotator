// Pipelines sample frames (face analysis about once a second, person tracking
// every few frames), so a detection is a snapshot at one timestamp. Drawing it
// only within a fixed tolerance makes it flash during playback and vanish when
// paused between samples; drawing every snapshot inside a wide window stacks
// several frames' boxes on top of each other.

interface Timestamped {
    timestamp: number;
}

interface SampleIndex<T> {
    times: number[];
    byTime: Map<number, T[]>;
    maxHold: number;
}

/** How long a sample stays on screen, as a multiple of the clip's typical sample interval. */
export const HOLD_FACTOR = 1.5;

// When a track has a single sampled frame there is no interval to measure.
const SINGLE_SAMPLE_HOLD = 1.0;

const indexCache = new WeakMap<readonly Timestamped[], SampleIndex<Timestamped>>();

function median(values: number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function buildIndex<T extends Timestamped>(items: readonly T[]): SampleIndex<T> {
    const byTime = new Map<number, T[]>();
    for (const item of items) {
        const group = byTime.get(item.timestamp);
        if (group) group.push(item);
        else byTime.set(item.timestamp, [item]);
    }
    const times = [...byTime.keys()].sort((a, b) => a - b);
    const gaps = times.slice(1).map((time, i) => time - times[i]);
    // The median ignores the long gaps where nothing was detected.
    const maxHold = gaps.length ? HOLD_FACTOR * median(gaps) : SINGLE_SAMPLE_HOLD;
    return { times, byTime, maxHold };
}

/**
 * The detections of the latest sampled frame at or before `currentTime`, held
 * until the next sampled frame or for at most HOLD_FACTOR × the median sample
 * interval, so a detection that disappears doesn't stay on screen.
 */
export function sampledAtTime<T extends Timestamped>(items: readonly T[], currentTime: number): T[] {
    if (items.length === 0) return [];
    let index = indexCache.get(items) as SampleIndex<T> | undefined;
    if (!index) {
        index = buildIndex(items);
        indexCache.set(items, index as SampleIndex<Timestamped>);
    }

    const { times, byTime, maxHold } = index;
    let low = 0;
    let high = times.length - 1;
    let latest = -1;
    while (low <= high) {
        const middle = (low + high) >> 1;
        if (times[middle] <= currentTime) {
            latest = middle;
            low = middle + 1;
        } else {
            high = middle - 1;
        }
    }
    if (latest < 0) return [];
    const sampleTime = times[latest];
    if (currentTime - sampleTime > maxHold) return [];
    return byTime.get(sampleTime) ?? [];
}
