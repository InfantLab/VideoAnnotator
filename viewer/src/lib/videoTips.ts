/** The Video page's tips (UAT #70): what each says and what it points at. */

export interface Tip {
  id: string;
  /** What it points at; a tip whose target isn't on screen isn't shown. */
  target: string;
  text: string;
}

export const TIPS: Tip[] = [
  {
    id: 'list',
    target: '[data-testid="list-navigation"]',
    text: 'You opened this from a list: ‹ › go to the previous or next video in it, with the same lanes shown.',
  },
  {
    id: 'layers',
    target: '[data-tip="layers"]',
    text: 'Choose which results show as lanes, and which are drawn on the video.',
  },
  {
    id: 'lane',
    target: '[data-testid="lane-label"]',
    text: 'Click a lane to select it; [ and ] jump to its previous or next event.',
  },
  {
    id: 'notes',
    target: '[data-tip="notes"]',
    text: "Press N for a note at this moment, Shift + N twice for a span. They're kept with the video on the server and listed in the Notes tab.",
  },
];

export const TIPS_SEEN_KEY = 'videoannotator_video_tips_seen';

/** Whether this browser has seen (or skipped) the tips; storage can be blocked. */
export function tipsSeen(): boolean {
  try {
    return localStorage.getItem(TIPS_SEEN_KEY) === '1';
  } catch {
    return true;
  }
}

export function markTipsSeen(): void {
  try {
    localStorage.setItem(TIPS_SEEN_KEY, '1');
  } catch {
    // Blocked storage: they show again next time, which is harmless.
  }
}

// Tests drive the tips themselves, through "Show the tips again".
export const TIPS_START_BY_THEMSELVES = import.meta.env.MODE !== 'test';
