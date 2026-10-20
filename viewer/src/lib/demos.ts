/** The demo results that open in the Video page without a server (spec 029 R14). */
import { DEMO_DATA_SETS } from '@/utils/debugUtils';

export type DemoKey = keyof typeof DEMO_DATA_SETS;

export const DEMO_KEY: DemoKey = 'peekaboo-rep3-v1.1.1';

export const DEMOS: { key: DemoKey; title: string; description: string }[] = [
  {
    key: 'peekaboo-rep3-v1.1.1',
    title: 'Peekaboo',
    description: 'A parent and baby playing peekaboo: speech, speakers, faces and people.',
  },
];

export function isDemoKey(value: string | null): value is DemoKey {
  return value !== null && value in DEMO_DATA_SETS;
}
