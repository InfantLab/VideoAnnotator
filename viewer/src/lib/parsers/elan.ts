// ELAN (.eaf) ground-truth parser.
// Ported from mother-infant-touch-detection/00_preprocessing/parse_eaf_to_timeline.py's
// parse_elan()/fourway_at() — reads TIME_SLOT timings and TIER/ALIGNABLE_ANNOTATION
// elements from the standard ELAN XML format, then (at lookup time) collapses
// whichever tiers are active at a given moment into a four-way category
// using a scheme-specific tier-to-side mapping (default: Crucianelli et al.
// 2019's mother/infant touch coding scheme).

import type { ElanTierAnnotation } from '@/types/annotations';

// Crucianelli et al. (2019) touch-coding tier names — the scheme this was
// originally built for. Pass a different mapping to getElanFourwayAtTime
// for a different ELAN coding scheme.
export const DEFAULT_MOTHER_TIERS = new Set([
    'Excitatory touch',
    'Downregulatory touch',
    'Static touch',
    'Instrumental touch',
    'Incidental touch'
]);
export const DEFAULT_INFANT_TIERS = new Set([
    'Infant incidental',
    'Infant intentional'
]);

export type ElanFourwayCategory = 'BOTH_TOUCH' | 'MATERNAL_TOUCH' | 'INFANT_TOUCH' | 'NO_TOUCH';

export interface ElanFourwayResult {
    category: ElanFourwayCategory;
    motherActive: string[];
    infantActive: string[];
}

/**
 * Parses a .eaf (ELAN Annotation Format) XML file into a flat list of tier
 * annotations with second-resolution timestamps.
 */
export async function parseElanFile(file: File): Promise<ElanTierAnnotation[]> {
    const text = await file.text();
    const doc = new DOMParser().parseFromString(text, 'text/xml');

    const parserError = doc.querySelector('parsererror');
    if (parserError) {
        throw new Error(`Invalid ELAN (.eaf) XML: ${parserError.textContent}`);
    }

    const slots = new Map<string, number>();
    doc.querySelectorAll('TIME_ORDER > TIME_SLOT').forEach((slot) => {
        const id = slot.getAttribute('TIME_SLOT_ID');
        const value = slot.getAttribute('TIME_VALUE');
        if (id && value !== null) {
            slots.set(id, parseInt(value, 10));
        }
    });

    const annotations: ElanTierAnnotation[] = [];
    doc.querySelectorAll('TIER').forEach((tier) => {
        const tierId = tier.getAttribute('TIER_ID') || '';
        tier.querySelectorAll('ALIGNABLE_ANNOTATION').forEach((ann) => {
            const ref1 = ann.getAttribute('TIME_SLOT_REF1');
            const ref2 = ann.getAttribute('TIME_SLOT_REF2');
            const t1 = ref1 ? slots.get(ref1) : undefined;
            const t2 = ref2 ? slots.get(ref2) : undefined;
            if (t1 === undefined || t2 === undefined) return;

            const valueElem = ann.querySelector('ANNOTATION_VALUE');
            const value = valueElem?.textContent?.trim() || '';

            annotations.push({
                tier: tierId,
                value,
                startSec: t1 / 1000,
                endSec: t2 / 1000
            });
        });
    });

    if (annotations.length === 0) {
        console.warn('ELAN file contains no alignable annotations');
    }

    return annotations.sort((a, b) => a.startSec - b.startSec);
}

/**
 * Collapses whichever ELAN tiers are active at `timeSec` into a four-way
 * category, mirroring parse_eaf_to_timeline.py's fourway_at(). Default tier
 * mapping is the Crucianelli et al. (2019) touch-coding scheme; pass
 * motherTiers/infantTiers for a different ELAN coding scheme.
 */
export function getElanFourwayAtTime(
    annotations: ElanTierAnnotation[],
    timeSec: number,
    motherTiers: Set<string> = DEFAULT_MOTHER_TIERS,
    infantTiers: Set<string> = DEFAULT_INFANT_TIERS
): ElanFourwayResult {
    const motherActive: string[] = [];
    const infantActive: string[] = [];

    for (const a of annotations) {
        if (timeSec >= a.startSec && timeSec <= a.endSec) {
            const label = a.value ? `${a.tier}: ${a.value}` : a.tier;
            if (motherTiers.has(a.tier)) {
                motherActive.push(label);
            } else if (infantTiers.has(a.tier)) {
                infantActive.push(label);
            }
        }
    }

    let category: ElanFourwayCategory;
    if (motherActive.length > 0 && infantActive.length > 0) {
        category = 'BOTH_TOUCH';
    } else if (motherActive.length > 0) {
        category = 'MATERNAL_TOUCH';
    } else if (infantActive.length > 0) {
        category = 'INFANT_TOUCH';
    } else {
        category = 'NO_TOUCH';
    }

    return { category, motherActive, infantActive };
}

/**
 * Cheap sniff-test for the merger's file-type detector. .eaf files are
 * XML, detected by extension (like .vtt/.rttm), not content — this exists
 * mainly to reject obviously-non-ELAN XML.
 */
export async function isValidElanFile(file: File): Promise<boolean> {
    try {
        const sample = await file.slice(0, 2000).text();
        return sample.includes('<ANNOTATION_DOCUMENT') || sample.includes('<TIER');
    } catch {
        return false;
    }
}
