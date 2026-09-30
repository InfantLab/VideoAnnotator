// WebVTT parser for speech recognition data
// Reference: https://developer.mozilla.org/en-US/docs/Web/API/WebVTT_API
// VideoAnnotator produces WebVTT files for speech recognition

import type { WebVTTCue } from '@/types/annotations';
import { validateWebVTTData } from '@/lib/validation';

/**
 * Parses WebVTT timestamp format (HH:MM:SS.mmm) to seconds
 */
function parseTimestamp(timestamp: string): number {
    const parts = timestamp.split(':');
    if (parts.length !== 3) {
        throw new Error(`Invalid timestamp format: ${timestamp}`);
    }

    const hours = parseInt(parts[0], 10);
    const minutes = parseInt(parts[1], 10);
    const seconds = parseFloat(parts[2]);

    return hours * 3600 + minutes * 60 + seconds;
}

/**
 * Parses a WebVTT cue block
 */
function parseCue(cueBlock: string): WebVTTCue {
    const lines = cueBlock.trim().split('\n');

    if (lines.length < 2) {
        throw new Error('Invalid cue block: must have at least timestamp and text');
    }

    let id: string | undefined;
    let timestampLine: string;
    let textLines: string[];

    // Check if first line is an ID (doesn't contain -->)
    if (!lines[0].includes('-->')) {
        id = lines[0];
        timestampLine = lines[1];
        textLines = lines.slice(2);
    } else {
        timestampLine = lines[0];
        textLines = lines.slice(1);
    }

    // Parse timestamp line (e.g., "00:00:01.000 --> 00:00:03.000")
    const timestampMatch = timestampLine.match(/^(\S+)\s+-->\s+(\S+)(?:\s+(.*))?$/);
    if (!timestampMatch) {
        throw new Error(`Invalid timestamp line: ${timestampLine}`);
    }

    const startTime = parseTimestamp(timestampMatch[1]);
    const endTime = parseTimestamp(timestampMatch[2]);
    const settings = timestampMatch[3]; // WebVTT cue settings (optional)

    // Join text lines
    const text = textLines.join('\n');

    return {
        id,
        startTime,
        endTime,
        text,
        settings
    };
}

/**
 * Parses WebVTT file content and returns array of cues
 */
export async function parseWebVTT(file: File): Promise<WebVTTCue[]> {
    try {
        const content = await file.text();

        // Split into lines and remove empty lines
        const lines = content.split('\n').map(line => line.trim());

        // Validate WebVTT header
        if (lines.length === 0 || lines[0] !== 'WEBVTT') {
            throw new Error('Invalid WebVTT file: missing WEBVTT header');
        }

        // Find cue blocks (separated by empty lines)
        const cueBlocks: string[] = [];
        let currentBlock = '';

        for (let i = 1; i < lines.length; i++) {
            const line = lines[i];

            if (line === '') {
                if (currentBlock.trim()) {
                    cueBlocks.push(currentBlock.trim());
                    currentBlock = '';
                }
            } else {
                currentBlock += line + '\n';
            }
        }

        // Add final block if exists
        if (currentBlock.trim()) {
            cueBlocks.push(currentBlock.trim());
        }

        // Parse each cue block
        const cues: WebVTTCue[] = [];

        for (const block of cueBlocks) {
            // Skip NOTE blocks and other non-cue content
            if (block.startsWith('NOTE') || !block.includes('-->')) {
                continue;
            }

            try {
                const cue = parseCue(block);
                cues.push(cue);
            } catch (error) {
                console.warn(`Skipping invalid cue block: ${error}`);
                // Continue parsing other cues instead of failing completely
            }
        }

        // Validate parsed data
        const validatedCues = validateWebVTTData(cues);

        // Sort by start time
        validatedCues.sort((a, b) => a.startTime - b.startTime);

        return validatedCues;

    } catch (error) {
        throw new Error(`Failed to parse WebVTT file: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
}

/**
 * Converts WebVTT cues to timeline events for the viewer
 */
export function webVTTToTimelineEvents(cues: WebVTTCue[]) {
    return cues.map((cue, index) => ({
        id: cue.id || `subtitle_${index}`,
        type: 'subtitle' as const,
        label: 'Speech',
        content: cue.text,
        startTime: cue.startTime,
        endTime: cue.endTime,
        confidence: 1.0 // WebVTT doesn't include confidence, assume high
    }));
}

/**
 * Gets subtitle text for a specific time
 */
export function getSubtitleAtTime(cues: WebVTTCue[], currentTime: number): string | null {
    const activeCue = cues.find(
        cue => currentTime >= cue.startTime && currentTime <= cue.endTime
    );

    return activeCue ? activeCue.text : null;
}

/**
 * Validates WebVTT file format without fully parsing
 */
export async function isValidWebVTT(file: File): Promise<boolean> {
    try {
        const firstLine = await file.slice(0, 100).text();
        return firstLine.trim().startsWith('WEBVTT');
    } catch {
        return false;
    }
}
