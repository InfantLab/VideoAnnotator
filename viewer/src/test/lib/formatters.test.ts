import { describe, it, expect } from 'vitest';
import {
    formatUptime,
    formatMemory,
    getWorkerStatusColor,
    getHealthStatusColor,
} from '@/lib/formatters';

describe('formatUptime', () => {
    it('should format seconds', () => {
        expect(formatUptime(0)).toBe('0 seconds');
        expect(formatUptime(1)).toBe('1 second');
        expect(formatUptime(30)).toBe('30 seconds');
        expect(formatUptime(59)).toBe('59 seconds');
    });

    it('should format minutes', () => {
        expect(formatUptime(60)).toBe('1 minute');
        expect(formatUptime(120)).toBe('2 minutes');
        expect(formatUptime(3599)).toBe('59 minutes');
    });

    it('should format hours', () => {
        expect(formatUptime(3600)).toBe('1 hour');
        expect(formatUptime(7200)).toBe('2 hours');
    });

    it('should format hours with remaining minutes', () => {
        expect(formatUptime(3660)).toBe('1 hour, 1 minute');
        expect(formatUptime(3720)).toBe('1 hour, 2 minutes');
        expect(formatUptime(7260)).toBe('2 hours, 1 minute');
    });

    it('should format days', () => {
        expect(formatUptime(86400)).toBe('1 day');
        expect(formatUptime(172800)).toBe('2 days');
    });

    it('should format days with remaining hours', () => {
        expect(formatUptime(90000)).toBe('1 day, 1 hour');
        expect(formatUptime(93600)).toBe('1 day, 2 hours');
        expect(formatUptime(176400)).toBe('2 days, 1 hour');
    });
});

describe('formatMemory', () => {
    it('should format zero bytes', () => {
        const result = formatMemory(0, 0);
        expect(result).toEqual({ used: '0 B', total: '0 B', percentage: 0 });
    });

    it('should format bytes', () => {
        const result = formatMemory(512, 1024);
        expect(result.used).toBe('512 B');
        expect(result.total).toBe('1.0 KB');
        expect(result.percentage).toBe(50);
    });

    it('should format megabytes', () => {
        const result = formatMemory(1048576, 2097152); // 1 MB / 2 MB
        expect(result.used).toBe('1.0 MB');
        expect(result.total).toBe('2.0 MB');
        expect(result.percentage).toBe(50);
    });

    it('should format gigabytes', () => {
        const result = formatMemory(5368709120, 10737418240); // 5 GB / 10 GB
        expect(result.used).toBe('5.0 GB');
        expect(result.total).toBe('10.0 GB');
        expect(result.percentage).toBe(50);
    });

    it('should handle total of zero without division error', () => {
        const result = formatMemory(100, 0);
        expect(result.percentage).toBe(0);
    });

    it('should round percentage correctly', () => {
        const result = formatMemory(1, 3);
        expect(result.percentage).toBe(33);
    });
});

describe('getWorkerStatusColor', () => {
    it('should return green for idle', () => {
        expect(getWorkerStatusColor('idle')).toContain('green');
    });

    it('should return yellow for busy', () => {
        expect(getWorkerStatusColor('busy')).toContain('yellow');
    });

    it('should return red for overloaded', () => {
        expect(getWorkerStatusColor('overloaded')).toContain('red');
    });

    it('should return gray for unknown status', () => {
        expect(getWorkerStatusColor('unknown' as 'idle')).toContain('gray');
    });
});

describe('getHealthStatusColor', () => {
    it('should return green for healthy', () => {
        expect(getHealthStatusColor('healthy')).toContain('green');
    });

    it('should return yellow for degraded', () => {
        expect(getHealthStatusColor('degraded')).toContain('yellow');
    });

    it('should return red for unhealthy', () => {
        expect(getHealthStatusColor('unhealthy')).toContain('red');
    });

    it('should return gray for unknown status', () => {
        expect(getHealthStatusColor('whatever')).toContain('gray');
    });
});
