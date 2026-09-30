// Unit tests for server capability detection
// Tests detectServerCapabilities function with v1.2.x and v1.3.0 health responses

import { describe, it, expect } from 'vitest';
import { detectServerCapabilities, isVersionAtLeast } from '@/api/capabilities';
import type { EnhancedHealthResponse, LegacyHealthResponse } from '@/types/api';

describe('Server Capability Detection', () => {
    describe('v1.3.0 health response parsing', () => {
        it('should detect v1.3.0 capabilities from enhanced health response with uptime_seconds', () => {
            const healthResponse: EnhancedHealthResponse = {
                status: 'healthy',
                version: '1.3.0',
                api_version: '1.3.0',
                uptime_seconds: 86400, // Key v1.3.0 marker: uptime_seconds field present
                message: 'System operational',
            };

            const capabilities = detectServerCapabilities(healthResponse);

            expect(capabilities.version).toBe('1.3.0');
            expect(capabilities.supportsJobCancellation).toBe(true);
            expect(capabilities.supportsConfigValidation).toBe(true);
            expect(capabilities.supportsEnhancedHealth).toBe(true);
            expect(capabilities.supportsEnhancedErrors).toBe(true);
            expect(capabilities.detectedAt).toBeInstanceOf(Date);
        });

        it('should detect v1.3.0 with gpu_status present', () => {
            const healthResponse: EnhancedHealthResponse = {
                status: 'healthy',
                version: '1.3.0',
                api_version: '1.3.0',
                uptime_seconds: 3600,
                gpu_status: {
                    available: true,
                    device_name: 'NVIDIA GeForce RTX 3090',
                    cuda_version: '12.1',
                    memory_total: 25769803776,
                    memory_used: 5153960755,
                    memory_free: 20615843020,
                },
            };

            const capabilities = detectServerCapabilities(healthResponse);

            expect(capabilities.version).toBe('1.3.0');
            expect(capabilities.supportsEnhancedHealth).toBe(true);
        });

        it('should detect v1.3.0 with worker_info present', () => {
            const healthResponse: EnhancedHealthResponse = {
                status: 'healthy',
                version: '1.3.0',
                api_version: '1.3.0',
                uptime_seconds: 1800,
                worker_info: {
                    active_jobs: 2,
                    queued_jobs: 1,
                    max_concurrent_jobs: 4,
                    worker_status: 'busy',
                },
            };

            const capabilities = detectServerCapabilities(healthResponse);

            expect(capabilities.version).toBe('1.3.0');
            expect(capabilities.supportsEnhancedHealth).toBe(true);
        });
    });

    describe('v1.2.x health response parsing', () => {
        it('should detect v1.2.x from legacy health response without uptime_seconds', () => {
            const healthResponse: LegacyHealthResponse = {
                status: 'ok',
                api_version: '1.2.1',
                message: 'Server healthy',
            };

            const capabilities = detectServerCapabilities(healthResponse);

            expect(capabilities.version).toBe('1.2.1');
            expect(capabilities.supportsJobCancellation).toBe(false);
            expect(capabilities.supportsConfigValidation).toBe(false);
            expect(capabilities.supportsEnhancedHealth).toBe(false);
            expect(capabilities.supportsEnhancedErrors).toBe(false);
        });

        it('should handle v1.2.x with videoannotator_version field', () => {
            const healthResponse: LegacyHealthResponse = {
                status: 'ok',
                videoannotator_version: '1.2.0',
                message: 'OK',
            };

            const capabilities = detectServerCapabilities(healthResponse);

            expect(capabilities.version).toBe('1.2.0');
            expect(capabilities.supportsEnhancedHealth).toBe(false);
        });

        it('should handle minimal v1.2.x response with just status', () => {
            const healthResponse: LegacyHealthResponse = {
                status: 'ok',
                message: 'Healthy',
            };

            const capabilities = detectServerCapabilities(healthResponse);

            expect(capabilities.version).toBe('unknown');
            expect(capabilities.supportsEnhancedHealth).toBe(false);
        });
    });

    describe('version comparison', () => {
        it('should correctly identify v1.3.0+ with isVersionAtLeast', () => {
            expect(isVersionAtLeast('1.3.0', '1.3.0')).toBe(true);
            expect(isVersionAtLeast('1.3.1', '1.3.0')).toBe(true);
            expect(isVersionAtLeast('1.4.0', '1.3.0')).toBe(true);
            expect(isVersionAtLeast('2.0.0', '1.3.0')).toBe(true);
        });

        it('should correctly identify versions below v1.3.0', () => {
            expect(isVersionAtLeast('1.2.1', '1.3.0')).toBe(false);
            expect(isVersionAtLeast('1.2.0', '1.3.0')).toBe(false);
            expect(isVersionAtLeast('1.0.0', '1.3.0')).toBe(false);
            expect(isVersionAtLeast('0.9.9', '1.3.0')).toBe(false);
        });

        it('should handle version strings with prerelease tags', () => {
            expect(isVersionAtLeast('1.3.0-alpha.1', '1.3.0')).toBe(true);
            expect(isVersionAtLeast('1.3.0-rc.1', '1.3.0')).toBe(true);
            expect(isVersionAtLeast('v1.3.0', '1.3.0')).toBe(true); // Handle 'v' prefix
        });
    });
});
