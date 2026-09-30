// System health and status types for VideoAnnotator API

export interface SystemMemory {
    total: number;
    available: number;
    percent: number;
    used: number;
    free: number;
}

export interface SystemDisk {
    total: number;
    used: number;
    free: number;
    percent: number;
}

export interface SystemInfo {
    platform: string;
    python_version: string;
    cpu_count: number;
    cpu_percent: number;
    memory_percent: number;
    memory: SystemMemory;
    disk: SystemDisk;
}

export interface GPUInfo {
    available: boolean;
    device_count?: number;
    current_device?: number;
    device_name?: string;
    memory_allocated?: number;
    memory_reserved?: number;
    compatibility_warning?: string;
    pytorch_version?: string;
    min_compute_capability?: number;
    compute_capability?: number;
}

export interface DatabaseInfo {
    status: string;
    message: string;
}

export interface PipelinesInfo {
    total: number;
    names: string[];
}

export interface ServiceInfo {
    status?: string;
    message?: string;
}

export interface ServicesInfo {
    database: ServiceInfo;
    job_queue: string;
    pipelines: string;
}

export interface SecurityInfo {
    auth_required: boolean;
}

export interface WorkersInfo {
    status: string;
    active_jobs: number;
    processing_jobs: string[];
    queued_jobs: number;
    max_concurrent_workers: number;
    poll_interval_seconds: number;
}

export interface SystemHealthResponse {
    status: string;
    timestamp: string;
    api_version: string;
    videoannotator_version: string;
    system: SystemInfo;
    database: DatabaseInfo;
    gpu: GPUInfo;
    pipelines: PipelinesInfo;
    services: ServicesInfo;
    security: SecurityInfo;
    workers?: WorkersInfo;
    uptime_seconds: number;
}
