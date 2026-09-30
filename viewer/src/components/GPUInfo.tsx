import { useState } from 'react';
import { Cpu, AlertTriangle, CheckCircle2, XCircle, ChevronDown, ChevronRight } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Skeleton } from '@/components/ui/skeleton';
import { Button } from '@/components/ui/button';
import { useSystemHealth } from '@/hooks/useSystemHealth';

export function GPUInfo() {
    const [isOpen, setIsOpen] = useState(false);
    const { data: health, isLoading, error } = useSystemHealth();

    if (isLoading) {
        return (
            <Card>
                <CardHeader>
                    <div className="flex items-center justify-between">
                        <div>
                            <CardTitle className="flex items-center gap-2">
                                <Cpu className="h-5 w-5" />
                                Server GPU Information
                            </CardTitle>
                            <CardDescription>Hardware acceleration status</CardDescription>
                        </div>
                        <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => setIsOpen(!isOpen)}
                            className="gap-1"
                        >
                            {isOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                            {isOpen ? 'Collapse' : 'Expand'}
                        </Button>
                    </div>
                </CardHeader>
                {isOpen && (
                    <CardContent>
                        <Skeleton className="h-20 w-full" />
                    </CardContent>
                )}
            </Card>
        );
    }

    if (error || !health) {
        return null; // Silently fail if health endpoint not available
    }

    const gpu = health.gpu;
    const hasGPU = gpu?.available === true;
    const gpuName = gpu?.device_name || 'Unknown GPU';
    const deviceCount = gpu?.device_count || 0;
    const hasCompatibilityWarning = Boolean(gpu?.compatibility_warning);

    // Determine GPU status
    let status: 'working' | 'warning' | 'unavailable';
    let statusIcon: React.ReactNode;
    let statusVariant: 'default' | 'secondary' | 'destructive' | 'outline';

    if (!hasGPU) {
        status = 'unavailable';
        statusIcon = <XCircle className="h-4 w-4" />;
        statusVariant = 'outline';
    } else if (hasCompatibilityWarning) {
        status = 'warning';
        statusIcon = <AlertTriangle className="h-4 w-4" />;
        statusVariant = 'secondary';
    } else {
        status = 'working';
        statusIcon = <CheckCircle2 className="h-4 w-4" />;
        statusVariant = 'secondary';
    }

    return (
        <Card>
            <CardHeader>
                <div className="flex items-center justify-between">
                    <div>
                        <CardTitle className="flex items-center gap-2">
                            <Cpu className="h-5 w-5" />
                            Server GPU Information
                        </CardTitle>
                        <CardDescription>Hardware acceleration status for video processing</CardDescription>
                    </div>
                    <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setIsOpen(!isOpen)}
                        className="gap-1"
                    >
                        {isOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                        {isOpen ? 'Collapse' : 'Expand'}
                    </Button>
                </div>
            </CardHeader>
            {isOpen && (
                <CardContent className="space-y-4">
                    <div className="grid gap-4 md:grid-cols-2">
                        <div>
                            <p className="font-medium text-sm">Status</p>
                            <div className="mt-1 flex items-center gap-2">
                                <Badge variant={statusVariant} className="gap-1">
                                    {statusIcon}
                                    {status === 'working' ? 'Active' : status === 'warning' ? 'Warning' : 'Unavailable'}
                                </Badge>
                            </div>
                        </div>

                        {hasGPU && (
                            <>
                                <div>
                                    <p className="font-medium text-sm">Device Name</p>
                                    <p className="text-sm text-muted-foreground mt-1">{gpuName}</p>
                                </div>

                                <div>
                                    <p className="font-medium text-sm">Device Count</p>
                                    <p className="text-sm text-muted-foreground mt-1">{deviceCount}</p>
                                </div>

                                <div>
                                    <p className="font-medium text-sm">Current Device</p>
                                    <p className="text-sm text-muted-foreground mt-1">
                                        {gpu?.current_device !== undefined ? `Device ${gpu.current_device}` : 'N/A'}
                                    </p>
                                </div>

                                {gpu?.compute_capability !== undefined && (
                                    <div>
                                        <p className="font-medium text-sm">Compute Capability</p>
                                        <p className="text-sm text-muted-foreground mt-1">{gpu.compute_capability}</p>
                                    </div>
                                )}

                                {gpu?.pytorch_version && (
                                    <div>
                                        <p className="font-medium text-sm">PyTorch Version</p>
                                        <p className="text-sm text-muted-foreground mt-1">{gpu.pytorch_version}</p>
                                    </div>
                                )}

                                {(gpu?.memory_allocated !== undefined || gpu?.memory_reserved !== undefined) && (
                                    <>
                                        <div>
                                            <p className="font-medium text-sm">Memory Allocated</p>
                                            <p className="text-sm text-muted-foreground mt-1">
                                                {gpu?.memory_allocated ? `${(gpu.memory_allocated / 1024 / 1024 / 1024).toFixed(2)} GB` : '0 GB'}
                                            </p>
                                        </div>

                                        <div>
                                            <p className="font-medium text-sm">Memory Reserved</p>
                                            <p className="text-sm text-muted-foreground mt-1">
                                                {gpu?.memory_reserved ? `${(gpu.memory_reserved / 1024 / 1024 / 1024).toFixed(2)} GB` : '0 GB'}
                                            </p>
                                        </div>
                                    </>
                                )}
                            </>
                        )}
                    </div>

                    {hasCompatibilityWarning && gpu?.compatibility_warning && (
                        <Alert>
                            <AlertTriangle className="h-4 w-4" />
                            <AlertTitle>GPU Compatibility Warning</AlertTitle>
                            <AlertDescription className="space-y-2">
                                <p>{gpu.compatibility_warning}</p>
                                {gpu.min_compute_capability && (
                                    <p className="text-xs">
                                        Minimum compute capability required: {gpu.min_compute_capability}
                                        {gpu.compute_capability && ` (your GPU: ${gpu.compute_capability})`}
                                    </p>
                                )}
                                <p className="text-xs font-medium">
                                    Processing will automatically fall back to CPU. Consider downgrading PyTorch or upgrading your GPU for better performance.
                                </p>
                            </AlertDescription>
                        </Alert>
                    )}

                    {!hasGPU && (
                        <Alert>
                            <AlertDescription>
                                No GPU detected. Video processing will use CPU, which may be slower for large files or complex pipelines.
                                For best performance, ensure the server has a compatible NVIDIA GPU with CUDA support.
                            </AlertDescription>
                        </Alert>
                    )}

                    {hasGPU && !hasCompatibilityWarning && gpu?.memory_allocated === 0 && gpu?.memory_reserved === 0 && (
                        <Alert>
                            <AlertTriangle className="h-4 w-4" />
                            <AlertTitle>GPU Not Currently Active</AlertTitle>
                            <AlertDescription>
                                A GPU is available but no memory is currently allocated. The GPU will be activated when processing jobs.
                                If you experience issues, check the server logs for GPU initialization errors.
                            </AlertDescription>
                        </Alert>
                    )}
                </CardContent>
            )}
        </Card>
    );
}
