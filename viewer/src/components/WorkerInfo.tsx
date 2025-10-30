import { useState } from 'react';
import { Users, CheckCircle2, AlertTriangle, XCircle, ChevronDown, ChevronRight } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Skeleton } from '@/components/ui/skeleton';
import { Button } from '@/components/ui/button';
import { useSystemHealth } from '@/hooks/useSystemHealth';

export function WorkerInfo() {
    const [isOpen, setIsOpen] = useState(false);
    const { data: health, isLoading, error } = useSystemHealth();

    if (isLoading) {
        return (
            <Card>
                <CardHeader>
                    <div className="flex items-center justify-between">
                        <div>
                            <CardTitle className="flex items-center gap-2">
                                <Users className="h-5 w-5" />
                                Worker Information
                            </CardTitle>
                            <CardDescription>Job processing worker status</CardDescription>
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

    if (error || !health || !health.workers) {
        return null; // Silently fail if health endpoint not available or no worker info
    }

    const workers = health.workers;
    const workerStatus = workers.status.toLowerCase();

    // Determine worker status display
    let statusIcon: React.ReactNode;
    let statusVariant: 'default' | 'secondary' | 'destructive' | 'outline';
    let statusLabel: string;

    if (workerStatus === 'running' || workerStatus === 'active') {
        statusIcon = <CheckCircle2 className="h-4 w-4" />;
        statusVariant = 'secondary';
        statusLabel = 'Running';
    } else if (workerStatus === 'stopped' || workerStatus === 'idle') {
        statusIcon = <XCircle className="h-4 w-4" />;
        statusVariant = 'outline';
        statusLabel = 'Stopped';
    } else if (workerStatus === 'overloaded' || workerStatus === 'busy') {
        statusIcon = <AlertTriangle className="h-4 w-4" />;
        statusVariant = 'destructive';
        statusLabel = 'Overloaded';
    } else {
        statusIcon = <AlertTriangle className="h-4 w-4" />;
        statusVariant = 'secondary';
        statusLabel = workers.status;
    }

    // Determine if workers are busy
    const isBusy = workers.active_jobs > 0;
    const isOverloaded = workers.active_jobs >= workers.max_concurrent_workers && workers.queued_jobs > 0;

    return (
        <Card>
            <CardHeader>
                <div className="flex items-center justify-between">
                    <div>
                        <CardTitle className="flex items-center gap-2">
                            <Users className="h-5 w-5" />
                            Worker Information
                        </CardTitle>
                        <CardDescription>Job processing worker status</CardDescription>
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
                                    {statusLabel}
                                </Badge>
                            </div>
                        </div>

                        <div>
                            <p className="font-medium text-sm">Active Jobs</p>
                            <p className="text-sm text-muted-foreground mt-1">
                                {workers.active_jobs} / {workers.max_concurrent_workers} slots
                            </p>
                        </div>

                        <div>
                            <p className="font-medium text-sm">Queued Jobs</p>
                            <p className="text-sm text-muted-foreground mt-1">{workers.queued_jobs}</p>
                        </div>

                        <div>
                            <p className="font-medium text-sm">Max Concurrent Workers</p>
                            <p className="text-sm text-muted-foreground mt-1">{workers.max_concurrent_workers}</p>
                        </div>

                        <div>
                            <p className="font-medium text-sm">Poll Interval</p>
                            <p className="text-sm text-muted-foreground mt-1">{workers.poll_interval_seconds}s</p>
                        </div>

                        {workers.processing_jobs.length > 0 && (
                            <div className="md:col-span-2">
                                <p className="font-medium text-sm">Processing Jobs</p>
                                <div className="mt-1 flex flex-wrap gap-1">
                                    {workers.processing_jobs.map((jobId) => (
                                        <Badge key={jobId} variant="secondary" className="text-[10px] px-1.5 py-0">
                                            {jobId}
                                        </Badge>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>

                    {isOverloaded && (
                        <Alert>
                            <AlertTriangle className="h-4 w-4" />
                            <AlertDescription>
                                Workers are at maximum capacity ({workers.active_jobs}/{workers.max_concurrent_workers}) with {workers.queued_jobs} job{workers.queued_jobs !== 1 ? 's' : ''} queued.
                                New jobs will be queued until workers become available.
                            </AlertDescription>
                        </Alert>
                    )}

                    {!isBusy && workerStatus === 'stopped' && (
                        <Alert>
                            <AlertDescription>
                                Workers are currently stopped. Jobs will not be processed until workers are started.
                                Check server configuration or restart the worker service.
                            </AlertDescription>
                        </Alert>
                    )}

                    {isBusy && !isOverloaded && (
                        <Alert>
                            <CheckCircle2 className="h-4 w-4" />
                            <AlertDescription>
                                Workers are processing {workers.active_jobs} job{workers.active_jobs !== 1 ? 's' : ''}.
                                {workers.queued_jobs > 0 && ` ${workers.queued_jobs} job${workers.queued_jobs !== 1 ? 's are' : ' is'} queued.`}
                            </AlertDescription>
                        </Alert>
                    )}
                </CardContent>
            )}
        </Card>
    );
}
