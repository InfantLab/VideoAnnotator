import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Loader2, RefreshCw, Eye, Play } from "lucide-react";
import { Link } from "react-router-dom";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import vavIcon from "@/assets/v-a-v.icon.png";
import { JobCancelButton } from "@/components/JobCancelButton";
import { canCancelJob } from "@/hooks/useJobCancellation";
import type { JobStatus } from "@/types/api";

const CreateJobs = () => {
  const [page, setPage] = useState(1);
  const perPage = 10;

  const {
    data: jobsData,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ["jobs", page],
    queryFn: () => apiClient.getJobs(page, perPage),
    refetchInterval: 5000, // Refetch every 5 seconds for real-time updates
  });

  const getStatusBadge = (status: string) => {
    const statusMap = {
      pending: { variant: "secondary" as const, color: "text-yellow-600" },
      running: { variant: "default" as const, color: "text-blue-600" },
      completed: { variant: "default" as const, color: "text-green-600" },
      failed: { variant: "destructive" as const, color: "text-red-600" },
      cancelled: { variant: "secondary" as const, color: "text-gray-600" },
      cancelling: { variant: "secondary" as const, color: "text-orange-600" },
    };

    const config = statusMap[status as keyof typeof statusMap] || statusMap.pending;
    
    return (
      <Badge variant={config.variant} className={config.color}>
        {status.toUpperCase()}
      </Badge>
    );
  };

  const formatDuration = (seconds: number | null) => {
    if (!seconds) return "N/A";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const formatFileSize = (bytes: number | null) => {
    if (!bytes) return "N/A";
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(1)} MB`;
  };

  if (error) {
    return (
      <div className="text-center py-8">
        <p className="text-red-600 mb-4">
          Failed to load jobs: {error instanceof Error ? error.message : "Unknown error"}
        </p>
        <Button onClick={() => refetch()} variant="outline">
          <RefreshCw className="h-4 w-4 mr-2" />
          Retry
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-3">
          <img src={vavIcon} alt="VideoAnnotator" className="h-8 w-8" />
          <div>
            <h2 className="text-2xl font-bold">Annotation Jobs</h2>
            <p className="text-gray-600">Monitor and manage your annotation jobs</p>
          </div>
        </div>
        <div className="flex gap-2">
          <Button onClick={() => refetch()} variant="outline" size="sm">
            <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Link to="/create/new">
            <Button>
              <Play className="h-4 w-4 mr-2" />
              New Job
            </Button>
          </Link>
        </div>
      </div>

      {/* Jobs Table */}
      {isLoading && !jobsData ? (
        <div className="flex justify-center py-8">
          <Loader2 className="h-8 w-8 animate-spin" />
        </div>
      ) : (
        <Card>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Job ID</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Video</TableHead>
                <TableHead>Duration</TableHead>
                <TableHead>Size</TableHead>
                <TableHead>Pipelines</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {jobsData?.jobs?.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} className="text-center py-8">
                    <p className="text-gray-500 mb-4">No jobs found</p>
                    <Link to="/create/new">
                      <Button>Create your first job</Button>
                    </Link>
                  </TableCell>
                </TableRow>
              ) : (
                jobsData?.jobs?.map((job) => (
                  <TableRow key={job.id}>
                    <TableCell className="font-mono text-sm">
                      {job.id.slice(0, 8)}...
                    </TableCell>
                    <TableCell>
                      {getStatusBadge(job.status)}
                    </TableCell>
                    <TableCell className="max-w-[200px] truncate">
                      {job.video_filename || "N/A"}
                    </TableCell>
                    <TableCell>
                      {formatDuration(job.video_duration_seconds)}
                    </TableCell>
                    <TableCell>
                      {formatFileSize(job.video_size_bytes)}
                    </TableCell>
                    <TableCell>
                      <div className="flex flex-wrap gap-1">
                        {job.selected_pipelines?.slice(0, 2).map((pipeline) => (
                          <Badge key={pipeline} variant="outline" className="text-xs">
                            {pipeline}
                          </Badge>
                        ))}
                        {job.selected_pipelines && job.selected_pipelines.length > 2 && (
                          <Badge variant="outline" className="text-xs">
                            +{job.selected_pipelines.length - 2}
                          </Badge>
                        )}
                      </div>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        {canCancelJob(job.status as JobStatus) && (
                          <JobCancelButton 
                            jobId={job.id} 
                            jobStatus={job.status as JobStatus}
                            size="sm"
                            variant="outline"
                          />
                        )}
                        <Link to={`/create/jobs/${job.id}`}>
                          <Button variant="outline" size="sm">
                            <Eye className="h-4 w-4 mr-1" />
                            View
                          </Button>
                        </Link>
                      </div>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </Card>
      )}

      {/* Pagination */}
      {jobsData && jobsData.total > perPage && (
        <div className="flex justify-center gap-2">
          <Button
            variant="outline"
            onClick={() => setPage(p => Math.max(1, p - 1))}
            disabled={page === 1}
          >
            Previous
          </Button>
          <span className="px-4 py-2 text-sm">
            Page {page} of {Math.ceil(jobsData.total / perPage)}
          </span>
          <Button
            variant="outline"
            onClick={() => setPage(p => p + 1)}
            disabled={page >= Math.ceil(jobsData.total / perPage)}
          >
            Next
          </Button>
        </div>
      )}
    </div>
  );
};

export default CreateJobs;