import { Badge } from '@/components/ui/badge';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { AlertCircle } from 'lucide-react';

const STATUS_CLASSES: Record<string, string> = {
  pending: 'bg-yellow-100 text-yellow-800 hover:bg-yellow-100 border-yellow-200',
  running: 'bg-blue-100 text-blue-800 hover:bg-blue-100 border-blue-200',
  completed: 'bg-green-100 text-green-800 hover:bg-green-100 border-green-200',
  failed: 'bg-red-100 text-red-800 hover:bg-red-100 border-red-200',
  cancelled: 'bg-gray-100 text-gray-800 hover:bg-gray-100 border-gray-200',
  cancelling: 'bg-orange-100 text-orange-800 hover:bg-orange-100 border-orange-200',
};

const PARTIAL_SUCCESS_CLASS =
  'bg-orange-100 text-orange-800 hover:bg-orange-100 border-orange-200';

export function JobStatusBadge({
  status,
  errorMessage,
  className = '',
}: {
  status: string;
  errorMessage?: string | null;
  className?: string;
}) {
  // A job that completed but carries an error message succeeded only partially.
  const isPartialSuccess = status === 'completed' && !!errorMessage;
  const colours = isPartialSuccess
    ? PARTIAL_SUCCESS_CLASS
    : STATUS_CLASSES[status] ?? STATUS_CLASSES.pending;

  const badge = (
    <Badge variant="outline" className={`${colours} ${className}`}>
      {status.toUpperCase()}
      {(isPartialSuccess || (status === 'failed' && errorMessage)) && (
        <AlertCircle className="ml-1 h-3 w-3 inline" />
      )}
    </Badge>
  );

  if ((status === 'failed' || isPartialSuccess) && errorMessage) {
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>{badge}</TooltipTrigger>
          <TooltipContent className="max-w-xs">
            <p className="font-semibold">{isPartialSuccess ? 'Completed with errors:' : 'Error:'}</p>
            <p>{errorMessage}</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }

  return badge;
}
