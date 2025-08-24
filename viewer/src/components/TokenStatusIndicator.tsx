import { useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Popover, 
  PopoverContent, 
  PopoverTrigger 
} from '@/components/ui/popover';
import { 
  CheckCircle, 
  AlertTriangle, 
  Settings, 
  RefreshCw, 
  ExternalLink 
} from 'lucide-react';
import { Link } from 'react-router-dom';
import { useTokenStatus } from '@/hooks/useTokenStatus';

interface TokenStatusIndicatorProps {
  showDetails?: boolean;
  className?: string;
}

export function TokenStatusIndicator({ 
  showDetails = false, 
  className = '' 
}: TokenStatusIndicatorProps) {
  const { isValid, user, error, isLoading, refresh } = useTokenStatus();
  const [isOpen, setIsOpen] = useState(false);

  if (isLoading) {
    return (
      <Badge variant="secondary" className={className}>
        <RefreshCw className="h-3 w-3 mr-1 animate-spin" />
        Checking...
      </Badge>
    );
  }

  const statusBadge = (
    <Badge 
      variant={isValid ? "default" : "destructive"} 
      className={`cursor-pointer ${className}`}
    >
      {isValid ? (
        <CheckCircle className="h-3 w-3 mr-1" />
      ) : (
        <AlertTriangle className="h-3 w-3 mr-1" />
      )}
      {isValid ? 'API Connected' : 'API Issue'}
    </Badge>
  );

  if (!showDetails) {
    return statusBadge;
  }

  return (
    <Popover open={isOpen} onOpenChange={setIsOpen}>
      <PopoverTrigger asChild>
        {statusBadge}
      </PopoverTrigger>
      <PopoverContent className="w-80" align="end">
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="font-medium">API Status</h4>
            <Button 
              variant="ghost" 
              size="sm" 
              onClick={refresh}
              disabled={isLoading}
            >
              <RefreshCw className={`h-3 w-3 ${isLoading ? 'animate-spin' : ''}`} />
            </Button>
          </div>

          {isValid ? (
            <Alert>
              <CheckCircle className="h-4 w-4 text-green-600" />
              <AlertDescription>
                <div className="space-y-1">
                  <p className="font-medium text-green-700">✅ Connected to VideoAnnotator API</p>
                  {user && <p className="text-sm">Authenticated as: {user}</p>}
                </div>
              </AlertDescription>
            </Alert>
          ) : (
            <Alert variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>
                <div className="space-y-2">
                  <p className="font-medium">❌ API Connection Issue</p>
                  <p className="text-sm">{error || 'Unable to connect to server'}</p>
                </div>
              </AlertDescription>
            </Alert>
          )}

          <div className="flex items-center gap-2 pt-2 border-t">
            <Link to="/create/settings">
              <Button 
                variant="outline" 
                size="sm" 
                onClick={() => setIsOpen(false)}
                className="flex items-center gap-1"
              >
                <Settings className="h-3 w-3" />
                Settings
              </Button>
            </Link>
            
            <Button 
              variant="link" 
              size="sm" 
              onClick={() => setIsOpen(false)}
              asChild
              className="p-0 h-auto"
            >
              <a 
                href="https://github.com/InfantLab/VideoAnnotator/blob/master/docs/CLIENT_TOKEN_SETUP_GUIDE.md"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1 text-xs"
              >
                Help <ExternalLink className="h-3 w-3" />
              </a>
            </Button>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}