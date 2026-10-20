import { Link } from 'react-router-dom';
import { FolderOpen, PlugZap, Sparkles } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { DEMO_KEY } from '@/lib/demos';

/**
 * What a page needs a server for, how to connect, and what works without one
 * (spec 029 FR-026).
 */
export function NeedsServer({ what }: { what: string }) {
  return (
    <Card className="max-w-2xl" data-testid="needs-server">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-lg">
          <PlugZap className="h-5 w-5" aria-hidden="true" />
          This needs a VideoAnnotator server
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <p>{what}</p>
        <Button asChild>
          <Link to="/settings">Connect</Link>
        </Button>
        <div className="space-y-2 border-t pt-4">
          <p className="text-muted-foreground">Without one, you can still:</p>
          <div className="flex flex-wrap gap-2">
            <Button asChild variant="outline" size="sm">
              <Link to="/viewer">
                <FolderOpen className="mr-2 h-4 w-4" aria-hidden="true" />
                Open files from this computer
              </Link>
            </Button>
            <Button asChild variant="outline" size="sm">
              <Link to={`/viewer?demo=${DEMO_KEY}`}>
                <Sparkles className="mr-2 h-4 w-4" aria-hidden="true" />
                Demos
              </Link>
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
