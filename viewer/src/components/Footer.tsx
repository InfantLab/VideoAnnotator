import { Github, ExternalLink } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { VERSION, GITHUB_URL, APP_NAME } from '@/utils/version';

export const Footer = () => {
  return (
    <footer className="border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="px-4 py-2">
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          {/* Compact Version Info */}
          <div className="flex items-center gap-2">
            <span>{APP_NAME}</span>
            <span className="px-1.5 py-0.5 bg-primary/10 text-primary rounded font-mono">
              v{VERSION}
            </span>
          </div>

          {/* Compact Links */}
          <div className="flex items-center gap-1">
            {/* VideoAnnotator Documentation */}
            <Button
              variant="ghost"
              size="sm"
              asChild
              className="text-muted-foreground hover:text-foreground h-6 px-2 text-xs"
            >
              <a
                href="https://github.com/InfantLab/VideoAnnotator"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1"
                title="VideoAnnotator generates the data files used by this viewer"
              >
                <span>🎬</span>
                <span>VideoAnnotator</span>
                <ExternalLink className="w-2.5 h-2.5" />
              </a>
            </Button>

            {/* GitHub Link */}
            <Button
              variant="ghost"
              size="sm"
              asChild
              className="text-muted-foreground hover:text-foreground h-6 px-2 text-xs"
            >
              <a
                href={GITHUB_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1"
              >
                <Github className="w-3 h-3" />
                <span>GitHub</span>
                <ExternalLink className="w-2.5 h-2.5" />
              </a>
            </Button>
          </div>
        </div>
      </div>
    </footer>
  );
};
