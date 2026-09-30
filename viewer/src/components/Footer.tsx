import { Github, ExternalLink } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { VERSION, GITHUB_URL, APP_NAME } from '@/utils/version';

export const Footer = () => {
  return (
    <footer className="border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="px-4 py-2">
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          {/* Left: Version Info */}
          <div className="flex items-center gap-2">
            <span>{APP_NAME}</span>
            <span className="px-1.5 py-0.5 bg-primary/10 text-primary rounded font-mono">
              v{VERSION}
            </span>
          </div>

          {/* Center: Powered by VideoAnnotator */}
          <div className="flex items-center gap-2">
            <span>Powered by</span>
            <Button
              variant="ghost"
              size="sm"
              asChild
              className="text-muted-foreground hover:text-foreground h-6 px-2 text-xs font-medium"
            >
              <a
                href="https://github.com/InfantLab/VideoAnnotator"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1"
                title="VideoAnnotator processes videos to generate annotation data"
              >
                <span>🎬</span>
                <span>VideoAnnotator</span>
                <ExternalLink className="w-2.5 h-2.5" />
              </a>
            </Button>
          </div>

          {/* Right: Source and Docs */}
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="sm"
              asChild
              className="text-muted-foreground hover:text-foreground h-6 px-2 text-xs"
            >
              <a
                href="https://deepwiki.com/InfantLab/video-annotation-viewer"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1"
                title="AI-powered interactive codebase documentation"
              >
                <span>🤖</span>
                <span>Docs</span>
                <ExternalLink className="w-2.5 h-2.5" />
              </a>
            </Button>
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
                title="Source code and documentation for this viewer interface"
              >
                <Github className="w-3 h-3" />
                <span>Source</span>
                <ExternalLink className="w-2.5 h-2.5" />
              </a>
            </Button>
          </div>
        </div>
      </div>
    </footer>
  );
};
