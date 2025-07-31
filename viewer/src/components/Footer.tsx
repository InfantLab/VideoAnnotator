import { Github, ExternalLink } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { VERSION, GITHUB_URL, APP_NAME } from '@/utils/version';

export const Footer = () => {
  return (
    <footer className="border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto px-6 py-6">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          {/* Version Info */}
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <span>{APP_NAME}</span>
            <span className="px-2 py-1 bg-primary/10 text-primary rounded text-xs font-mono">
              v{VERSION}
            </span>
          </div>

          {/* GitHub Link */}
          <Button
            variant="ghost"
            size="sm"
            asChild
            className="text-muted-foreground hover:text-foreground"
          >
            <a
              href={GITHUB_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2"
            >
              <Github className="w-4 h-4" />
              <span>View on GitHub</span>
              <ExternalLink className="w-3 h-3" />
            </a>
          </Button>
        </div>

        {/* Copyright */}
        <div className="mt-4 pt-4 border-t text-center text-xs text-muted-foreground">
          <p>
            © {new Date().getFullYear()} InfantLab. 
            Advanced multimodal video annotation analysis tool.
          </p>
        </div>
      </div>
    </footer>
  );
};
