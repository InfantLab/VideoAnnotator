import { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Loader2 } from 'lucide-react';

import { AppLayout } from '@/components/AppLayout';

import { FileUploader, type LoadedFiles } from '@/components/FileUploader';
import { PageHelp } from '@/components/PageHelp';
import { LocalVideo } from '@/components/video/LocalVideo';
import { DEMOS, isDemoKey, loadDemoFiles } from '@/lib/demos';

/**
 * Standalone mode: results you have as files, or a demo, on the Video page,
 * with no server (spec 028; spec 029 R14: `/viewer?demo=<key>`).
 */
const Index = () => {
  const [params, setParams] = useSearchParams();
  const demo = params.get('demo');
  const [loaded, setLoaded] = useState<LoadedFiles | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  useEffect(() => {
    if (!isDemoKey(demo)) return;
    let cancelled = false;
    setProblem(null);
    loadDemoFiles(demo)
      .then((files) => !cancelled && setLoaded(files))
      .catch((error) => !cancelled && setProblem(error instanceof Error ? error.message : String(error)));
    return () => {
      cancelled = true;
    };
  }, [demo]);

  const close = () => {
    setLoaded(null);
    setParams({}, { replace: true });
  };

  if (loaded) return <LocalVideo loaded={loaded} onClose={close} />;
  if (isDemoKey(demo) && !problem) {
    return (
      <p className="flex items-center gap-2 p-8 text-sm text-muted-foreground" data-testid="demo-loading">
        <Loader2 className="h-4 w-4 animate-spin" /> Opening the demo…
      </p>
    );
  }

  // The nav stays while choosing files (UAT #60); the Video page itself is full screen.
  return (
    <AppLayout>
      <div className="container mx-auto max-w-3xl space-y-6 px-6 py-8">
        <div className="space-y-2">
          <h1 className="text-3xl font-bold">Open files from your computer</h1>
          <PageHelp page="videoPage">
            A video and its result files (from VideoAnnotator or another tool), shown on the Video page. Nothing is
            sent anywhere.
          </PageHelp>
        </div>
        {problem && (
          <p className="text-sm text-destructive" role="alert">
            {problem}
          </p>
        )}
        <FileUploader onVideoLoad={() => {}} onAnnotationLoad={() => {}} onFilesLoaded={setLoaded} />
        <p className="text-sm text-muted-foreground">
          No files to hand? Open a demo:{' '}
          {DEMOS.map((d, i) => (
            <span key={d.key}>
              {i > 0 && ', '}
              <button type="button" className="underline" onClick={() => setParams({ demo: d.key })} title={d.description}>
                {d.title}
              </button>
            </span>
          ))}
          .
        </p>
      </div>
    </AppLayout>
  );
};

export default Index;
