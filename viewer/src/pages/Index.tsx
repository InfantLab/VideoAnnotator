import { useEffect, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { ArrowLeft, Loader2, Sparkles } from 'lucide-react';

import { FileUploader, type LoadedFiles } from '@/components/FileUploader';
import { PageHelp } from '@/components/PageHelp';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
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

  return (
    <div className="container mx-auto max-w-4xl space-y-6 px-6 py-8">
      <Button variant="ghost" size="sm" asChild>
        <Link to="/">
          <ArrowLeft className="mr-2 h-4 w-4" />
          Home
        </Link>
      </Button>
      <div className="space-y-2">
        <h1 className="text-3xl font-bold">Open files from this computer</h1>
        <PageHelp page="videoPage">
          A video and its result files (from VideoAnnotator or another tool), shown on the Video page. Nothing is sent
          anywhere.
        </PageHelp>
      </div>
      {problem && (
        <p className="text-sm text-destructive" role="alert">
          {problem}
        </p>
      )}
      <FileUploader onVideoLoad={() => {}} onAnnotationLoad={() => {}} onFilesLoaded={setLoaded} />
      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Demos</h2>
        <div className="grid gap-3 sm:grid-cols-2">
          {DEMOS.map((d) => (
            <Card key={d.key}>
              <CardHeader>
                <CardTitle className="text-base">{d.title}</CardTitle>
                <CardDescription>{d.description}</CardDescription>
              </CardHeader>
              <CardContent>
                <Button size="sm" onClick={() => setParams({ demo: d.key })}>
                  <Sparkles className="mr-2 h-4 w-4" />
                  Open the demo
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>
    </div>
  );
};

export default Index;
