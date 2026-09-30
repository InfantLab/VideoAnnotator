import { useEffect, useMemo, useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { useToast } from '@/hooks/use-toast';
import {
  getJobDatasetIndex,
  getLegacyDatasetsDirName,
  getRootDirHandle,
  resolveDatasetsDir,
  setRootDirHandle,
  type JobDatasetIndexEntry
} from '@/lib/localLibrary/libraryStore';
import { DEMO_JOB_ID_PREFIX, DEMO_LABELS, getDemoLabel, installAllBundledDemos } from '@/lib/localLibrary/installDemoDataset';
import { createCopyAction } from '@/lib/toastHelpers';
import {
  FolderOpen, RefreshCw, Download, Sparkles, ArrowRight,
} from 'lucide-react';

type Row = {
  jobId: string;
  entry: JobDatasetIndexEntry;
};

const Library = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [rootName, setRootName] = useState<string | null>(null);
  const [rows, setRows] = useState<Row[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isInstallingDemo, setIsInstallingDemo] = useState(false);
  const [demoProgress, setDemoProgress] = useState<string | null>(null);

  const legacyDirName = useMemo(() => getLegacyDatasetsDirName(), []);
  const [usingLegacySubdir, setUsingLegacySubdir] = useState(false);

  const refresh = async () => {
    setIsLoading(true);
    try {
      const root = await getRootDirHandle();
      setRootName(root?.name ?? null);

      if (root) {
        const resolved = await resolveDatasetsDir(root);
        setUsingLegacySubdir(resolved !== root);
      } else {
        setUsingLegacySubdir(false);
      }

      const index = await getJobDatasetIndex();
      const nextRows = Object.entries(index)
        .map(([jobId, entry]) => ({ jobId, entry }))
        .sort((a, b) => b.entry.createdAt.localeCompare(a.entry.createdAt));
      setRows(nextRows);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  const pickRootFolder = async () => {
    if (!('showDirectoryPicker' in window)) return;

    // @ts-expect-error - File System Access API is not always present in TS lib.dom
    const handle: FileSystemDirectoryHandle = await window.showDirectoryPicker({ mode: 'readwrite' });
    await setRootDirHandle(handle);
    await refresh();
  };

  const installDemos = async () => {
    setIsInstallingDemo(true);
    setDemoProgress(null);
    try {
      const result = await installAllBundledDemos((msg) => setDemoProgress(msg));

      if (result.installed === 0 && result.failed.length === 0) {
        toast({ title: 'Demo datasets already installed', description: `All ${result.skipped} demos are up to date.`, duration: 4000 });
      } else if (result.failed.length > 0) {
        const detail = result.failed.map(f => `${DEMO_LABELS[f.key] ?? f.key}: ${f.error}`).join('\n');
        const fullText = `Demo install errors\n\nInstalled ${result.installed}, failed ${result.failed.length}.\n${detail}`;
        toast({ title: 'Some demos failed to install', description: `Installed ${result.installed}, failed ${result.failed.length}.`, variant: 'destructive', duration: 12000, action: createCopyAction(fullText) });
      } else {
        toast({ title: 'Demo datasets installed', description: `Installed ${result.installed} demo dataset${result.installed === 1 ? '' : 's'}.`, duration: 5000 });
      }
      await refresh();
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : String(error);
      toast({ title: 'Demo install failed', description: message, variant: 'destructive', duration: 10000, action: createCopyAction(`Demo install failed\n\n${message}`) });
    } finally {
      setIsInstallingDemo(false);
      setDemoProgress(null);
    }
  };

  return (
    <div className="container mx-auto p-6 max-w-5xl">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">Library</h1>
        <p className="text-muted-foreground mt-2">
          Browse and open your video annotation datasets.
        </p>
      </div>

      {/* Library Folder - Prominent Display */}
      <Card className="p-5 mb-6">
        <div className="flex items-start gap-4 flex-wrap">
          <div className={`w-12 h-12 rounded-lg flex items-center justify-center flex-shrink-0 ${rootName ? 'bg-blue-100 text-blue-600' : 'bg-gray-100 text-gray-400'}`}>
            <FolderOpen className="w-6 h-6" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-sm font-medium text-muted-foreground mb-1">Library Folder</div>
            {rootName ? (
              <>
                <div className="text-lg font-semibold truncate">{rootName}</div>
                {usingLegacySubdir && (
                  <Badge variant="secondary" className="mt-1 text-xs">
                    Using legacy subdirectory: {legacyDirName}
                  </Badge>
                )}
              </>
            ) : (
              <div className="text-lg text-muted-foreground">No folder selected</div>
            )}
          </div>
          <div className="flex gap-2 flex-shrink-0">
            <Button
              variant="outline"
              size="sm"
              onClick={() => void refresh()}
              disabled={isLoading}
            >
              <RefreshCw className={`h-4 w-4 mr-1 ${isLoading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
            <Button variant="outline" size="sm" onClick={() => void pickRootFolder()}>
              Choose Folder
            </Button>
          </div>
        </div>
      </Card>

      {/* Demo Datasets */}
      <Card className="p-4 mb-6">
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <div>
            <div className="font-semibold">Demo Datasets</div>
            <div className="text-sm text-muted-foreground">
              Install bundled sample datasets to explore the viewer without a server.
            </div>
          </div>
          <Button
            onClick={() => void installDemos()}
            disabled={isInstallingDemo}
            size="sm"
            className="gap-2"
          >
            <Download className="h-4 w-4" />
            {isInstallingDemo ? (demoProgress ?? 'Installing…') : 'Install Demos'}
          </Button>
        </div>
      </Card>

      {/* Datasets List */}
      <Card className="p-4">
        <div className="flex items-center justify-between mb-4">
          <div className="font-semibold">Downloaded Datasets</div>
          <div className="text-sm text-muted-foreground">{rows.length} total</div>
        </div>

        {rows.length === 0 ? (
          <div className="text-center py-8">
            <div className="text-muted-foreground mb-4">
              {rootName
                ? 'No datasets found. View a completed job to download artifacts into your library.'
                : 'Choose a library folder to get started, or install the demo datasets.'}
            </div>
            <Link to="/getting-started">
              <Button variant="outline" size="sm" className="gap-2">
                <Sparkles className="h-4 w-4" />
                Getting Started Guide
                <ArrowRight className="h-4 w-4" />
              </Button>
            </Link>
          </div>
        ) : (
          <div className="space-y-2">
            {rows.map(({ jobId, entry }) => (
              <div key={jobId} className="flex items-center justify-between gap-4 border rounded-md p-3">
                <div className="min-w-0">
                  <div className="font-medium truncate">
                    {getDemoLabel(jobId) ? `Demo: ${getDemoLabel(jobId)}` : `Job ${jobId}`}
                  </div>
                  <div className="text-sm text-muted-foreground truncate">
                    {entry.videoFileName ? `Video: ${entry.videoFileName}` : `Folder: ${entry.folderName}`}
                    {entry.createdAt ? ` · Saved: ${new Date(entry.createdAt).toLocaleString()}` : ''}
                  </div>
                </div>
                <div className="flex gap-2 shrink-0">
                  <Button variant="outline" onClick={() => navigate(`/view/${jobId}`)}>
                    Open
                  </Button>
                  {!jobId.startsWith(DEMO_JOB_ID_PREFIX) && (
                    <Button onClick={() => navigate(`/jobs/${jobId}`)}>Job</Button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
};

export default Library;
