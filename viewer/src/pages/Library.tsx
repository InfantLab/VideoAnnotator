import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { useToast } from '@/hooks/use-toast';
import {
  getJobDatasetIndex,
  getLegacyDatasetsDirName,
  getRootDirHandle,
  resolveDatasetsDir,
  setRootDirHandle,
  type JobDatasetIndexEntry
} from '@/lib/localLibrary/libraryStore';
import { DEMO_JOB_ID_PREFIX, installBundledDemoDataset } from '@/lib/localLibrary/installDemoDataset';

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

  const installDemo = async () => {
    setIsInstallingDemo(true);
    try {
      const root = await getRootDirHandle();
      if (!root) {
        toast({
          title: 'Choose a library folder first',
          description: 'Pick a folder where demo data should be saved.',
          variant: 'destructive'
        });
        return;
      }

      const result = await installBundledDemoDataset({ rootDir: root });
      toast({
        title: 'Demo dataset installed',
        description: `Saved under “${result.entry.folderName}”.`,
        duration: 5000
      });
      await refresh();
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : String(error);
      toast({
        title: 'Demo install failed',
        description: message,
        variant: 'destructive',
        duration: 8000
      });
    } finally {
      setIsInstallingDemo(false);
    }
  };

  return (
    <div className="container mx-auto p-6 max-w-5xl">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">Local Library</h1>
        <p className="text-muted-foreground mt-2">
          The viewer works from local files: a video file plus annotation outputs. This page manages the folder where
          downloaded job results are stored for fast reopening.
        </p>
      </div>

      <Card className="p-4 mb-6">
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <div>
            <div className="font-semibold">Library folder</div>
            <div className="text-sm text-muted-foreground">
              {rootName ? (
                <span>
                  Using <span className="font-medium">{rootName}</span>
                  {usingLegacySubdir ? ` / ${legacyDirName}` : ''}
                </span>
              ) : (
                <span>No library folder selected yet.</span>
              )}
            </div>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" onClick={() => void refresh()} disabled={isLoading}>
              Refresh
            </Button>
            <Button onClick={() => void pickRootFolder()}>Choose Folder</Button>
            <Button
              variant="outline"
              onClick={() => void installDemo()}
              disabled={!rootName || isInstallingDemo}
              title={!rootName ? 'Choose a library folder first' : undefined}
            >
              {isInstallingDemo ? 'Installing demo…' : 'Install demo dataset'}
            </Button>
          </div>
        </div>
      </Card>

      <Card className="p-4">
        <div className="flex items-center justify-between mb-4">
          <div className="font-semibold">Downloaded datasets</div>
          <div className="text-sm text-muted-foreground">{rows.length} total</div>
        </div>

        {rows.length === 0 ? (
          <div className="text-sm text-muted-foreground">
            Nothing here yet. When you click “View” on a completed job, the artifacts will be downloaded once and saved
            into this library.
          </div>
        ) : (
          <div className="space-y-2">
            {rows.map(({ jobId, entry }) => (
              <div key={jobId} className="flex items-center justify-between gap-4 border rounded-md p-3">
                <div className="min-w-0">
                  <div className="font-medium truncate">
                    {jobId.startsWith(DEMO_JOB_ID_PREFIX) ? 'Demo dataset' : `Job ${jobId}`}
                  </div>
                  <div className="text-sm text-muted-foreground truncate">
                    Folder: {entry.folderName}
                    {entry.videoFileName ? ` • Video: ${entry.videoFileName}` : ''}
                    {entry.createdAt ? ` • Saved: ${new Date(entry.createdAt).toLocaleString()}` : ''}
                  </div>
                </div>
                <div className="flex gap-2 shrink-0">
                  <Button variant="outline" onClick={() => navigate(`/view/${jobId}`)}>
                    Open
                  </Button>
                  {!jobId.startsWith(DEMO_JOB_ID_PREFIX) && (
                    <Button onClick={() => navigate(`/create/jobs/${jobId}`)}>Job</Button>
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
