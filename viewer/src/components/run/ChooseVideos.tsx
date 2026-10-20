import { useRef, useState } from 'react';
import { FolderOpen, FolderKanban, History, Upload } from 'lucide-react';

import { NoFolderAccess, ServerFolderPicker, type ServerFolderSelection } from '@/components/ServerFolderPicker';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useIngestAccess } from '@/hooks/useIngestAccess';
import { useVideos } from '@/hooks/useVideos';
import type { RunTarget } from '@/lib/runTarget';
import { videosWord } from '@/lib/runs';

export type Chosen = { target: RunTarget } | { files: File[] };

const VIDEO_TYPES = 'video/*,.mp4,.avi,.mov,.mkv,.webm,.wmv,.flv,.m4v';

/** New run without a target: which videos (spec 029 R9). */
export function ChooseVideos({ onChoose }: { onChoose: (chosen: Chosen) => void }) {
  const access = useIngestAccess();
  const [tab, setTab] = useState('folder');
  const [folder, setFolder] = useState<ServerFolderSelection | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [search, setSearch] = useState('');
  const fileInput = useRef<HTMLInputElement>(null);
  const known = useVideos({}, tab === 'project' || tab === 'video');

  const useFolder = () => {
    if (!folder) return;
    if (folder.files?.length) {
      const base = folder.path.replace(/[\\/]+$/, '');
      onChoose({
        target: {
          kind: 'videos',
          paths: folder.files.map((f) => `${base}/${f}`),
          label: `${videosWord(folder.files.length)} in ${folder.path}`,
        },
      });
    } else {
      onChoose({ target: { kind: 'folder', path: folder.path, recursive: folder.recursive } });
    }
  };

  const matches = (known.data?.videos ?? []).filter(
    (v) => Object.keys(v.results).length > 0 && v.name.toLowerCase().includes(search.trim().toLowerCase()),
  );

  return (
    <Tabs value={tab} onValueChange={setTab} className="space-y-4">
      <TabsList className="flex-wrap">
        <TabsTrigger value="folder">
          <FolderOpen className="mr-2 h-4 w-4" />A shared folder
        </TabsTrigger>
        <TabsTrigger value="upload">
          <Upload className="mr-2 h-4 w-4" />This computer
        </TabsTrigger>
        <TabsTrigger value="project">
          <FolderKanban className="mr-2 h-4 w-4" />A project
        </TabsTrigger>
        <TabsTrigger value="video">
          <History className="mr-2 h-4 w-4" />A video you ran before
        </TabsTrigger>
      </TabsList>

      <TabsContent value="folder" className="space-y-3">
        {access.access && !access.canReadInPlace ? (
          <NoFolderAccess access={access.access} onUpload={() => setTab('upload')} />
        ) : (
          <>
            <ServerFolderPicker selection={folder} onSelect={setFolder} />
            <Button onClick={useFolder} disabled={!folder}>
              Use {folder ? videosWord(folder.files?.length ?? folder.videoCount) : 'this folder'}
            </Button>
          </>
        )}
      </TabsContent>

      <TabsContent value="upload" className="space-y-3">
        <p className="text-sm text-muted-foreground">
          Videos on this computer are uploaded to the server, then run there. Videos already on the server are quicker
          from a shared folder.
        </p>
        <input
          ref={fileInput}
          type="file"
          multiple
          accept={VIDEO_TYPES}
          className="hidden"
          data-testid="upload-input"
          onChange={(e) => {
            setFiles([...files, ...Array.from(e.target.files ?? [])]);
            e.target.value = '';
          }}
        />
        <Button variant="outline" onClick={() => fileInput.current?.click()}>
          <Upload className="mr-2 h-4 w-4" />
          Choose videos…
        </Button>
        {files.length > 0 && (
          <div className="space-y-2 text-sm">
            <p>{files.map((f) => f.name).join(', ')}</p>
            <div className="flex gap-2">
              <Button onClick={() => onChoose({ files })}>Use {videosWord(files.length)}</Button>
              <Button variant="ghost" onClick={() => setFiles([])}>
                Clear
              </Button>
            </div>
          </div>
        )}
      </TabsContent>

      <TabsContent value="project" className="space-y-2">
        {(known.data?.counts.projects ?? []).length === 0 && (
          <p className="text-sm text-muted-foreground">No projects yet. Make one on the Videos page by selecting videos.</p>
        )}
        <div className="flex flex-wrap gap-2">
          {(known.data?.counts.projects ?? []).map((p) => (
            <Button key={p.id} variant="outline" onClick={() => onChoose({ target: { kind: 'project', id: p.id, name: p.name } })}>
              {p.name} · {videosWord(p.count)}
            </Button>
          ))}
        </div>
      </TabsContent>

      <TabsContent value="video" className="space-y-2">
        <Input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Find a video by name" aria-label="Find a video by name" className="max-w-sm" />
        <ul className="max-h-72 space-y-1 overflow-auto text-sm">
          {matches.slice(0, 50).map((v) => (
            <li key={v.path}>
              <button
                type="button"
                className="w-full rounded px-2 py-1 text-left hover:bg-muted disabled:opacity-50"
                disabled={!v.available}
                title={v.available ? v.path : 'Not on the server any more'}
                onClick={() => onChoose({ target: { kind: 'videos', paths: [v.path], label: v.name } })}
              >
                {v.name} <span className="text-xs text-muted-foreground">{v.folder === 'uploads' ? 'Uploaded' : v.folder}</span>
              </button>
            </li>
          ))}
        </ul>
      </TabsContent>
    </Tabs>
  );
}
