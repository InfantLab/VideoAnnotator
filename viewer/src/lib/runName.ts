/**
 * A run's name when the user doesn't give one: what it ran on, then what it
 * found ("expCP2 · PySceneDetect + CLIP", "peekaboo · pyannote community-1,
 * Whisper large-v3-turbo"). The date is shown beside a name, never folded into
 * it (viewer design "Names", UAT #41). The server makes the same name for runs
 * started without the viewer.
 */
import type { RunTarget } from '@/lib/runTarget';

const lastPart = (path: string) => path.replace(/[\\/]+$/, '').split(/[\\/]/).pop() ?? path;
const stem = (path: string) => lastPart(path).replace(/\.[^.]+$/, '');
const videos = (n: number) => `${n} ${n === 1 ? 'video' : 'videos'}`;

function subjectOfPaths(paths: string[]): string {
  if (paths.length === 1) return stem(paths[0]);
  const folders = new Set(paths.map((p) => p.replace(/[\\/][^\\/]*$/, '')));
  return folders.size === 1 && paths.length ? lastPart([...folders][0]) : videos(paths.length);
}

/** What the run is on: a video, a folder, a project, or a count. */
export function runSubject(target: RunTarget | null, paths: string[] | null, files: File[] | null): string {
  if (files) {
    const folder = (files[0] as File & { webkitRelativePath?: string })?.webkitRelativePath?.split('/')[0];
    return folder || (files.length === 1 ? stem(files[0].name) : videos(files.length));
  }
  if (!target) return '';
  switch (target.kind) {
    case 'project':
      return target.name;
    case 'folder':
      return lastPart(target.path);
    case 'rerun':
      return target.label;
    case 'videos':
      return paths?.length ? subjectOfPaths(paths) : target.label;
  }
}

/** "subject · what was found": up to two pipelines by name, more as a count. */
export function runName(subject: string, pipelineNames: string[]): string {
  if (!pipelineNames.length) return subject;
  const found = pipelineNames.length <= 2 ? pipelineNames.join(', ') : `${pipelineNames.length} pipelines`;
  return subject ? `${subject} · ${found}` : found;
}
