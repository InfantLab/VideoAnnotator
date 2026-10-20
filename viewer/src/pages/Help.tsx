import { Link } from 'react-router-dom';
import { ExternalLink, FolderOpen, Play, Sparkles } from 'lucide-react';

import { PageHelp } from '@/components/PageHelp';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { DEMOS } from '@/lib/demos';
import { docsUrl, type DocsPage } from '@/lib/docsLinks';
import { GLOSSARY } from '@/lib/glossary';

const SECTIONS: { page: DocsPage; title: string; about: string }[] = [
  { page: 'firstRun', title: 'Your first run', about: 'A folder of videos to one results table.' },
  { page: 'videos', title: 'The Videos page', about: 'Every video, by folder and project, with its results.' },
  { page: 'projects', title: 'Projects', about: 'Grouping videos for a study, whenever you like.' },
  { page: 'projectSettings', title: 'Settings a project remembers', about: 'What the next run starts from, and why.' },
  { page: 'workflows', title: 'Workflows', about: 'Settings saved to reuse and share.' },
  { page: 'videoPage', title: 'The Video page', about: 'One video with every result as a track, and your notes.' },
  { page: 'ask', title: 'Ask AI', about: 'Asking a vision model about a video.' },
  { page: 'table', title: 'The results table', about: 'Its columns, for R or Python.' },
];

/** What VideoAnnotator does, its words, where to start, and the docs (spec 029 FR-027). */
export default function Help() {
  return (
    <div className="container mx-auto max-w-4xl space-y-8 px-6 py-8">
      <div className="space-y-3">
        <h1 className="text-3xl font-bold">Help</h1>
        <PageHelp page="help">How the viewer is organised, and where each thing is explained in full.</PageHelp>
        <p>
          VideoAnnotator finds things in your videos: who speaks and what they say, faces and their expressions,
          people and how they move, scene changes, and answers to questions you ask a vision model. It runs on a
          server you control, so your videos never leave it. You look at the results here, and take them out as one
          table for your statistics, ELAN files, and a methods paragraph for your paper.
        </p>
      </div>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold">Where to start</h2>
        <div className="flex flex-wrap gap-2">
          <Button asChild>
            <Link to="/ask">
              <Sparkles className="mr-2 h-4 w-4" />
              Ask AI about a video
            </Link>
          </Button>
          <Button asChild variant="outline">
            <Link to="/run">
              <Play className="mr-2 h-4 w-4" />
              Run pipelines on videos
            </Link>
          </Button>
          <Button asChild variant="outline">
            <Link to="/viewer">
              <FolderOpen className="mr-2 h-4 w-4" />
              Open files from this computer
            </Link>
          </Button>
        </div>
        <p className="text-sm text-muted-foreground">
          No server yet? Open a demo:{' '}
          {DEMOS.map((d) => (
            <Link key={d.key} to={`/viewer?demo=${d.key}`} className="underline">
              {d.title}
            </Link>
          ))}
          .
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold">The words</h2>
        <dl className="grid gap-x-6 gap-y-2 sm:grid-cols-[9rem_1fr]" data-testid="glossary">
          {Object.entries(GLOSSARY).map(([word, meaning]) => (
            <div key={word} className="contents">
              <dt className="font-medium capitalize">{word}</dt>
              <dd className="text-muted-foreground">{meaning}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section className="space-y-3">
        <h2 className="text-xl font-semibold">In the docs</h2>
        <div className="grid gap-3 sm:grid-cols-2">
          {SECTIONS.map((s) => (
            <Card key={s.page}>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">
                  <a href={docsUrl(s.page)} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 hover:underline">
                    {s.title}
                    <ExternalLink className="h-3 w-3" aria-hidden="true" />
                  </a>
                </CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground">{s.about}</CardContent>
            </Card>
          ))}
        </div>
      </section>
    </div>
  );
}
