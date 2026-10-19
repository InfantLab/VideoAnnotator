import { useState } from 'react';

import type { LoadedFiles } from '@/components/FileUploader';
import { VideoAnnotationViewer } from '@/components/VideoAnnotationViewer';
import { LocalVideo } from '@/components/video/LocalVideo';

/** Standalone mode: open files from this computer, then view them on the Video page. */
const Index = () => {
  const [loaded, setLoaded] = useState<LoadedFiles | null>(null);
  if (loaded) return <LocalVideo loaded={loaded} onClose={() => setLoaded(null)} />;
  return <VideoAnnotationViewer onLocalFiles={setLoaded} />;
};

export default Index;
