import { Navigate } from 'react-router-dom';

import { useServerCapabilitiesContext } from '@/contexts/ServerCapabilitiesContext';
import { useVideos } from '@/hooks/useVideos';
import Home from '@/pages/Home';

/** `/`: Videos once the server has a finished result, else Home (spec 029 R18). */
export function RootRoute() {
  const { capabilities, isLoading } = useServerCapabilitiesContext();
  const { data, isLoading: listing, error } = useVideos({}, !!capabilities);
  if (isLoading || (capabilities && listing)) return <p className="p-8 text-sm text-muted-foreground">Loading…</p>;
  const finished = data?.videos.some((v) => Object.values(v.results).some((r) => r.state === 'done'));
  if (capabilities && !error && finished) return <Navigate to="/videos" replace />;
  return <Home />;
}
