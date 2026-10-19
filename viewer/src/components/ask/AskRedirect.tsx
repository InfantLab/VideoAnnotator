import { Navigate, useLocation } from 'react-router-dom';

/** /workbench and /prompts land on Ask AI, keeping their question and model (Principle V). */
export function AskRedirect({ history = false }: { history?: boolean }) {
  const { search } = useLocation();
  const from = new URLSearchParams(search);
  const to = new URLSearchParams();
  for (const key of ['prompt', 'model', 'video']) {
    const value = from.get(key);
    if (value) to.set(key, value);
  }
  if (history) to.set('history', 'open');
  const query = to.toString();
  return <Navigate to={`/ask${query ? `?${query}` : ''}`} replace />;
}
