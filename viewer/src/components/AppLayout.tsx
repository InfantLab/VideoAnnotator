import type { ReactNode } from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Footer } from '@/components/Footer';
import { ConnectionStatus } from '@/components/ConnectionStatus';
import { ActivityIndicator } from '@/components/ActivityIndicator';
import { ConnectionErrorBanner } from '@/components/ConnectionErrorBanner';
import { useServerCapabilitiesContext } from '@/contexts/ServerCapabilitiesContext';
import { NEEDS_KEY_MESSAGE } from '@/api/handleError';
import { useConnection } from '@/hooks/useConnection';
import { isCorsOrNetworkError } from '@/lib/connectionUtils';
import { APP_NAME } from '@/utils/version';
import { HelpCircle, Home, KeyRound, Layers, Settings, Sparkles, Video } from 'lucide-react';
import vavIcon from '@/assets/v-a-v.icon.png';

// Spec 029 FR-018: four places. New run opens from Run… on what it runs on.
const navItems = [
  { path: '/home', label: 'Home', icon: Home },
  { path: '/videos', label: 'Videos', icon: Video },
  { path: '/ask', label: 'Ask AI', icon: Sparkles },
  { path: '/pipelines', label: 'Pipelines', icon: Layers },
];

/** The nav around a page: the routes' Outlet, or `children` for a page shown outside them. */
export const AppLayout = ({ children }: { children?: ReactNode }) => {
  const location = useLocation();
  const { error, refresh, capabilities, isLoading } = useServerCapabilitiesContext();
  const { state } = useConnection();
  // Home and Settings say it in their own words; the banner would say it twice.
  const showsOwnConnection =
    ['/', '/home'].includes(location.pathname) || location.pathname.startsWith('/settings');

  const apiUrl =
    localStorage.getItem('videoannotator_api_url') ||
    import.meta.env.VITE_API_BASE_URL ||
    '';

  const showConnectionError =
    error && !isLoading && !capabilities && isCorsOrNetworkError(error);

  const isActive = (path: string) => {
    if (path === '/home') return location.pathname === '/home' || location.pathname === '/';
    if (path === '/videos') return location.pathname.startsWith('/videos') || location.pathname === '/run';
    return location.pathname.startsWith(path);
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Top Navigation Bar */}
      <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50">
        <div className="container mx-auto px-4">
          <div className="flex h-14 items-center justify-between gap-2">
            {/* Left: Logo + App Name */}
            <Link to="/home" aria-label={APP_NAME} className="flex shrink-0 items-center gap-2 hover:opacity-80 transition-opacity">
              <img src={vavIcon} alt="" className="w-7 h-7" />
              <span className="hidden whitespace-nowrap text-lg font-semibold xl:inline">{APP_NAME}</span>
            </Link>

            {/* Center: Navigation Links */}
            <nav className="flex min-w-0 items-center gap-1">
              {navItems.map(({ path, label, icon: Icon }) => (
                <Link key={path} to={path}>
                  <Button
                    variant={isActive(path) ? 'default' : 'ghost'}
                    size="sm"
                    title={label}
                    className="flex items-center gap-2"
                  >
                    <Icon className="h-4 w-4" />
                    <span className="hidden sm:inline">{label}</span>
                  </Button>
                </Link>
              ))}
            </nav>

            {/* Right: Settings + Token Status */}
            <div className="flex shrink-0 items-center gap-1">
              <ActivityIndicator />
              <ConnectionStatus />
              <Link to="/help">
                <Button variant={isActive('/help') ? 'default' : 'ghost'} size="sm" title="Help" aria-label="Help">
                  <HelpCircle className="h-4 w-4" />
                  <span className="hidden lg:inline ml-1">Help</span>
                </Button>
              </Link>
              <Link to="/settings">
                <Button
                  variant={isActive('/settings') ? 'default' : 'ghost'}
                  size="sm"
                  title="Settings"
                  aria-label="Settings"
                >
                  <Settings className="h-4 w-4" />
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </header>

      {/* Connection Error Banner */}
      {showConnectionError && (
        <div className="container mx-auto px-4 pt-4">
          <ConnectionErrorBanner error={error} apiUrl={apiUrl} onRetry={refresh} />
        </div>
      )}

      {(state === 'needs-key' || state === 'key-refused') && !showsOwnConnection && (
        <div className="container mx-auto px-4 pt-4">
          <div
            className="flex flex-wrap items-center gap-3 rounded-md border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm"
            data-testid="needs-key-banner"
          >
            <KeyRound className="h-4 w-4 shrink-0 text-amber-500" />
            <span className="flex-1">
              {state === 'needs-key' ? NEEDS_KEY_MESSAGE : "The server didn't accept your key. Ask whoever runs it for a new one."}
            </span>
            <Button asChild size="sm">
              <Link to="/settings">Connect with your key</Link>
            </Button>
          </div>
        </div>
      )}

      {/* Page Content */}
      <main className="flex-1">
        {children ?? <Outlet />}
      </main>

      <Footer />
    </div>
  );
};
