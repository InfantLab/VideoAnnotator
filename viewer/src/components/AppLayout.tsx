import { Outlet, Link, useLocation } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Footer } from '@/components/Footer';
import { TokenStatusIndicator } from '@/components/TokenStatusIndicator';
import { ConnectionErrorBanner } from '@/components/ConnectionErrorBanner';
import { useServerCapabilitiesContext } from '@/contexts/ServerCapabilitiesContext';
import { isCorsOrNetworkError } from '@/lib/connectionUtils';
import { APP_NAME } from '@/utils/version';
import { BookOpen, Briefcase, MonitorPlay, Settings } from 'lucide-react';
import vavIcon from '@/assets/v-a-v.icon.png';

const navItems = [
  { path: '/library', label: 'Library', icon: BookOpen },
  { path: '/jobs', label: 'Jobs', icon: Briefcase },
  { path: '/viewer', label: 'View Files', icon: MonitorPlay },
];

export const AppLayout = () => {
  const location = useLocation();
  const { error, refresh, capabilities, isLoading } = useServerCapabilitiesContext();

  const apiUrl =
    localStorage.getItem('videoannotator_api_url') ||
    import.meta.env.VITE_API_BASE_URL ||
    '';

  const showConnectionError =
    error && !isLoading && !capabilities && isCorsOrNetworkError(error);

  const isActive = (path: string) => {
    if (path === '/') return location.pathname === '/';
    return location.pathname.startsWith(path);
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Top Navigation Bar */}
      <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50">
        <div className="container mx-auto px-4">
          <div className="flex items-center justify-between h-14">
            {/* Left: Logo + App Name */}
            <Link to="/" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
              <img src={vavIcon} alt="" className="w-7 h-7" />
              <span className="font-semibold text-lg hidden sm:inline">{APP_NAME}</span>
            </Link>

            {/* Center: Navigation Links */}
            <nav className="flex items-center gap-1">
              {navItems.map(({ path, label, icon: Icon }) => (
                <Link key={path} to={path}>
                  <Button
                    variant={isActive(path) ? 'default' : 'ghost'}
                    size="sm"
                    className="flex items-center gap-2"
                  >
                    <Icon className="h-4 w-4" />
                    <span className="hidden sm:inline">{label}</span>
                  </Button>
                </Link>
              ))}
            </nav>

            {/* Right: Settings + Token Status */}
            <div className="flex items-center gap-2">
              <TokenStatusIndicator compact />
              <Link to="/settings">
                <Button
                  variant={isActive('/settings') ? 'default' : 'ghost'}
                  size="sm"
                  title="Settings"
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

      {/* Page Content */}
      <main className="flex-1">
        <Outlet />
      </main>

      <Footer />
    </div>
  );
};
