import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

import { setNotifyNavigator } from '@/lib/notify';

/** Inside the router: lets a toast's next step move within the viewer. */
export function NotifyNavigator() {
  const navigate = useNavigate();
  useEffect(() => {
    setNotifyNavigator((to) => navigate(to));
    return () => setNotifyNavigator(null);
  }, [navigate]);
  return null;
}
