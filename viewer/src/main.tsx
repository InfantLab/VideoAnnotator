import { createRoot } from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import './utils/version' // Load version info and make it available globally
import './utils/debugUtils' // Load debug utilities and make them available globally
import { reloadForNewBuild } from './lib/staleBuild'

// Vite fires this when a lazily loaded page's file is gone: this tab predates
// an upgrade. Reloading gets the current build instead of an error screen.
window.addEventListener('vite:preloadError', (event) => {
  if (reloadForNewBuild()) event.preventDefault();
});

console.log('🚀 Application starting...');

try {
  const rootElement = document.getElementById("root");
  if (!rootElement) {
    throw new Error("Root element not found");
  }
  
  const root = createRoot(rootElement);
  root.render(<App />);
  console.log('✅ React root rendered');
} catch (error) {
  console.error('❌ Fatal error during app initialization:', error);
}
