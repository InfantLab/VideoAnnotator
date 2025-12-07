import { createRoot } from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import './utils/version' // Load version info and make it available globally
import './utils/debugUtils' // Load debug utilities and make them available globally

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
