import { createRoot } from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import './utils/version' // Load version info and make it available globally
import './utils/debugUtils' // Load debug utilities and make them available globally

createRoot(document.getElementById("root")!).render(<App />);
