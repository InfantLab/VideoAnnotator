import { createRoot } from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import './utils/version' // Load version info and make it available globally

createRoot(document.getElementById("root")!).render(<App />);
