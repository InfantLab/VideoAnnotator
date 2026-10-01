/// <reference types="vite/client" />

interface Window {
	version?: {
		VERSION: string;
		GITHUB_URL: string;
		APP_NAME: string;
		getVersionString: () => string;
		getAppTitle: () => string;
		logVersionInfo: () => void;
	};
}
