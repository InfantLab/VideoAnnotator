/// <reference types="vite/client" />

interface Window {
	__dns_correction_logged?: boolean;
	version?: {
		VERSION: string;
		GITHUB_URL: string;
		APP_NAME: string;
		getVersionString: () => string;
		getAppTitle: () => string;
		logVersionInfo: () => void;
	};
}
