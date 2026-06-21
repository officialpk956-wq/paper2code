const DEFAULT_BACKEND_URL = 'http://127.0.0.1:8000';

export function getBackendBaseUrl(): string {
  const configured = process.env.PAPER2CODE_BACKEND_URL?.trim();
  if (configured) {
    return configured.replace(/\/$/, '');
  }

  return DEFAULT_BACKEND_URL;
}

export function getBackendUrl(path: string): string {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${getBackendBaseUrl()}${normalizedPath}`;
}
