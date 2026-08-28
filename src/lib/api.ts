const BASE =
  (typeof process !== 'undefined' && process.env.NEXT_PUBLIC_API_URL) ||
  'http://127.0.0.1:8000';

const REQUEST_TIMEOUT_MS = 15000;

const TIMEOUT_MESSAGE = 'Request timed out — check your connection and try again';

export class ApiTimeoutError extends Error {
  constructor(message = TIMEOUT_MESSAGE) {
    super(message);
    this.name = 'TimeoutError';
  }
}

let refreshPromise: Promise<string | null> | null = null;

async function refreshAccessToken(): Promise<string | null> {
  if (refreshPromise) return refreshPromise;
  if (typeof window === 'undefined') return null;

  const storedRefresh = localStorage.getItem('refresh_token');
  if (!storedRefresh) return null;

  refreshPromise = (async () => {
    try {
      const res = await fetch(`${BASE}/api/auth/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: storedRefresh }),
        signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
      });

      if (!res.ok) {
        clearTokens();
        return null;
      }

      const data = await res.json().catch(() => null);
      if (data?.access_token) {
        setTokens(data);
        return data.access_token as string;
      }

      clearTokens();
      return null;
    } catch {
      clearTokens();
      return null;
    } finally {
      refreshPromise = null;
    }
  })();

  return refreshPromise;
}

function authHeaders(): Record<string, string> {
  const token = typeof window !== 'undefined' ? localStorage.getItem('access_token') : null;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function handle<T>(res: Response): Promise<T> {
  if (res.status === 401) {
    clearTokens();
    throw new Error('Unauthorized');
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    // FastAPI validation errors (422) return `detail` as an array of
    // {loc, msg} objects, not a string — stringify it into something readable
    // instead of letting it render as "[object Object]" wherever it's shown.
    const detail = Array.isArray(err.detail)
      ? err.detail.map((e: { msg?: string }) => e?.msg ?? JSON.stringify(e)).join('; ')
      : err.detail;
    throw new Error(detail ?? 'Request failed');
  }
  return res.json() as Promise<T>;
}

export function setTokens(tokens: { access_token: string; refresh_token?: string }) {
  if (typeof window !== 'undefined') {
    localStorage.setItem('access_token', tokens.access_token);
    if (tokens.refresh_token) {
      localStorage.setItem('refresh_token', tokens.refresh_token);
    }
    window.dispatchEvent(new Event('auth-changed'));
  }
}

export function clearTokens() {
  if (typeof window !== 'undefined') {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user_profile');
    window.dispatchEvent(new Event('auth-changed'));
  }
}

export function isLoggedIn(): boolean {
  if (typeof window !== 'undefined') {
    return !!localStorage.getItem('access_token');
  }
  return false;
}

function isTimeoutError(err: unknown): boolean {
  return err instanceof Error && (err.name === 'AbortError' || err.name === 'TimeoutError');
}

async function fetchWithTimeout(
  url: string,
  init: RequestInit,
  timeoutMs = REQUEST_TIMEOUT_MS
): Promise<Response> {
  if (!init.signal) {
    init.signal = AbortSignal.timeout(timeoutMs);
  }
  try {
    return await fetch(url, init);
  } catch (err) {
    if (isTimeoutError(err)) {
      throw new ApiTimeoutError();
    }
    throw err;
  }
}

async function requestWithRetry<T>(
  buildRequest: () => { url: string; init: RequestInit },
  timeoutMs = REQUEST_TIMEOUT_MS
): Promise<T> {
  const { url, init } = buildRequest();
  let res = await fetchWithTimeout(url, init, timeoutMs);

  if (res.status === 401) {
    const newToken = await refreshAccessToken();
    if (newToken) {
      const retryReq = buildRequest();
      res = await fetchWithTimeout(retryReq.url, retryReq.init, timeoutMs);
    }
  }

  return handle<T>(res);
}

export async function apiGet<T>(path: string, timeoutMs = REQUEST_TIMEOUT_MS): Promise<T> {
  return requestWithRetry<T>(() => ({
    url: `${BASE}${path}`,
    init: {
      method: 'GET',
      headers: authHeaders(),
    },
  }), timeoutMs);
}

export async function apiPost<T>(
  path: string,
  body: unknown,
  timeoutMs = REQUEST_TIMEOUT_MS
): Promise<T> {
  return requestWithRetry<T>(() => ({
    url: `${BASE}${path}`,
    init: {
      method: 'POST',
      headers: {
        ...authHeaders(),
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    },
  }), timeoutMs);
}

export async function apiPostForm<T>(
  path: string,
  form: FormData | URLSearchParams | string,
  contentType?: string,
  timeoutMs = REQUEST_TIMEOUT_MS
): Promise<T> {
  return requestWithRetry<T>(() => {
    const headers: Record<string, string> = authHeaders();
    if (contentType) {
      headers['Content-Type'] = contentType;
    }
    return {
      url: `${BASE}${path}`,
      init: {
        method: 'POST',
        headers,
        body: form,
      },
    };
  }, timeoutMs);
}

export function warmUpBackend(): void {
  try {
    fetch(`${BASE}/api/health`, { method: 'GET', cache: 'no-store' }).catch(() => {});
  } catch {}
}

export async function apiDelete<T>(path: string): Promise<T> {
  return requestWithRetry<T>(() => ({
    url: `${BASE}${path}`,
    init: {
      method: 'DELETE',
      headers: authHeaders(),
    },
  }));
}
