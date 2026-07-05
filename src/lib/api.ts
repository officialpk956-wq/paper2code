const BASE = process.env.NEXT_PUBLIC_API_URL ?? 'https://paper2code-1-81y5.onrender.com';

function authHeaders(): Record<string, string> {
  const token = typeof window !== 'undefined' ? localStorage.getItem('access_token') : null;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function handle<T>(res: Response): Promise<T> {
  if (res.status === 401) {
    clearTokens();
    window.location.href = '/';
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
  }
}

export function clearTokens() {
  if (typeof window !== 'undefined') {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user_profile');
  }
}

export function isLoggedIn(): boolean {
  if (typeof window !== 'undefined') {
    return !!localStorage.getItem('access_token');
  }
  return false;
}

export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'GET',
    headers: authHeaders(),
  });
  return handle<T>(res);
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: {
      ...authHeaders(),
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  return handle<T>(res);
}

export async function apiPostForm<T>(path: string, form: FormData | URLSearchParams | string, contentType?: string): Promise<T> {
  const headers: Record<string, string> = authHeaders();
  if (contentType) {
    headers['Content-Type'] = contentType;
  }
  
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: headers,
    body: form,
  });
  return handle<T>(res);
}

export async function apiDelete<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'DELETE',
    headers: authHeaders(),
  });
  return handle<T>(res);
}
