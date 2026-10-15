'use client';

import { useState, type FormEvent } from 'react';

export function MfaChallenge({ verify, cancel }: { verify: (code: string) => Promise<void>; cancel: () => void }) {
  const [code, setCode] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  async function submit(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError('');
    try { await verify(code); }
    catch (err) { setError(err instanceof Error ? err.message : 'Verification failed'); }
    finally { setLoading(false); }
  }
  return <form onSubmit={submit} className="flex flex-col gap-3">
    <h2 className="text-xl font-semibold text-white">Two-step verification</h2>
    <p className="text-sm text-[#A3A3A3]">Enter a code from your authenticator, or one unused backup code.</p>
    <label className="text-sm text-white">Authentication or backup code
      <input required autoFocus autoComplete="one-time-code" value={code} onChange={e => setCode(e.target.value)}
        className="mt-2 w-full rounded-lg border border-[#525252] bg-[#111111] p-3 text-white" />
    </label>
    {error && <p role="alert" className="text-sm text-red-400">{error}</p>}
    <button disabled={loading} type="submit" className="rounded-lg bg-[#A78BFA] p-3 text-black disabled:opacity-50">
      {loading ? 'Verifying…' : 'Verify and sign in'}
    </button>
    <button disabled={loading} type="button" onClick={cancel} className="text-sm text-[#A3A3A3]">Back to sign in</button>
  </form>;
}
