import * as Sentry from "@sentry/nextjs";

const dsn = process.env.NEXT_PUBLIC_SENTRY_DSN;

if (dsn) {
  Sentry.init({
    dsn,
    // NEXT_PUBLIC_ENVIRONMENT is an optional override (e.g. to distinguish
    // "staging" from "production" on the same NODE_ENV=production build).
    // NODE_ENV is the safe default — Next.js sets it correctly with zero
    // config ("production" for `next build`/`next start`, "development" for
    // `next dev`), unlike NEXT_PUBLIC_ENVIRONMENT which nothing in this
    // project's deploy config currently sets, and which would otherwise
    // silently tag every production Sentry event as "development" forever.
    environment: process.env.NEXT_PUBLIC_ENVIRONMENT || process.env.NODE_ENV || "development",
    release: process.env.NEXT_PUBLIC_APP_VERSION || undefined,
    tracesSampleRate: parseFloat(process.env.NEXT_PUBLIC_SENTRY_TRACES_SAMPLE_RATE || "0.05"),
    
    // Disable console logging when DSN is present in production unless requested
    debug: false,
  });
}
