import * as Sentry from "@sentry/nextjs";

const dsn = process.env.NEXT_PUBLIC_SENTRY_DSN;

if (dsn) {
  Sentry.init({
    dsn,
    // See src/instrumentation-client.ts for why NODE_ENV is the safe default here.
    environment: process.env.NEXT_PUBLIC_ENVIRONMENT || process.env.NODE_ENV || "development",
    release: process.env.NEXT_PUBLIC_APP_VERSION || undefined,
    tracesSampleRate: parseFloat(process.env.NEXT_PUBLIC_SENTRY_TRACES_SAMPLE_RATE || "0.05"),
    
    // Disable console logging when DSN is present in production unless requested
    debug: false,
  });
}
