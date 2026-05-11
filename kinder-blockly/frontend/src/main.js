import './app.css';
import App from './App.svelte';
import * as Sentry from '@sentry/browser';

// Initialise Sentry before the Svelte app mounts so any errors raised during
// component construction are captured. VITE_SENTRY_DSN is a build-time env
// var: Vite inlines its value into the bundle at npm run build time. When it
// is unset (local dev, unconfigured builds) the SDK does not initialise and
// every Sentry call becomes a no-op.
const sentryDsn = import.meta.env.VITE_SENTRY_DSN;
if (sentryDsn) {
  Sentry.init({
    dsn: sentryDsn,
    environment: import.meta.env.MODE,
    release: import.meta.env.VITE_SENTRY_RELEASE,
    // No performance tracing — matches backend choice (PR #48).
    tracesSampleRate: 0,
    sendDefaultPii: false,
  });
}

const app = new App({ target: document.getElementById('app') });
export default app;
