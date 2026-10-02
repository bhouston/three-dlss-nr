import { createRootRoute, HeadContent, Link, Outlet, Scripts } from '@tanstack/react-router';

import { GITHUB_URL, PARITY_PATH, UPSTREAM_AUTHOR_URL, UPSTREAM_COMMIT_URL, UPSTREAM_URL } from '@/lib/links';
import appCss from '@/styles.css?url';

export const Route = createRootRoute({
  head: () => ({
    meta: [
      { charSet: 'utf-8' },
      { name: 'viewport', content: 'width=device-width, initial-scale=1' },
      { title: 'three-dlss-nr: OpenDLSS-NR for Three.js' },
      {
        name: 'description',
        content: 'A port to Three.js (TSL / WebGPU) of OpenDLSS-NR by maan, an open neural rendering network.',
      },
    ],
    links: [
      { rel: 'stylesheet', href: appCss },
      { rel: 'icon', type: 'image/svg+xml', href: '/favicon.svg' },
    ],
  }),
  shellComponent: RootDocument,
  component: RootLayout,
});

function RootLayout() {
  return (
    <div className="flex min-h-svh flex-col">
      <header className="flex items-center justify-between gap-4 border-b border-border px-6 py-3">
        <Link to="/" className="font-semibold">
          three-dlss-nr
        </Link>
        <nav className="flex items-center gap-4 text-sm">
          {/* A static site (fidelity-kit), served from public/parity: a plain link, not a router route. */}
          <a href={PARITY_PATH} className="text-primary underline underline-offset-4">
            Parity results
          </a>
          <a href={GITHUB_URL} className="text-primary underline underline-offset-4">
            GitHub
          </a>
        </nav>
      </header>
      <div className="flex min-h-0 flex-1 flex-col">
        <Outlet />
      </div>
      <footer className="border-t border-border px-4 py-3 text-center text-sm text-muted-foreground">
        <p>
          A port to Three.js (TSL / WebGPU) of{' '}
          <a href={UPSTREAM_URL} className="text-primary underline underline-offset-4">
            OpenDLSS-NR
          </a>{' '}
          by{' '}
          <a href={UPSTREAM_AUTHOR_URL} className="text-primary underline underline-offset-4">
            maan
          </a>{' '}
          (MIT, pinned at{' '}
          <a href={UPSTREAM_COMMIT_URL} className="text-primary underline underline-offset-4">
            9d08f41
          </a>
          ). Ported by{' '}
          <a href="https://ben3d.ca" className="text-primary underline underline-offset-4">
            Ben Houston
          </a>
          . The{' '}
          <a href={PARITY_PATH} className="text-primary underline underline-offset-4">
            parity results
          </a>{' '}
          compare the three.js ports with OpenDLSS-NR side by side.
        </p>
        <p className="mt-1 text-xs">
          Not affiliated with, endorsed by, or supported by NVIDIA Corporation. &quot;DLSS&quot; is a trademark of
          NVIDIA Corporation, used here only descriptively. No NVIDIA software or weights are included.
        </p>
      </footer>
    </div>
  );
}

function RootDocument({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <HeadContent />
      </head>
      <body className="bg-background text-foreground antialiased">
        {children}
        <Scripts />
      </body>
    </html>
  );
}
