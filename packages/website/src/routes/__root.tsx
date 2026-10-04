import { createRootRoute, HeadContent, Outlet, Scripts } from '@tanstack/react-router';

import appCss from '@/styles.css?url';

export const Route = createRootRoute({
  head: () => ({
    meta: [
      { charSet: 'utf-8' },
      { name: 'viewport', content: 'width=device-width, initial-scale=1' },
      { title: 'Új fényben · DLSS 5 és Three.js érthetően' },
      {
        name: 'description',
        content:
          'Interaktív magyar bemutató a DLSS 5, a Three.js, az OpenDLSS-NR és a three-dlss-nr világáról. Valódi összehasonlítások, 3D és közérthető magyarázatok.',
      },
    ],
    links: [
      { rel: 'stylesheet', href: appCss },
      { rel: 'icon', type: 'image/svg+xml', href: '/favicon.svg' },
      {
        rel: 'preload',
        href: '/fonts/SpaceGrotesk-Variable.ttf',
        as: 'font',
        type: 'font/ttf',
        crossOrigin: 'anonymous',
      },
    ],
  }),
  shellComponent: RootDocument,
  component: Outlet,
});

function RootDocument({ children }: { children: React.ReactNode }) {
  return (
    <html lang="hu">
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
