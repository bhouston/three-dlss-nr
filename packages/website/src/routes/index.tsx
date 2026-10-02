import { createFileRoute } from '@tanstack/react-router';
import { useState } from 'react';

import { SceneViewer } from '@/components/SceneViewer';
import { UPSTREAM_URL } from '@/lib/links';

export const Route = createFileRoute('/')({
  component: HomePage,
});

function HomePage() {
  const [nrEnabled, setNrEnabled] = useState(false);

  return (
    <div className="mx-auto flex w-full max-w-4xl flex-col gap-6 p-6">
      <div className="flex flex-col gap-3 text-center">
        <h1 className="text-2xl font-semibold">OpenDLSS-NR for Three.js</h1>
        <p className="mx-auto max-w-2xl text-sm text-muted-foreground">
          A Three.js (TSL / WebGPU) port of{' '}
          <a href={UPSTREAM_URL} className="text-primary underline underline-offset-4">
            OpenDLSS-NR
          </a>
          , an open-source reimplementation of a DLSS-style neural rendering network. The network port is in progress;
          the toggle below is a placeholder. You supply the model weights; none are included.
        </p>
      </div>

      <label className="flex items-center justify-center gap-2 text-sm">
        <input type="checkbox" checked={nrEnabled} onChange={(event) => setNrEnabled(event.target.checked)} />
        Neural rendering: <span className="font-semibold">{nrEnabled ? 'on' : 'off'}</span>
      </label>

      <SceneViewer nrEnabled={nrEnabled} />
    </div>
  );
}
