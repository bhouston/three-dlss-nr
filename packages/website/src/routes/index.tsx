import { createFileRoute } from '@tanstack/react-router';

import { DemoApp } from '@/components/DemoApp';
import { UPSTREAM_URL } from '@/lib/links';

export const Route = createFileRoute('/')({
  component: HomePage,
});

function HomePage() {
  return (
    <div className="flex w-full flex-col">
      <div className="mx-auto w-full max-w-[1500px] px-4 pt-4">
        <h1 className="text-xl font-semibold">OpenDLSS-NR for Three.js</h1>
        <p className="max-w-3xl text-sm text-muted-foreground">
          A Three.js (TSL / WebGPU) port of{' '}
          <a href={UPSTREAM_URL} className="text-primary underline underline-offset-4">
            OpenDLSS-NR
          </a>
          , an open reimplementation of a DLSS-style neural rendering network, running live in your browser on the scene
          below. You supply the model weights: none are included or hosted.
        </p>
      </div>
      <DemoApp />
    </div>
  );
}
