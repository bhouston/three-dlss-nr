import { useEffect, useRef, useState } from 'react';

import { createDemoScene, type DemoScene } from '@/lib/demo-scene';

/**
 * Renders the demo scene with a WebGPURenderer. `nrEnabled` is a placeholder
 * for the neural rendering pass; until the port lands it only tints the
 * background so the toggle is visibly wired up.
 */
export function SceneViewer({ nrEnabled }: { nrEnabled: boolean }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<DemoScene | null>(null);
  const nrEnabledRef = useRef(nrEnabled);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    let disposed = false;
    let demo: DemoScene | null = null;
    createDemoScene(container)
      .then((created) => {
        if (disposed) {
          created.dispose();
          return;
        }
        demo = created;
        created.setNeuralRendering(nrEnabledRef.current);
        sceneRef.current = created;
      })
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : String(reason)));
    return () => {
      disposed = true;
      demo?.dispose();
      sceneRef.current = null;
    };
  }, []);

  useEffect(() => {
    nrEnabledRef.current = nrEnabled;
    sceneRef.current?.setNeuralRendering(nrEnabled);
  }, [nrEnabled]);

  return (
    <div ref={containerRef} className="relative aspect-video w-full overflow-hidden rounded border border-border">
      {error ? <p className="p-4 text-sm text-muted-foreground">WebGPU is unavailable: {error}</p> : null}
    </div>
  );
}
