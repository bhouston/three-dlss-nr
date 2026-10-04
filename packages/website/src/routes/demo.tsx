import { createFileRoute, Link } from '@tanstack/react-router';

import { DemoApp } from '@/components/DemoApp';
import { GITHUB_URL, PARITY_PATH, UPSTREAM_URL } from '@/lib/links';

export const Route = createFileRoute('/demo')({
  head: () => ({ meta: [{ title: 'three-dlss-nr · Élő WebGPU demó' }] }),
  component: DemoPage,
});

function DemoPage() {
  return (
    <div className="demo-page">
      <header className="demo-header">
        <Link to="/">← Vissza a bemutatóhoz</Link>
        <nav aria-label="Demó hivatkozásai">
          <a href={PARITY_PATH}>Egyezőségi eredmények</a>
          <a href={GITHUB_URL}>GitHub</a>
        </nav>
      </header>
      <div className="mx-auto w-full max-w-[1500px] px-4 pt-4">
        <h1 className="text-xl font-semibold">OpenDLSS-NR a Three.js-ben</h1>
        <p className="max-w-3xl text-sm text-muted-foreground">
          A működő WebGPU demó. Saját kompatibilis modellmappát tölthetsz be. A szintetikus súlyok csak a számítás
          kipróbálására szolgálnak; nem adnak értelmes képet.
        </p>
      </div>
      <DemoApp />
      <footer className="demo-footer">
        <a href={UPSTREAM_URL}>OpenDLSS-NR: maan</a> · Three.js port: Ben Houston. A kód MIT-licencű;
        NVIDIA-modellsúlyokat nem tartalmaz. Független, nem hivatalos NVIDIA-projekt.
      </footer>
    </div>
  );
}
