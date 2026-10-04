import { useRef, useState, type CSSProperties } from 'react';

import { DEMO_MODELS } from '@/lib/models';

const subjects = [
  {
    id: 'head',
    name: 'Emberi arc',
    model: 'lee-perry-smith',
    detail:
      'Figyeld a bőr árnyalatait, a szem környékét és az apró részleteket. Az AI a karakter megjelenését is módosíthatja.',
  },
  {
    id: 'car',
    name: 'Fényes felületek',
    model: 'car-concept',
    detail: 'Figyeld a karosszérián végigfutó fényt és a tükröződéseket. A különbség a felületek megjelenésében van.',
  },
] as const;

export function Comparison() {
  const [selected, setSelected] = useState(0);
  const [split, setSplit] = useState(50);
  const dragging = useRef(false);
  const subject = subjects[selected];
  const model = DEMO_MODELS.find((entry) => entry.id === subject.model)!;
  const position = (event: React.PointerEvent<HTMLInputElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    setSplit(Math.round(Math.max(0, Math.min(100, ((event.clientX - box.left) / box.width) * 100))));
  };
  return (
    <section className="comparison-section page-width reveal" id="kulonbseg" aria-labelledby="comparison-title">
      <div className="comparison-copy">
        <span className="section-index">01 / NÉZD MEG</span>
        <h2 id="comparison-title">
          A különbséget
          <br />
          látni kell.
        </h2>
        <p>Ugyanaz a jelenet, ugyanaz a kamera. Balra az eredeti kép, jobbra a neurális hálózat eredménye.</p>
        <p className="small-copy">{subject.detail}</p>
        <fieldset className="subject-controls" aria-label="Összehasonlítás tárgya">
          {subjects.map((entry, index) => (
            <button
              key={entry.id}
              aria-pressed={selected === index}
              onClick={() => {
                setSelected(index);
                setSplit(50);
              }}
            >
              {entry.name}
            </button>
          ))}
        </fieldset>
        <span className="mono-note">Húzd a választóvonalat. Billentyűzettel: nyilak, Home, End.</span>
      </div>
      <figure className="comparison-figure">
        <div className="image-comparison" style={{ '--split': `${split}%` } as CSSProperties}>
          <img
            src={`/explainer/${subject.id}-off.jpg`}
            alt={`${subject.name}: eredeti Three.js render`}
            width="1110"
            height="624"
          />
          <img
            className="comparison-after"
            src={`/explainer/${subject.id}-on.jpg`}
            alt={`${subject.name}: neurálisan feldolgozott kép`}
            width="1110"
            height="624"
          />
          <span className="comparison-label label-before">Eredeti render</span>
          <span className="comparison-label label-after">Neurális render</span>
          <div className="comparison-divider" aria-hidden="true">
            <span>
              <svg viewBox="0 0 32 32" fill="none">
                <path d="m12 11-5 5 5 5m8-10 5 5-5 5" stroke="currentColor" strokeWidth="1.5" />
              </svg>
            </span>
          </div>
          <input
            type="range"
            min="0"
            max="100"
            value={split}
            aria-label="Az eredeti és neurális kép elválasztója"
            aria-valuetext={`${split}% eredeti kép, ${100 - split}% neurális kép`}
            onChange={(event) => setSplit(Number(event.target.value))}
            onPointerDown={(event) => {
              event.preventDefault();
              dragging.current = true;
              event.currentTarget.focus();
              event.currentTarget.setPointerCapture(event.pointerId);
              position(event);
            }}
            onPointerMove={(event) => {
              if (dragging.current) position(event);
            }}
            onPointerUp={() => {
              dragging.current = false;
            }}
            onPointerCancel={() => {
              dragging.current = false;
            }}
          />
        </div>
        <figcaption>
          Valódi helyi demófelvétel · 640 × 360 · Natural stílus · Reference WGSL backend. A képek álló felvételek, az
          elválasztó interaktív.
        </figcaption>
        <p className="asset-credit">
          {model.attribution.title} — {model.attribution.author} ·{' '}
          <a href={model.attribution.licenseUrl}>{model.attribution.license}</a> ·{' '}
          <a href={model.attribution.sourceUrl}>Modell forrása</a>. A jobb oldali kép neurálisan módosított.
        </p>
      </figure>
    </section>
  );
}
