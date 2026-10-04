import { useEffect, useRef, useState } from 'react';

import { DEMO_MODELS } from '@/lib/models';
import { Arrow } from './Icons';

const captures = [
  {
    id: 'car',
    model: 'car-concept',
    title: 'Fém és tükröződés',
    description: 'Sportautó: fények, élek, csillogó felületek.',
  },
  {
    id: 'helmet',
    model: 'flight-helmet',
    title: 'Sokféle anyag, egy tárgy',
    description: 'Pilótasisak: bőr, fém és üveg találkozása.',
  },
  {
    id: 'sofa',
    model: 'leather-sofa',
    title: 'Textúra és apró részlet',
    description: 'Bőrkanapé: puha felületek és mintás párnák.',
  },
];

export function Gallery() {
  const [selected, setSelected] = useState<number | null>(null);
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const element = dialog.current;
    if (!element) return;
    if (selected !== null) element.showModal();
    else element.close();
  }, [selected]);
  const capture = selected === null ? null : captures[selected];
  const model = capture ? DEMO_MODELS.find((entry) => entry.id === capture.model)! : null;
  return (
    <section className="gallery page-width reveal" aria-labelledby="gallery-title">
      <div className="gallery-heading">
        <span className="section-index">KÉPERNYŐKÉPEK A DEMÓBÓL</span>
        <h2 id="gallery-title">
          Ugyanaz a hálózat.
          <br />
          Más történetek.
        </h2>
        <p>Három jelenet, eltérő felületek. Nyisd meg a képet, és nézd meg közelebbről.</p>
      </div>
      <div className="gallery-rail">
        {captures.map((entry, index) => {
          const attribution = DEMO_MODELS.find((item) => item.id === entry.model)!.attribution;
          return (
            <figure key={entry.id}>
              <button
                className="gallery-image"
                onClick={() => setSelected(index)}
                aria-label={`${entry.title} képernyőkép nagyítása`}
              >
                <img
                  src={`/explainer/${entry.id}-split.jpg`}
                  alt={`${entry.description} Osztott demónézet: balra NR off, jobbra NR on.`}
                  loading="lazy"
                  width="1110"
                  height="624"
                />
                <span>
                  <Arrow diagonal />
                </span>
              </button>
              <figcaption>
                <h3>{entry.title}</h3>
                <p>{entry.description}</p>
                <button className="gallery-open" onClick={() => setSelected(index)}>
                  Nagyítás <Arrow diagonal />
                </button>
                <p className="asset-credit">
                  {attribution.author} · <a href={attribution.licenseUrl}>{attribution.license}</a>
                </p>
              </figcaption>
            </figure>
          );
        })}
      </div>
      <dialog
        ref={dialog}
        className="screenshot-dialog"
        onClose={() => setSelected(null)}
        onCancel={() => setSelected(null)}
        aria-labelledby="screenshot-title"
      >
        {capture && model ? (
          <>
            <div className="dialog-header">
              <h2 id="screenshot-title">{capture.title}</h2>
              <button onClick={() => setSelected(null)} aria-label="Képernyőkép bezárása">
                Bezárás ×
              </button>
            </div>
            <img src={`/explainer/${capture.id}-split.jpg`} alt={capture.description} />
            <p>Valódi demófelvétel · Eredeti render balra, neurális render jobbra · 640 × 360, Natural.</p>
            <p className="asset-credit">
              {model.attribution.title} — {model.attribution.author} ·{' '}
              <a href={model.attribution.licenseUrl}>{model.attribution.license}</a> ·{' '}
              <a href={model.attribution.sourceUrl}>Forrás és kredit</a>. A jobb oldali kép neurálisan módosított.
            </p>
          </>
        ) : null}
      </dialog>
    </section>
  );
}
