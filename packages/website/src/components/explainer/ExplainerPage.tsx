import { useState } from 'react';
import { Link } from '@tanstack/react-router';

import { GITHUB_URL, PARITY_PATH, UPSTREAM_URL } from '@/lib/links';
import { BackendHelp } from '../BackendHelp';
import { Comparison } from './Comparison';
import { Gallery } from './Gallery';
import { Arrow, Cube } from './Icons';
import { OrbitScene, type StudioMode } from './OrbitScene';
import { usePresentationMotion } from './motion';

const nvidia = 'https://www.nvidia.com/en-us/geforce/news/dlss-5-3d-guided-neural-rendering/';
const studioModes: { id: StudioMode; title: string; short: string; description: string }[] = [
  {
    id: 'shape',
    title: 'Forma',
    short: 'miből áll az objektum',
    description: 'A drótváz megmutatja a térbeli formát felépítő háromszögeket. Ez a tárgy szerkezete.',
  },
  {
    id: 'material',
    title: 'Anyag',
    short: 'hogyan néz ki a felülete',
    description: 'Ugyanaz a forma, más felület: a kerámia lágyan, a fém élesen veri vissza a fényt.',
  },
  {
    id: 'light',
    title: 'Fény',
    short: 'hogyan látjuk mindezt',
    description: 'A kék fény más hangulatot ad. A forma megmarad, mégis másképp érzékeljük a tárgyat.',
  },
];

export function ExplainerPage() {
  const motion = usePresentationMotion();
  const [mode, setMode] = useState<StudioMode>('material');
  const activeMode = studioModes.find((entry) => entry.id === mode)!;

  return (
    <div className="explainer" id="top">
      <a className="skip-link" href="#lenyeg">
        Ugrás a magyarázathoz
      </a>
      <header className="presentation-header page-width">
        <a href="#top" className="presentation-brand" aria-label="Render újragondolva, az oldal eleje">
          <Cube />
          <span>
            RENDER /<br />
            ÚJRAGONDOLVA
          </span>
        </a>
        <nav aria-label="Bemutató fejezetei">
          <a href="#lenyeg">A lényeg</a>
          <a href="#threejs">Three.js</a>
          <a href="#projektek">Nyílt projektek</a>
        </nav>
        <a className="button button-outline header-demo" href="/local-demo.html">
          Élő demó <Arrow />
        </a>
      </header>
      <main>
        <section className="hero page-width" aria-labelledby="hero-title">
          <div className="hero-copy">
            <h1 id="hero-title">
              Ugyanaz a világ.
              <br />
              <span>Új fényben.</span>
            </h1>
            <p>
              A DLSS 5 az AI segítségével új fényt és részleteket ad a már megrajzolt 3D világnak. Nézd meg, mi történik
              a képpel — és hogyan jut el mindez a böngészőig.
            </p>
            <div className="hero-actions">
              <a className="button button-dark" href="#kulonbseg">
                Mutasd a különbséget <Arrow />
              </a>
              <a className="text-link" href="#lenyeg">
                Mi az a DLSS 5?
              </a>
            </div>
          </div>
          <div className="hero-art" data-parallax="0.13">
            <OrbitScene motion={motion.enabled} hero />
            <span className="hero-art-note">3D ILLUSZTRÁCIÓ · THREE.JS</span>
            <span className="art-coordinate" aria-hidden="true">
              +<br />
              FORMA
              <br />
              FÉNY
              <br />
              RÉSZLET
            </span>
          </div>
          <div className="hero-bottom">
            <span className="mono-note">GÖRGESS, NÉZD MEG, PRÓBÁLD KI</span>
            <button
              className="motion-toggle"
              aria-pressed={motion.enabled}
              disabled={motion.reduced}
              onClick={motion.toggle}
            >
              {motion.reduced
                ? 'Csökkentett mozgás'
                : motion.enabled
                  ? 'Animációk szüneteltetése'
                  : 'Animációk indítása'}
              <span aria-hidden="true">{motion.enabled ? 'Ⅱ' : '▷'}</span>
            </button>
          </div>
        </section>
        <Comparison />
        <section className="explanation page-width reveal" id="lenyeg" aria-labelledby="essence-title">
          <div>
            <span className="section-index">02 / A LÉNYEG</span>
            <h2 id="essence-title">
              A jelenet kész.
              <br />
              Az AI továbbrajzolja.
            </h2>
            <p className="large-copy">
              Képzeld el úgy, mint egy virtuális világ utolsó világítási és részletezési lépését.
            </p>
          </div>
          <div className="explanation-body">
            <p>
              A játékmotor megrajzolja a képet. A DLSS 5 neurális renderelése ebből indul ki: életszerűbb fényeket és
              anyagrészleteket adhat a bőrhöz, fémhez vagy növényzethez. A mozgásadatok segítenek, hogy az eredmény
              képről képre következetes maradjon.
            </p>
            <div className="plain-distinction">
              <strong>Három külön feladat</strong>
              <dl>
                <div>
                  <dt>Felskálázás</dt>
                  <dd>Kisebb képből nagyobb felbontású kép.</dd>
                </div>
                <div>
                  <dt>Képkockagenerálás</dt>
                  <dd>További képek a simább mozgáshoz.</dd>
                </div>
                <div>
                  <dt>Neurális renderelés</dt>
                  <dd>A megrajzolt kép megjelenésének továbbalakítása. Ez az itt bemutatott hálózat feladata.</dd>
                </div>
              </dl>
            </div>
            <p className="small-copy">
              A látvány megváltozhat, ezért számít a művészi szándék és az erősség beállítása. A szebbnek ható kép nem
              automatikusan hűbb az eredetihez.
            </p>
            <a className="text-link" href={nvidia} target="_blank" rel="noreferrer">
              A DLSS 5 magyarázata az NVIDIA-tól <Arrow diagonal />
            </a>
          </div>
        </section>
        <section className="three-section page-width reveal" id="threejs" aria-labelledby="three-title">
          <div className="three-stage">
            <span className="section-index">03 / THREE.JS</span>
            <h2 id="three-title">
              A 3D színpad.
              <br />A böngésződben.
            </h2>
            <div className="studio-art" data-parallax="0.08">
              <OrbitScene motion={motion.enabled} mode={mode} />
              <span className="studio-note mono-note">FORGASD AZ EGÉRREL VAGY AZ UJJADDAL</span>
            </div>
            <fieldset className="studio-controls" aria-label="A 3D színpad megjelenése">
              {studioModes.map((entry) => (
                <button key={entry.id} aria-pressed={mode === entry.id} onClick={() => setMode(entry.id)}>
                  {entry.title}
                </button>
              ))}
            </fieldset>
          </div>
          <div className="three-copy">
            <h3>Ez a Three.js.</h3>
            <p>
              Egy JavaScript-könyvtár, amellyel térbeli jeleneteket építhetünk a weben. Modell, anyag, fény és kamera:
              ugyanazok az alapok, mint egy virtuális stúdióban.
            </p>
            <div className="studio-rows">
              {studioModes.map((entry, index) => (
                <button key={entry.id} aria-pressed={mode === entry.id} onClick={() => setMode(entry.id)}>
                  <span className="row-number">0{index + 1}</span>
                  <strong>{entry.title}</strong>
                  <span>{entry.short}</span>
                  <Arrow />
                </button>
              ))}
            </div>
            <p className="studio-description" aria-live="polite">
              {activeMode.description}
            </p>
            <p className="small-copy">
              A fenti tárgyat valóban a Three.js rajzolja. Ez szemléltető 3D animáció; a neurális feldolgozás a külön
              demóban fut.
            </p>
            <a className="text-link" href="https://threejs.org/" target="_blank" rel="noreferrer">
              Nézz körül a Three.js világában <Arrow diagonal />
            </a>
          </div>
        </section>
        <section className="open-section" id="projektek" aria-labelledby="open-title">
          <div className="page-width">
            <div className="open-grid reveal">
              <div>
                <span className="section-index">04 / OPEN SOURCE</span>
                <h2 id="open-title">
                  A motor nyitva.
                  <br />
                  <span>OpenDLSS-NR.</span>
                </h2>
                <p>
                  maan nyílt kódú megvalósítása a DLSS 5 neurális hálózatáról. Megmutatja, hogyan számol a rendszer — és
                  lehetővé teszi az eredmények ellenőrzését.
                </p>
                <p className="small-copy">
                  Van Vulkan-megvalósítása és külön böngészős WebGPU-portja is. A cél az eredeti hálózat számításának
                  pontos újraalkotása.
                </p>
              </div>
              <div className="network-flow" aria-label="Megrajzolt kép, neurális hálózat, új kép">
                <div>
                  <Cube />
                  <span>
                    Megrajzolt
                    <br />
                    kép
                  </span>
                </div>
                <Arrow />
                <div className="flow-neural">
                  <Cube network />
                  <span>
                    Neurális
                    <br />
                    hálózat
                  </span>
                </div>
                <Arrow />
                <div>
                  <svg className="cube-icon" viewBox="0 0 48 48" fill="none" aria-hidden="true">
                    <rect x="5" y="5" width="38" height="38" stroke="currentColor" strokeWidth="1.5" />
                    <path d="m5 36 12-13 9 9 6-6 11 11" stroke="currentColor" strokeWidth="1.5" />
                    <circle cx="32" cy="15" r="4" fill="currentColor" />
                  </svg>
                  <span>
                    Új
                    <br />
                    kép
                  </span>
                </div>
              </div>
            </div>
            <div className="open-foot">
              <p>
                <span aria-hidden="true">!</span>A nyílt kód és a betanított modell két külön dolog. A projektek
                NVIDIA-modellsúlyokat nem tartalmaznak.
              </p>
              <a className="text-link" href={UPSTREAM_URL} target="_blank" rel="noreferrer">
                OpenDLSS-NR a GitHubon <Arrow diagonal />
              </a>
            </div>
          </div>
        </section>
        <section className="port-section page-width reveal" aria-labelledby="port-title">
          <div>
            <span className="section-index">05 / THREE-DLSS-NR</span>
            <h2 id="port-title">
              És mindez
              <br />
              <span>a webre költözik.</span>
            </h2>
            <p className="large-copy">
              A three-dlss-nr az OpenDLSS-NR hálózatát Three.js és WebGPU környezetbe ülteti át.
            </p>
            <a className="text-link" href={GITHUB_URL} target="_blank" rel="noreferrer">
              A three-dlss-nr projekt <Arrow diagonal />
            </a>
          </div>
          <div className="port-benefits">
            <article>
              <Cube />
              <div>
                <h3>A jelenetből közvetlenül</h3>
                <p>
                  A 3D jelenet és a neurális feldolgozás ugyanazon a grafikus eszközön dolgozik. A képet nem kell a
                  CPU-ra visszamásolni.
                </p>
              </div>
            </article>
            <article>
              <Cube network />
              <div>
                <h3>Összehasonlítható eredmények</h3>
                <p>
                  A portot a referencia számításaival ellenőrzik. A nyilvános tesztek szintetikus súlyokkal is
                  bizonyítják a számítás egyezését.
                </p>
                <a href={PARITY_PATH}>
                  Egyezőségi vizsgálatok <Arrow diagonal />
                </a>
              </div>
            </article>
            <article>
              <Arrow />
              <div>
                <h3>Kísérletezés a böngészőben</h3>
                <p>
                  Jelenetváltás, beállítások és osztott nézet egy weboldalon. Jó eszköz tanuláshoz, bemutatókhoz és
                  fejlesztői kísérletekhez.
                </p>
              </div>
            </article>
          </div>
        </section>
        <section id="backends" className="backend-section page-width reveal" aria-labelledby="backends-title">
          <span className="section-index">06 / A KÉT BACKEND</span>
          <h2 id="backends-title">Ugyanaz a hálózat, kétféle futtatás.</h2>
          <BackendHelp className="backend-help" />
        </section>
        <Gallery />
        <section className="demo-cta" aria-labelledby="demo-cta-title">
          <div className="page-width reveal">
            <h2 id="demo-cta-title">Most te jössz.</h2>
            <div>
              <p>Nyisd meg a működő demót, válts jelenetet, és húzd az elválasztót.</p>
              <a className="button button-lime" href="/local-demo.html">
                Élő demó megnyitása <Arrow />
              </a>
            </div>
          </div>
        </section>
        <section className="practical-notes page-width">
          <h2>Amit érdemes tudni</h2>
          <div>
            <p>
              <strong>A bemutató mindenkié.</strong> Az álló képes összehasonlítás WebGPU nélkül is működik. Az élő
              neurális demóhoz WebGPU-képes böngésző és megfelelő grafikus eszköz kell.
            </p>
            <p>
              <strong>A modell külön szükséges.</strong> A helyi demó az ezen a gépen már beállított modellt használja.
              Máshol a <Link to="/demo">kézi modellbetöltésű demó</Link> nyitható meg. A szintetikus tesztsúlyok képe
              nem értelmes.
            </p>
            <p>
              <strong>Ez egy kísérleti böngészős port.</strong> A sebessége függ a géptől és a felbontástól. A hivatalos
              játékos DLSS 5 teljesítménye és funkciókészlete nem következik ebből a demóból.
            </p>
          </div>
        </section>
      </main>
      <footer className="presentation-footer page-width">
        <div className="footer-sources">
          <span>FORRÁSOK</span>
          <a href={nvidia}>NVIDIA</a>
          <a href="https://threejs.org/">Three.js</a>
          <a href={UPSTREAM_URL}>OpenDLSS-NR</a>
          <a href={GITHUB_URL}>three-dlss-nr</a>
        </div>
        <p>
          OpenDLSS-NR: maan · Three.js port: Ben Houston. Független bemutató, nincs hivatalos kapcsolat az NVIDIA-val. A
          DLSS az NVIDIA védjegye.
        </p>
      </footer>
    </div>
  );
}
