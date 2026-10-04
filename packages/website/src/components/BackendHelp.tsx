/** Shared plain-language explanation for the presentation and the live demo's backend selector. */
export function BackendHelp({ className }: { className?: string }) {
  return (
    <div className={className}>
      <p>
        A backend azt mondja meg, hogyan fut a neurális hálózat a videokártyán. Mindkét mód az OpenDLSS-NR ugyanazon
        hálózatát és a betöltött modellsúlyokat használja.
      </p>
      <dl>
        <div>
          <dt>TSL (native three.js)</dt>
          <dd>
            A hálózat Three.js saját shader-rendszerére átírt változata. Szorosan illeszkedik a Three.js jelenethez, és
            nem igényel külön 16 bites számolási támogatást (shader-f16). A Three.js-be épített megvalósítás
            kipróbálásához válaszd.
          </dd>
        </div>
        <div>
          <dt>Reference WGSL (OpenDLSS-NR)</dt>
          <dd>
            Az OpenDLSS-NR eredeti WebGPU-programjait futtatja ugyanabban a Three.js jelenetben. Referenciaként segít
            ellenőrizni az átírást. Ehhez a grafikus eszközön szükséges a shader-f16 támogatás.
          </dd>
        </div>
      </dl>
      <p>
        A projekt ellenőrzött tesztjeiben a két megvalósítás számítása megegyezik. A sebességük a géptől és a
        felbontástól függ: azonos modell, jelenet és beállítások mellett a kisebb GPU-idő jelenti a gyorsabb futtatást.
      </p>
    </div>
  );
}
