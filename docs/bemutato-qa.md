# Bemutatóoldal – ellenőrzési jegyzőkönyv

2026. október 4. A bemutató a meglévő React / TanStack Start weboldal főoldalán
      érhető el. A neurális demó a `/demo` útvonalra került; a helyi automatikus
      modellbetöltő továbbra is a `/local-demo.html` címen működik.

## Látvány és forráshűség

Három koordinált fejezetterv készült Image Gen eszközzel. Helyi referenciák:
`.local/explainer-design/01-dlss.png`, `02-three-opendlss.png`, `03-port.png`.
A tervképek és a kész böngészőképek külön `view_image` ellenőrzésen szerepeltek.
Az asztali ellenőrzés 1536 × 1024-es nézetben, a mobilos 390 × 844 és
320 × 740-es nézetben történt. A böngésző eredeti méretét a végén visszaállítottuk.
Az eredeti, 1766 × 1030-as böngészőnézetet is megvizsgáltuk és rögzítettük.

| Összevetési pont     | Terv                                                     | Kész oldal és döntés                                                                                                           |
| -------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Nyitó szöveg         | „Ugyanaz a világ. Új fényben.”, „Mutasd a különbséget”   | Azonos szöveg és sorrend; lime szöveg grafitszínű téglalapon.                                                                  |
| Fejléc               | Drótvázkocka, három fejezetlink, Élő demó                | Az első terv fejlécét használja a teljes oldal. A későbbi tervképek kitalált márkái és navigációi elmaradnak.                  |
| Tipográfia           | Nagy Space Grotesk címsorok, visszafogott törzsszöveg    | Helyi Space Grotesk és Manrope; a gombok és feliratok mérete külön beállított. A keskeny mobilon javított címsortörés.         |
| Paletta              | Papírszín, grafit, lime, vékony vonalak                  | `#f7f7f2`, `#171919`, `#c0fa67`, `#c9ccc3`; nincs a képekre tett színfedés.                                                    |
| Konténerek és ritmus | Nyitott oszlopok, sötét sáv, galéria                     | Megtartott szerkezet; a DLSS-fogalmak külön magyarázó fejezetet kaptak a közérthetőséghez.                                     |
| 3D tárgyak           | Kerámia és fémes lime formák, drótváz, orbitális vonalak | Valós időben rajzolt és forgatható Three.js-geometria. A kerámia sima felületű, a raszteres terv helyett valódi térbeli tárgy. |
| Fotók és felvételek  | A generált terv kitalált arcot és jeleneteket mutat      | Tudatos csere: tényleges, azonos kamerájú helyi demófelvételek. A képek eredete, beállítása és licence az oldalon látható.     |
| Ikonok               | Vékony kontúrkocka, nyilak, hálózat, kép                 | Saját SVG ikonok egységes vonalvastagsággal; a neurális hálózat összekötött csomópontokat kapott.                              |
| Mobil                | Az asztali terv szerkezeti folytatása                    | Egyoszlopos elrendezés; a hálózati ábra és a három 3D gomb 320 px-en is elfér. Nincs vízszintes túlfutás.                      |
| Mozgás               | Térbeli motívumok és finom mozgás                        | Automatikus forgás, parallax és szakaszbelépések; működő szüneteltető.                                                         |

A nyitó szöveg összevetése: a főcím, bevezető, elsődleges és másodlagos gomb,
márka és navigáció megmaradt. Szándékos kiegészítés a görgetési jelzés,
az animációkapcsoló és a 3D illusztráció egyértelmű felirata. A fejezetek
magyarázatai elsődleges forrásokhoz igazítottak; kitalált teljesítményadat vagy
generált „előtte–utána” bizonyíték nem szerepel az oldalon.

A fenti, dokumentált eltérésekkel a kész oldalt a három tervhez hűen ellenőriztük.
Az ellenőrzött nézetekben javítható, lényeges vizuális eltérés nem maradt.

Javítások az ellenőrzés során: mobilon a hálózati ábra min-content túlfutása,
a három 3D vezérlő szélessége, egy keskeny címsor rossz törése és a nagyított
kép párbeszédablakának középre igazítása. A színpad kameráját közelebb hoztuk
a jobban látható tárgyhoz; a mozgáskapcsoló betűméretét egyértelműen rögzítettük.

## Működés a böngészőben

Az ellenőrzés az Edge böngészőben, a beépített Browser / CUA eszközzel történt.
Playwright Chromium tartalék böngészőre nem volt szükség. A mentett JPEG-ek
a böngésző képernyőrögzítő API-jából származnak.

- Az arc és autó összehasonlítása váltáskor 50%-ra áll. A húzás 50 → 73%-ot
  adott; a Home, End és nyílbillentyűk működnek, a felolvasott százalék frissül.
- A Forma / Anyag / Fény vezérlők a 3D látványt, a kijelölt állapotot és a
  magyarázatot is módosítják. Szüneteltetett animációnál a kézi forgatás két
  mentett képen látható.
- A parallax görgetésre változik. Szüneteltetéskor mindkét eltolás `0px`,
  az oldal mozgásállapota `off`; a folytatás visszakapcsolja.
- A képgaléria sisakképe nagyítható. A Bezárás és Escape bezárja a natív
  párbeszédablakot; a fókusz a megnyitó gombra tér vissza.
- A fejezetlinkek, a kézi demó és a helyi automatikus modellbetöltő működnek.
  Az új útvonalon a demó `Network: ready`, helyi modell és 640 × 360-as
  referenciahálózat mellett valódi autós osztott nézetet mutatott.
- A kezdőoldal két Three.js-vászna megjelent, a galériaképek a szakaszhoz
  görgetéskor betöltődtek. A mobilos tartalom és vezérlők nem lógnak ki.
- A rendszer csökkentett mozgását a CSS és `matchMedia` kezeli. Ezt
  kódellenőrzéssel vizsgáltuk; a Windows rendszerbeállítását nem változtattuk meg.
- Böngészős JavaScript-hiba nem volt. Egy nem végzetes Windows/D3D12
  shaderfordítási figyelmeztetés megjelent a Three.js környezetfényénél.

## Helyi ellenőrzések

| Ellenőrzés                             | Eredmény                                                                                                                       |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `install --frozen-lockfile`            | Sikeres; a függőségek zárolása nem változott.                                                                                  |
| `build`, `tsc`, `lint`, `format:check` | Sikeres. A build nagy Three.js-csomagra figyelmeztet, de elkészül.                                                             |
| `test`                                 | 16 fájl, 88 sikeres teszt.                                                                                                     |
| `test:gpu`                             | 20 sikeres és 2 kihagyott fájl; 112 sikeres és 55 kihagyott teszt.                                                             |
| `release:check`                        | 8 sikeres, 1 kihagyott teszt; a teljes shell-pipeline Windows alatt kihagyott.                                                 |
| `size`                                 | A könyvtár és a szintetikus segédmodul a beállított méretkorláton belül.                                                       |
| `fidelity:check`                       | 8 jelenet, 1344 tenzor- és 192 képösszevetés; bitpontos egyezés.                                                               |
| HTTP és indító                         | `/`, `/demo`, `/local-demo.html`, képek és helyi font: HTTP 200. A bemutatóindító felismeri és újrahasználja a saját szervert. |
| `git diff --check`                     | Sikeres.                                                                                                                       |

A GPU-tesztek determinisztikus szintetikus súlyokkal futottak. A helyi,
korábban beállított betanított modellt csak a demó felvételeihez és az élő
útvonal ellenőrzéséhez használtuk. Modellsúly nem került Gitbe.

## Mentett képernyőképek

- [Asztali nyitókép](images/explainer/desktop.jpg)
- [Nyitókép az eredeti böngészőméretben](images/explainer/desktop-default.jpg)
- [Húzható összehasonlítás](images/explainer/comparison.jpg)
- [Three.js színpad](images/explainer/threejs.jpg)
- [Kézi forgatás után](images/explainer/threejs-drag.jpg)
- [Drótváznézet](images/explainer/threejs-wire.jpg)
- [OpenDLSS magyarázat](images/explainer/opendlss.jpg)
- [three-dlss-nr és galéria](images/explainer/project.jpg)
- [Képgaléria](images/explainer/gallery.jpg)
- [Nagyított képernyőkép](images/explainer/gallery-modal.jpg)
- [Mobilnyitó](images/explainer/mobile.jpg)
- [Működő neurális demó](images/explainer/live-demo.jpg)

Bemutatási menet és képkreditek: [bemutato.md](bemutato.md).
