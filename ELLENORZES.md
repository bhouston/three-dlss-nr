# Helyi telepítés és ellenőrzés

Ellenőrzés dátuma: **2026-10-04**.

## Telepített részek

| Rész                         | Verzió / állapot                                                  |
| ---------------------------- | ----------------------------------------------------------------- |
| three-dlss-nr                | `78347756633cff223d79ed35c9f03a60a8273f05`                        |
| OpenDLSS-NR alprojekt        | `9d08f4184bbcb9d858e2fb7a7834ec0837a9d2f1`                        |
| Helyi Node.js                | 26.10.0, win-x64, hivatalos ZIP SHA-256 ellenőrzéssel             |
| Helyi pnpm                   | 11.27.0                                                           |
| Függőségek                   | `pnpm install --frozen-lockfile`, sikeres                         |
| GPU                          | NVIDIA GeForce RTX 3090, 24 GiB VRAM, 616.92 driver               |
| Demó és összehasonlító oldal | Fordított változat, `http://localhost:3300`, `/parity/`           |
| Szintetikus tesztmodell      | seed 1, 11 szakasz, 12 fájl, 140.9 MiB, `.local/models/synthetic` |

Betanított modell: Visual Enhancer v14.0 / DLL 310.8.SF.0; 153 tensor,
147,683,778 bájt, `.local/models/nvidia-310.8`.
Automatikus helyi betöltés: `http://localhost:3300/local-demo.html`, TSL,
640 × 360, Natural, Split. Új böngészőlapon kipróbálva, modellkézi kiválasztás nélkül.

## Futtatott ellenőrzések

| Ellenőrzés                              | Eredmény                                                                           |
| --------------------------------------- | ---------------------------------------------------------------------------------- |
| `pnpm build`                            | Sikeres; könyvtár, referencia, parity-oldal és weboldal elkészült                  |
| Típusellenőrzés (`pnpm test` részeként) | Sikeres                                                                            |
| `pnpm test`                             | 16 tesztfájl, 88 sikeres teszt                                                     |
| `pnpm test:gpu`                         | 20 sikeres és 2 kihagyott tesztfájl; 111 sikeres és 55 kihagyott teszt; 547.02 s   |
| `pnpm lint`                             | Sikeres                                                                            |
| `pnpm format:check`                     | Sikeres                                                                            |
| `pnpm release:check`                    | 8 sikeres teszt; 1 Windows alatt kihagyott teszt                                   |
| `pnpm size`                             | Sikeres; 35,212 B / 40,000 B és 4,496 B / 8,000 B                                  |
| `pnpm fidelity:check`                   | A repóban tárolt eredmények épek: 8 jelenet, 1,344 tensor-összehasonlítás, 192 kép |
| Windows-indítók                         | Indítás, már futó példány felismerése és leállítás ellenőrizve                     |
| HTTP-kiszolgálás                        | 200-as válasz a 3300-as és a külön tesztelt 3301-es porton                         |

A GPU-tesztek kihagyásai opcionális benchmarkokat, valódi modell-/fixture-adatokat,
illetve a Node/Dawn környezetből hiányzó `shader-f16` funkciót igénylő ellenőrzéseket
érintenek. A böngészőben a `shader-f16` rendelkezésre állt, és a referenciahálózatot
ott külön futtatva ellenőriztem. A release-teszt egy részét az upstream teszt
Windows alatt automatikusan kihagyja a shell nélküli pnpm-indítás miatt.

A `fidelity:check` a már tárolt összehasonlítások ellenőrzése; ezekhez nem készült
új teljes képi parity-mérés ezen a gépen.

## Böngészős próba a helyi GPU-n

Helyi Chrome-ban, külön tesztprofil használatával ellenőrizve:

- A normál 3D fejmodell és a textúrák betöltődtek.
- WebGPU-adapter: `nvidia · ampere`; `shader-f16` és `timestamp-query` elérhető.
- Mindkét backend használható, eszközkorlát miatti hiba nincs.
- A szintetikus modell létrehozása és GPU-ra feltöltése sikeres.
- **TSL:** a teljes, 451 dispatchből álló hálózat futott 640 × 360 méreten.
- **Reference WGSL:** a teljes hálózat szintén futott 640 × 360 méreten.
- Az osztott nézet és a backendváltás működött.
- A mentett állapotokban nincs alkalmazás-, modellbetöltési vagy GPU-eszközhiba.

Bizonyítékok a `.local/logs` mappában:

- `demo-off.png`
- `demo-tsl-split.png`, `demo-tsl-state.json`
- `demo-reference-split.png`, `demo-reference-state.json`
- `demo-real-tsl-split.png`, `demo-real-tsl-state.json`
- `demo-real-reference-split.png`, `demo-real-reference-state.json`
- `demo-autoload.png`, `demo-autoload-state.json`
- `real-model-parity.json`, `real-model-parity.log`

A betanított modell forrása a felhasználó által megadott Visual Enhancer projekt
v14.0 kiadási ZIP-je, azon belül `bin/runtime/dlssnr/nvngx_dlssnr.dll`.
A ZIP-ből a szükséges DLL és a mellékelt licenc lett beolvasva, ZIP CRC-ellenőrzéssel.
A DLL PE-erőforrását az MLX-DLSS nyílt forrású beolvasója kezelte; a DLL kódját
nem kellett futtatni. A 153 tömörített réteg-adatrekord változatlanul került a
`three-dlss-nr` által várt 11 szakaszba. A tényleges rekordhosszak megmaradtak,
beleértve a szintetikus sablontól eltérő paddinget is.

Az `NRModel.load({files}, {verify:true})` minden szakasz SHA-256 hashét és a modell
153 rekordját sikeresen ellenőrizte. A betanított modell a böngészőben **TSL és
Reference WGSL móddal is végigfutott**, és a feldolgozott fejmodell jelent meg.
A mentett képeken az NR off/on különbség látható.

Az új, betanított modellen futtatott böngészős összehasonlításban a teljes,
451 dispatchből álló hálózat után **199/199 tensor bájtra azonos**, beleértve a
head kimenetet is. A referencia és a TSL hálózat ugyanazon RTX 3090-en futott.
Ez új számítási parity-mérés, külön a fent említett, repóban tárolt képi eredményektől.

DLL SHA-256: `6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927`.
PE-erőforrás SHA-256: `836f445d06ecd2e59bb9f17b84b91c143396fd76ccda1c9dc7fe81d5edd548f4`.
Részletes eredet: `.local/models/nvidia-310.8/provenance.json`.
A betanított modell gyártói licence és eredete megmaradt; a modellfájlok nincsenek
Gitbe felvéve.

Az első telepítési ellenőrzéskor a Git által követett projektfájlok változatlanok
voltak. A későbbi, felhasználó által kért gyorsítás a TSL kerekítését és a GEMM
memóriaelrendezését módosította; részletek a [GYORSITAS.md](GYORSITAS.md) fájlban.
A rögzített OpenDLSS-NR alprojekt változatlan. Az indítók, az automatikus
helyi betöltőoldal és a magyar útmutatók külön helyi kiegészítések. A demószerver
külön fut tovább; leállítására a `stop-demo.cmd` szolgál.

## A gyorsítás ellenőrzése – 2026-10-04

- Build, típusellenőrzés, lint és formázás: sikeres.
- CPU: 16 tesztfájl, 88 sikeres teszt.
- GPU: a teljes futásban 111 sikeres számítási teszt és egy frissítendő
  shader-pillanatkép volt. A pillanatkép javítása után a teljes érintett GEMM
  tesztfájl 27 sikeres / 23 kihagyott teszttel lefutott; az izolált
  pillanatkép-teszt is sikeres. Összesen 112 sikeres / 55 kihagyott GPU-teszt.
- Az új kerekítési teszt minden pozitív és negatív félprecíziós felezőpontot
  és a közvetlen f32 szomszédait ellenőrzi: 190 458 bitminta, két kerekítési helper.
- A valódi modell teljes, 451 dispatches hálózatán, **640 × 360 méreten
  199/199 tensor bájtra azonos** a változatlan referenciával, a head kimenet is.
- Release-ellenőrzés: 8 sikeres / 1 Windows alatt kihagyott teszt.
- Méretellenőrzés: 35 386 B / 40 000 B és 4 496 B / 8 000 B.
- Az automatikus betöltőoldal a valódi modellt Reference WGSL backenddel,
  Natural stílussal és 640 × 360-as Split nézettel tölti be.
- A `?backend=tsl` cím a valódi modellt az optimalizált TSL backenddel indítja;
  a betöltés, a hálózat és a Split nézet ezen az útvonalon is működik.
- A kész demó mért medián FPS-e: optimalizált TSL **11,39**, Reference WGSL
  **14,19**. Az NR off kontrollmérés előtte és utána egyaránt **56,9 FPS**.
- A későbbi, hosszabb TSL képi ellenőrzés állapotában 10,35 FPS szerepel;
  a teljesítmény a GPU aktuális órajelével és a gép terhelésével változhat.

Az új bizonyítékok: `perf-unit.log`, `perf-full-gpu.log`, `perf-gemm-final.log`,
`perf-gemm-isolated.log`, `perf-edges-snapshots.log`, `perf-build.log`,
`perf-real-model-parity.json` és `perf-real-model-parity.log` a `.local/logs` alatt.
Az FPS-mérés: `perf-demo-fps-640x360.json`; képi és állapotbizonyíték:
`perf-demo-reference.png`, `perf-demo-reference-state.json`,
`perf-demo-tsl.png`, `perf-demo-tsl-state.json`.
A teljes GPU-futás naplója megőrzi az első pillanatkép-eltérést; a későbbi
GEMM-naplók igazolják a javított ellenőrzést. A snapshot előtt a layout-helyettesítő
függvények felépülnek, így a deklarációk sorrendje teljes és izolált futásban azonos.
