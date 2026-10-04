# three-dlss-nr – helyi futtatás Windows alatt

A GitHub-projekt és a rögzített OpenDLSS-NR alprojekt telepítve van a
`D:\AI\three-dlss-nr` mappában. A demó a helyi RTX 3090-en, a böngésző WebGPU
felületén fut. A fej, a sportautó, a kanapé, a róka és a pilótasisak textúráikkal
együtt helyben vannak. A modelleket a [JELENETEK.md](JELENETEK.md) mutatja be.

## Indítás

Kattints duplán a `start-demo.cmd` fájlra, vagy futtasd PowerShellben:

```powershell
cd D:\AI\three-dlss-nr
.\start-demo.cmd
```

Cím: <http://localhost:3300/local-demo.html>. Az indító Chrome-ban (vagy Edge-ben)
nyitja meg az oldalt, a szerver pedig a terminálban fut. A helyi betanított modell
automatikusan betöltődik, és az osztott nézet indul el 640 × 360-as méreten,
Natural stílussal, a gépen gyorsabb **Reference WGSL** backenddel. A sportautó
jelenik meg álló kamerával; a **Scene → Model / scene** mezőben válthatsz modellt.
Az első shaderfordítás több tíz másodpercet igénybe vehet.
Újraindításkor felismeri a már futó saját szervert. Más program által használt
portot nem foglal el és más programot nem állít le.

Leállítás: `Ctrl+C` a szerver termináljában, vagy `stop-demo.cmd`. Ez utóbbi
az ellenőrzéshez elindított háttérben futó példányt is leállítja.

Másik port használata:

```powershell
.\start-demo.cmd -Port 3301
.\stop-demo.cmd -Port 3301
```

## A neurális hálózat kipróbálása

1. Várd meg az automatikus modellbetöltést és a shaderfordítást.
2. A **Split** nézetben bal oldalon az eredeti, jobb oldalon a neurálisan
   feldolgozott kép jelenik meg. A középső elválasztó húzható.
3. A **View** részen választhatod az **NR off**, **NR on** és **Split** nézetet.
4. A **Backend** részen válthatsz a TSL és Reference WGSL megvalósítás között.
   Alapértelmezés a Reference WGSL. Az optimalizált TSL közvetlen címe:
   <http://localhost:3300/local-demo.html?backend=tsl>.
5. A **Scene → Model / scene** választóban öt modell van. Mindegyik saját
   kezdő kameranézetet kap; ehhez a **Reset camera** gombbal térhetsz vissza.
6. A **Scene → Resolution** mezőben emelheted a felbontást. Ez a WebGPU-port
   jelenleg nem ad játékokra jellemző 60 fps-es neurális feldolgozást.

A betanított modell a Visual Enhancer v14.0 kiadásának
`bin/runtime/dlssnr/nvngx_dlssnr.dll` fájljából származik (310.8.SF.0).
A modell adatai változatlan bájtokkal kerültek a `three-dlss-nr` által várt
`manifest.json` + `model/stages/*.bin` formátumba.

Modellmappa: `D:\AI\three-dlss-nr\.local\models\nvidia-310.8`.
Az eredeti DLL, a forrásadatok és a NVIDIA licenc helyben megmaradtak.
A modellre a saját gyártói licence vonatkozik; nem vált MIT-licencűvé attól,
hogy az alkalmazás forrása MIT-licencű. A modellfájlok Git által figyelmen kívül
hagyott helyi könyvtárban vannak.

Forráskiadás: <https://github.com/Merserk/dlss5-visual-enhancer/releases/tag/v14.0>.
A beolvasó eszköz: <https://github.com/iamwavecut/MLX-DLSS>.

Az eredeti demó, automatikus modellbetöltés nélkül: <http://localhost:3300>.
Ezen az oldalon a **Load model directory…** gombbal választhatod ki a fenti
modellmappát vagy egy másik kompatibilis modellt.

A **Synthetic weights** gomb kizárólag szintetikus tesztsúlyokat generál;
ezek színes, értelmetlen kimenetet adnak. A lemezre is elkészített tesztmodell
a `.local/models/synthetic` mappában található.

## Fejlesztés és újrafordítás

A helyi Node.js **26.10.0** és pnpm **11.27.0** a `.local/tools` mappában található.
A globális Node.js telepítése nem változott. A `pnpm-local.ps1` ezeket a helyi
verziókat használja; nincs szüksége a globális Corepackra.

```powershell
cd D:\AI\three-dlss-nr
.\pnpm-local.ps1 install --frozen-lockfile
.\pnpm-local.ps1 build
.\start-demo.cmd
```

Fejlesztői módhoz először állítsd le a normál demószervert:

```powershell
.\stop-demo.cmd
.\pnpm-local.ps1 dev
```

A fejlesztői oldal címe <http://localhost:3300/local-demo.html>; leállítás: `Ctrl+C`.
Az automatikus betöltőoldal a `packages/website/public/local-demo.html` fájl;
a `private-model` helyi könyvtárhivatkozás a fenti modellmappára mutat.

Az eredeti TSL számításban pontos félprecíziós kerekítési gyorsítás és a mátrixszorzás
megosztott memóriájának jobb elrendezése került be. A rögzített OpenDLSS-NR alprojekt
és a függőségek verziói megmaradtak. A Windows-indítók és ez az útmutató külön
hozzáadott helyi fájlok. A méréseket a [GYORSITAS.md](GYORSITAS.md) foglalja össze.
Az ellenőrzési állományok a `.local/logs` mappában találhatók.
