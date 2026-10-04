# DLSS 5 és Three.js – magyar interaktív bemutató

Az oldal a DLSS 5 neurális renderelésének ötletét vezeti végig a Three.js-en,
az OpenDLSS-NR-en és a three-dlss-nr porton. Közönségnek szánt, közérthető
magyarázat, valódi felvételekkel és játszható 3D szemléltetéssel.

## Megnyitás

`start-presentation.cmd` → <http://localhost:3300/>. A szerver leállítása:
`Ctrl+C` vagy `stop-demo.cmd`. Fejlesztői módban: `pnpm-local.ps1 dev`.
Az eredeti hálózati demó `/demo`, a helyi modell automatikus betöltése
`/local-demo.html`. A nyilvános oldalt önmagában is lehet bemutatni: a képes
elválasztóhoz nem szükséges WebGPU vagy modellbetöltés.

## Rövid bemutatási menet

1. Nyitó animáció: a két térbeli anyag és a parallax a webes 3D szemléltetése.
2. Húzd az elválasztót az arcon, majd válts a fényes autófelületekre.
   A nyílbillentyűk, Home és End is működnek.
3. Mutasd meg a felskálázás, képkockagenerálás és neurális renderelés különbségét.
4. A Three.js színpadon válts Forma, Anyag és Fény között; forgasd a tárgyat.
5. Az OpenDLSS-NR a számítást alkotja újra, a three-dlss-nr azt a webes
   Three.js környezetbe kapcsolja. A kód és a betanított modell licence különálló.
6. Nagyítsd ki a galéria képeit. Escape vagy a Bezárás gomb zárja az ablakot.
7. Az Élő demó gombbal válts át a tényleges neurális feldolgozásra.

Az animációk szüneteltethetők. A rendszer csökkentett mozgás beállítását az
oldal követi. A 3D illusztrációk a látómezőn kívül és háttérben szünetelnek.

## A képek eredete

A `packages/website/public/explainer` JPEG képei a helyi működő WebGPU-demóból
készültek 2026. október 4-én, böngészős képernyőrögzítéssel. Ugyanaz a kamera,
640 × 360-as hálózati felbontás, Natural stílus, Reference WGSL backend és a
korábban helyben beállított modell szerepel az eredeti és neurális felvételen.
A böngésző a hálózati képet nagyobb vászonra jelenítette meg. A képek
nem AI-generált illusztrációk és nem a hivatalos játékos DLSS 5 teljesítménymérései.
A modellsúlyok nem kerülnek a bemutató mellé.

A 3D tárgyak forrásai és licencei a `src/lib/models.ts` nyilvántartásából jelennek
meg az oldalon és a nagyított képeknél. Lee Perry-Smith: CC BY 3.0; Car Concept:
CC BY 4.0, a logók kizárásával; Flight Helmet: CC0; Sheen Wood Leather Sofa:
CC0 és CC BY 4.0. A neurális módosítás a jobb oldali képfélre vonatkozik.
Teljes forrás- és licencekredit a meglévő `public/models` könyvtárakban.

A Space Grotesk és Manrope változó betűk helyben vannak, SIL Open Font License
1.1 szerint, a licencek a `public/fonts` könyvtárban találhatók. Külső
betűszolgáltatás betöltésére nincs szükség.

## Ellenőrzött elsődleges források

- [NVIDIA: DLSS 5 és 3D-Guided Neural Rendering](https://www.nvidia.com/en-us/geforce/news/dlss-5-3d-guided-neural-rendering/)
- [NVIDIA kutatási projekt](https://research.nvidia.com/labs/adlr/DLSS5/)
- [Three.js](https://threejs.org/)
- [OpenDLSS-NR, maan](https://github.com/maanHimself/OpenDLSS-NR)
- [three-dlss-nr, Ben Houston](https://github.com/bhouston/three-dlss-nr)

## Látványterv

Három összehangolt fejezetterv készült a beépített Image Gen eszközzel.
Helyi tervképek: `.local/explainer-design/01-dlss.png`, `02-three-opendlss.png`,
`03-port.png`. A promptok magyar oktatási oldalt, papírszínű hátteret,
grafitszürke és lime palettát, Space Grotesk tipográfiát, nyitott szerkesztett
elrendezést, valódi képernyőképeket és Three.js-ben rajzolt 3D formákat írtak elő.
A tervképek csak tervezési referenciák; kitalált fotóik nem szerepelnek az oldalon.

Tokenek: háttér `#f7f7f2`, szöveg `#171919`, kiemelés `#c0fa67`, másodlagos
szöveg `#575e58`, választóvonal `#c9ccc3`. A fejlécben a három fejezetlink és
az Élő demó gomb szerepel. A nyitó címsor „Ugyanaz a világ. Új fényben.”,
a fő gomb „Mutasd a különbséget”. A fő vizuális motívumok a drótváz,
az összefonódó térbeli formák és a nyitott vízszintes választóvonalak.

Szándékos terveltérések: a generált fotók helyett tényleges fej-, autó-, sisak-
és kanapéfelvételek; a raszteres forma helyett forgatható 3D geometria; az első
terv fejlécének megtartása mindhárom fejezetben. Kiegészítő magyarázat és
animációkapcsoló a felhasználó által kért érthetőség és mozgás miatt.
