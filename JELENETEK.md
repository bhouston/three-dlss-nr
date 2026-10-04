# Modellek a neurális renderelés bemutatásához

Demó: <http://localhost:3300/local-demo.html>. A **Scene → Model / scene**
választóban az eredeti fej mellett négy új modell érhető el. Mindegyik helyben
van, a textúráival és a licencével együtt.

| Modell                      | Mit érdemes nézni az NR off / on összehasonlításban? | GLB méret |
| --------------------------- | ---------------------------------------------------- | --------: |
| Sportautó – Car Concept     | Fényezés, gumifelületek, üveg, műszerfal és fények   |  9,58 MiB |
| Bőrkanapé és mintás párnák  | Bőr erezete, szövetminták, párnák széle, faanyag     |  9,64 MiB |
| Róka                        | A stilizált állat színe, árnyalása és formája        |  0,15 MiB |
| Pilótasisak – Flight Helmet | Kopott bőr, varrások, fém, szemüveglencse, faállvány | 13,61 MiB |

A helyi betöltőoldal a sportautóval indul, álló kamerával, 640 × 360-as
Natural / Split nézetben, Reference WGSL backenddel. Balra az eredeti render,
jobbra a neurálisan újrarenderelt kép látható. Az elválasztó húzható.
Az egérrel forgathatsz és közelíthetsz; a **Reset camera** a kiválasztott
modell saját kezdőnézetéhez tér vissza.

Közvetlen címek:

- [Sportautó](http://localhost:3300/local-demo.html?scene=car-concept)
- [Kanapé](http://localhost:3300/local-demo.html?scene=leather-sofa)
- [Róka](http://localhost:3300/local-demo.html?scene=fox)
- [Pilótasisak](http://localhost:3300/local-demo.html?scene=flight-helmet)
- [Eredeti fej](http://localhost:3300/local-demo.html?scene=lee-perry-smith)

Az optimalizált TSL például így kérhető:
<http://localhost:3300/local-demo.html?scene=car-concept&backend=tsl>.
A korábban megnyitott demót `Ctrl+F5`-tel frissítsd.

## Források és előkészítés

A négy új modell a Khronos glTF Sample Assets gyűjteményéből származik,
az `edc7c9e67c639d230715049ee31f9a96a6babbbe` commitból:

- [Car Concept](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/edc7c9e67c639d230715049ee31f9a96a6babbbe/Models/CarConcept):
  Eric Chadwick / Darmstadt Graphics Group; CC BY 4.0, a logók külön megjelölésével.
- [Sheen Wood Leather Sofa](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/edc7c9e67c639d230715049ee31f9a96a6babbbe/Models/SheenWoodLeatherSofa):
  Fran Calvente és Eric Chadwick / Darmstadt Graphics Group; CC0 és CC BY 4.0.
- [Fox](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/edc7c9e67c639d230715049ee31f9a96a6babbbe/Models/Fox):
  PixelMannen, tomkranis, AsoboStudio és scurest; CC0 és CC BY 4.0.
- [Flight Helmet](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/edc7c9e67c639d230715049ee31f9a96a6babbbe/Models/FlightHelmet):
  Gary Hsu glTF-konverziója; CC0.

Az önálló GLB-k glTF Transform 4.5.1-gyel készültek. A PNG/JPEG textúrák
legnagyobb mérete 1024 pixel, aránytartó méretezéssel és veszteségmentes WebP
csomagolással; a forrásban lévő WebP textúrák megmaradtak. A geometriát nem
egyszerűsítettem és nem kvantáltam. A beágyazott PBR anyagok megmaradtak.
Az egyes modellekhez saját kameranézet és méret tartozik.

A modellek a `packages/website/public/models` alatt találhatók. Minden új modell
mellett `ATTRIBUTION.txt`, `LICENSE.md`, `metadata.json` és `PROVENANCE.json`
van. A PROVENANCE az elkészült GLB SHA-256 hashét és a módosításokat is rögzíti.
A forrásfájlok a `.local/scene-sources` alatt megmaradtak.
