# Helyi gyorsítás az RTX 3090-en

A TSL hálózatban két gyorsítás került be, a modell és a 640 × 360-as képméret
megtartásával. A helyi automatikus betöltőoldal alapértelmezett backendje a gépen
gyorsabb **Reference WGSL** lett.

Demó: <http://localhost:3300/local-demo.html>.
Optimalizált TSL: <http://localhost:3300/local-demo.html?backend=tsl>.
A korábban megnyitott oldalt `Ctrl+F5`-tel frissítsd.

## A beépített megoldások

- A félprecíziós kerekítés a normál számokhoz rövid, pontos egészszámos utat
  használ. A szubnormális számok, nullák, NaN-ok és túlcsordulások eredeti
  kezelése megmarad. A kerekítés továbbra is round-to-nearest-even.
- Az FP8 mátrixszorzás munkacsoporton belüli memóriája paddinget kapott.
  Az ötlet az OpenDLSS-NR
  [padded.js](https://github.com/maanHimself/OpenDLSS-NR/blob/9d08f4184bbcb9d858e2fb7a7834ec0837a9d2f1/ports/browser-webgpu/src/matmul/padded.js)
  megoldásából származik, a TSL eltérő szálelrendezéséhez igazítva.
  A szorzások és összeadások sorrendje megmarad; a megosztott memória
  16-ról 18 KiB-ra nő.

Az exponensek további tömörítését és a memóriaelemek összevonását is kipróbáltam.
Ezek ezen a GPU-n lassabbak lettek, ezért nem kerültek be. A félprecíziós
visszaalakítás külön gyorsítása sem adott egyértelmű nyereséget.

## A hálózat GPU-ideje

Valódi helyi modell, 640 × 360-as bemenet, 640 × 384-re kitöltött mező,
5 bemelegítő és 30 mért képkocka, ugyanazon RTX 3090-en. A táblázat mediánokat
mutat; ez a neurális hálózat ideje, a demó teljes FPS-e ettől külön mérés.

| Megvalósítás                     | GPU-idő / képkocka |
| -------------------------------- | -----------------: |
| Eredeti TSL                      |           95,41 ms |
| TSL, pontos rövid kerekítési út  |           83,00 ms |
| TSL, kerekítés és memóriapadding |           80,15 ms |
| Reference WGSL                   |           59,25 ms |

A megtartott TSL-változat GPU-ideje **16%-kal rövidebb**, az elméleti hálózati
áteresztőképesség körülbelül **19%-kal nagyobb**. A 100%-os GPU-terhelés továbbra is
előfordulhat: folyamatos feldolgozásnál a felszabaduló kapacitást további
képkockák számítására használja az alkalmazás.

Mérési adatok: `.local/logs/perf-baseline-network.json`,
`perf-half-network.json`, `perf-padding-network.json`.

## A kész demó FPS-e

Chrome-ban, 640 × 360 méreten, Natural stílussal, a valódi modell NR on nézetében.
Módonként 7 másodperc bemelegítés után 24 minta készült, 500 ms-os közökkel.
Az értékek mediánok; a kijelzés teljes képkockasebességét mérik.

| Mód                               |   FPS |
| --------------------------------- | ----: |
| Eredeti TSL, korábbi alapmérés    |  9,48 |
| Optimalizált TSL                  | 11,39 |
| Reference WGSL, új alapértelmezés | 14,19 |

A TSL javulása körülbelül **20%**. A Reference WGSL alapértelmezés a korábbi
TSL-hez képest körülbelül **50%-kal több FPS-t** ad. A Reference WGSL kódja
változatlan; itt a gyorsabb meglévő backend kiválasztása adja a nyereséget.
A mérések eltérhetnek a GPU órajelétől, más programok terhelésétől és a böngésző
környezetétől. A célzott shadergyorsítás külön, ugyanazon hálózati benchmarkkal
mért 16%-os GPU-időnyeresége a fenti táblázatban látható.

Bizonyíték: `.local/logs/fps-benchmark-640x360.json` és
`.local/logs/perf-demo-fps-640x360.json`.

A hosszabb TSL képi ellenőrzésénél 10,35 FPS is előfordult, 85,88 ms hálózati
GPU-idő mellett. A későbbi GPU-leolvasás 81 °C-ot és 1635 MHz-es grafikus órajelet
mutatott. A 11,39 FPS a megadott mérési ablak eredménye; hosszabb futásban a gép
aktuális állapotával együtt változhat. A képi állapot:
`.local/logs/perf-demo-tsl-state.json`.

## Mit lehet átvenni a többi projektből?

Az OpenDLSS-NR böngészős referenciahálózata már része ennek a projektnek; ezt
használja a Reference WGSL backend. A natív változat FP8 Tensor Core műveleteket,
PTX-et, összevont kerneleket és Vulkan-bővítményeket használ. A dokumentált
[natív követelmény](https://github.com/maanHimself/OpenDLSS-NR#requirements)
Ada vagy újabb NVIDIA GPU; az RTX 3090 Ampere. A böngészős WebGPU út nem adja
ugyanezeket a műveleteket, így a natív teljesítményszámok közvetlenül nem
érhetők el a TSL kódba másolással.

A [Visual Enhancer](https://github.com/Merserk/dlss5-visual-enhancer) natív
Neuroframe Engine-t használ. A
[DLSS5-Swapper](https://github.com/rakanki911/DLSS5-Swapper) és a
[DLSS-5-MANAGER](https://github.com/NODIX-TECH/DLSS-5-MANAGER) natív futtatókat,
játékintegrációkat és telepítőket kezelnek. Ezekből nem találtam további,
közvetlenül átvehető WebGPU számítási gyorsítást. A natív DLL-ek böngészős
használata külön natív feldolgozószolgáltatást és képtovábbítást igényelne.
