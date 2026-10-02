# Suggested head assets

These are human head models for the demo's model switcher. A neural renderer like DLSS-NR / OpenDLSS-NR changes skin, tone and facial structure, so we want faces that show those changes clearly. Each entry records the exact license and the attribution text the demo UI must show.

Licensing policy: any Creative Commons license is acceptable. **NC (NonCommercial) and ND (NoDerivatives) licenses are unsuitable for a public demo**, because a neurally re-rendered frame is arguably a derivative work. Entries with extra restrictions, such as "NoAI" clauses, are flagged too.

Last verified: 2026-10-02. Before committing a downloaded file, re-check its license on the source page, because Sketchfab authors can change licenses.

## Default (already in the repo / fetched)

### 1. Lee Perry-Smith head scan ("Infinite, 3D Head Scan"): DEFAULT

|                |                                                                                                                                                                                                                                                                                                    |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author         | Lee Perry-Smith / Infinite-Realities (ir-ltd.net)                                                                                                                                                                                                                                                  |
| License        | **CC BY 3.0 Unported** (not CC0): https://creativecommons.org/licenses/by/3.0/                                                                                                                                                                                                                     |
| Source         | https://github.com/mrdoob/three.js/tree/dev/examples/models/gltf/LeePerrySmith (license notice: `LeePerrySmith_License.txt` in that folder)                                                                                                                                                        |
| Account needed | No (raw GitHub URLs)                                                                                                                                                                                                                                                                               |
| Format         | GLB (mesh only, about 9.3k vertices / 17.7k triangles, normals + UV0, no embedded textures) plus separate JPG maps                                                                                                                                                                                 |
| Size           | 405 KB GLB. Maps: albedo `Map-COL.jpg` 1K, specular `Map-SPEC.jpg` 1K, tangent normal `Infinite-Level_02_Tangent_SmoothUV.jpg` 1K, displacement 4K. About 2.1 MB in total                                                                                                                          |
| Quality        | A real photogrammetry scan of a shaved male head, so it has no hair cards to deal with. The mesh is decimated, but the normal map keeps pore and wrinkle detail. The 1K albedo is soft up close.                                                                                                   |
| Why for NR     | It is the canonical real-time skin test head (three.js SSS, decal and normal-map demos all use it). Neutral lighting makes NR's changes to skin tone and structure easy to see. It is small and loads instantly.                                                                                   |
| Notes          | The GLB's material has no textures, so the loader must assign the maps by hand (as the three.js examples do: `map`, normal map and optional displacement/specular). Bounding box is about 8.6 x 7.9 x 5.2 units, centred at the origin. The file also contains an unused second scene with a lamp. |

**Attribution to show:**

> "Infinite, 3D Head Scan" by Lee Perry-Smith (Infinite-Realities), licensed under [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). glTF via three.js examples.

## Recommended alternates (ordered)

### 2. Nemetona_NatureBeauty

|                |                                                                                                                                                                                                                               |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author         | JOJObrush (https://sketchfab.com/JOJObrush)                                                                                                                                                                                   |
| License        | **CC BY 4.0**: https://creativecommons.org/licenses/by/4.0/                                                                                                                                                                   |
| Source         | Sketchfab: https://skfb.ly/pAQvU. A ready-made GLB copy is also shipped in three.js: https://github.com/mrdoob/three.js/blob/dev/examples/models/gltf/nemetona.glb (attribution in `examples/webgpu_postprocessing_sss.html`) |
| Account needed | Sketchfab: yes (login to download). three.js copy: **no**                                                                                                                                                                     |
| Format         | GLB (three.js copy). Sketchfab original: glTF/GLB/source                                                                                                                                                                      |
| Size           | three.js GLB 3.8 MB. Sketchfab original: 323.6k triangles / 156.4k vertices                                                                                                                                                   |
| Quality        | Sculpted (ZBrush), Substance-painted female head with pore and wrinkle detail baked into normal maps. Semi-realistic and stylised-beautiful rather than a scan.                                                               |
| Why for NR     | Female face. three.js uses it as the SSS showcase, so it is tuned for skin rendering. It contrasts well with the male scan, and NR changes to makeup, skin tone and facial proportions should show clearly.                   |

**Attribution to show:**

> "Nemetona_NatureBeauty" by [JOJObrush](https://sketchfab.com/JOJObrush), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

### 3. Cowboy Gramps

|                |                                                                                                                                                                                                           |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author         | Muhammed Ismayil                                                                                                                                                                                          |
| License        | **CC0** (public domain), per BlendKit asset page                                                                                                                                                          |
| Source         | https://www.blendkit.com/asset-gallery-detail/96dce188-9c9c-4699-a45a-48663fbbbcb7/                                                                                                                       |
| Account needed | **Yes**: free BlendKit account and/or the BlendKit Blender add-on                                                                                                                                         |
| Format         | Blender `.blend` (Eevee/Cycles materials). **Must be exported to GLB in Blender**; procedural or Cycles shader nodes may need baking to textures first.                                                   |
| Size           | 72.6 MiB (.blend), 327k polygons. Expect to decimate and resize textures to 2K to get under about 30 MB.                                                                                                  |
| Quality        | High-quality stylised-realistic old man with hair, beard, hat and eyes (from "Tiny Eye").                                                                                                                 |
| Why for NR     | This is the asset the OpenDLSS-NR README uses for its NR on/off comparison, so we can compare directly against the reference implementation. Aged skin, facial hair and a hat brim are hard cases for NR. |

**Attribution (not required for CC0, but courteous):**

> "Cowboy Gramps" by Muhammed Ismayil (BlendKit), CC0.

### 4. Marcus: Free 3D Head Sample Model Scan

|                |                                                                                                                                                                                                                                                                             |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author         | Digital Reality Lab (https://sketchfab.com/digitalrealitylab)                                                                                                                                                                                                               |
| License        | **CC BY 4.0** (Sketchfab listing): https://creativecommons.org/licenses/by/4.0/                                                                                                                                                                                             |
| Source         | https://sketchfab.com/3d-models/marcus-free-3d-head-sample-model-scan-1aef740047ba4214998f068355a5f034                                                                                                                                                                      |
| Account needed | Yes (Sketchfab login)                                                                                                                                                                                                                                                       |
| Format         | glTF/GLB via Sketchfab's auto-conversion                                                                                                                                                                                                                                    |
| Size           | 498k triangles / 250k vertices. Likely needs decimation and texture resize (gltf-transform `simplify` + `resize --width 2048` + `webp`).                                                                                                                                    |
| Quality        | Professional photogrammetry head scan of a real man. Most photoreal option in this list.                                                                                                                                                                                    |
| Why for NR     | A real-skin scan at high texture resolution, the closest to what NR was trained to "re-render". Good for checking whether NR keeps someone's identity.                                                                                                                      |
| Caveat         | Use the **Sketchfab CC BY 4.0 copy** only. Digital Reality Lab's own website downloads (https://www.digitalrealitylab.com/sample-model/) come under a different custom license that forbids "free distribution on other websites", so do not use those for the public repo. |

**Attribution to show:**

> "Marcus - Free 3D Head Sample Model Scan" by [Digital Reality Lab](https://sketchfab.com/digitalrealitylab), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

### 5. Head scan 13 (photogrammetry)

|                |                                                                                                                                             |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Author         | yaro.pro (Sketchfab)                                                                                                                        |
| License        | **CC BY 4.0**: https://creativecommons.org/licenses/by/4.0/                                                                                 |
| Source         | https://sketchfab.com/3d-models/head-scan-13-photogrammetry-5e6d2804405449e6b3bd96cd12d8b1ab                                                |
| Account needed | Yes (Sketchfab login)                                                                                                                       |
| Format         | glTF/GLB via Sketchfab                                                                                                                      |
| Size           | 383.6k triangles / 192.4k vertices. Decimate.                                                                                               |
| Quality        | Photogrammetry of a mature man, captured on an iPhone. The lighting is baked into the albedo and the texture is noisier than a studio scan. |
| Why for NR     | Older face with baked lighting. Shows how NR handles "dirty" real-world capture data.                                                       |

**Attribution to show:**

> "Head scan 13 (photogrammetry)" by yaro.pro, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

### 6. Smithsonian Open Access portrait busts / life masks (CC0, no account)

|                |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Author         | Smithsonian Institution, National Portrait Gallery (Digitization Program Office)                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| License        | **CC0**: the Open Access API marks each 3D package `"usage": {"access": "CC0"}`. See https://www.si.edu/openaccess and https://3d.si.edu/cc0                                                                                                                                                                                                                                                                                                                                                                           |
| Candidates     | George Washington bust (Houdon): https://3d.si.edu/object/3d/george-washington:ff28cb3a-ad00-43b3-a928-fa61ab0a288f. glTF zip (7.7 MB): `https://3d-api.si.edu/content/document/3d_package:ff28cb3a-ad00-43b3-a928-fa61ab0a288f/resources/NPG_78_1-GeorgeWashington-150k-4096-gltf_std.zip`<br>Abraham Lincoln life mask (Clark Mills, 1865): glTF zip (7.5 MB): `https://3d-api.si.edu/content/document/3d_package:c02c239d-5ebf-4a7a-a368-e2288bbf4b31/resources/abraham-lincoln-mills-life-mask-150k-4096-gltf.zip` |
| Account needed | **No** (direct HTTP download, verified 200 OK)                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Format         | glTF (+ bin + 4K textures) in a zip. 150k faces. Pack to GLB with `gltf-transform copy in.gltf out.glb`.                                                                                                                                                                                                                                                                                                                                                                                                               |
| Quality        | Accurate scans, but of **sculptures/plaster** (white or terracotta material), not skin. The Lincoln life mask is a cast of a real face.                                                                                                                                                                                                                                                                                                                                                                                |
| Why for NR     | Most interesting as a "does NR hallucinate skin onto a statue?" test case. These are the only fully unrestricted, no-login options found.                                                                                                                                                                                                                                                                                                                                                                              |

**Attribution (not required for CC0; suggested):**

> "George Washington" bust by Jean-Antoine Houdon, 3D scan: Smithsonian National Portrait Gallery, CC0.<br>
> "Abraham Lincoln" life mask cast after Clark Mills, 3D scan: Smithsonian National Portrait Gallery, CC0.

### 7. Tennyson bust (Three D Scans), lower priority

|         |                                                                                                                                                                                                                     |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Source  | https://threedscans.com/lincoln/tennyson/ (STL). A 462 KB GLB copy is in three.js: `examples/models/gltf/tennyson-bust.glb`, credited as "Tennyson bust from Three D Scans" in `webgpu_postprocessing_ao.html`      |
| License | **Not stated** on the scan page or the site's info page (Three D Scans is generally described as having no copyright restrictions, but we could not find an explicit license). Get confirmation before shipping it. |
| Notes   | Untextured plaster bust, the same "statue" test case as the Smithsonian busts.                                                                                                                                      |

## Checked and not recommended

| Asset                                                       | Why not                                                                                                                                                          |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Nefertiti (three.js `examples/models/gltf/Nefertiti`)       | **CC BY-NC** (Fraunhofer IGD, per its README.md). Non-commercial, so it does not suit a public demo. It is also a painted sculpture.                             |
| Venice Mask (three.js `venice_mask.glb`)                    | **CC BY-NC 4.0** (DailyArt). It is also a mask, not a face.                                                                                                      |
| Digital Emily 2 (Wikihuman / USC ICT)                       | Excellent 8K skin data, but **licensed for non-commercial use only**. It also comes as Alembic/Maya files and needs heavy conversion.                            |
| Head scan by NumoScan (Sketchfab)                           | CC BY 4.0, but the author adds a **"NoAI" restriction** (it may not be used as input to generative AI programs). A neural renderer arguably counts, so avoid it. |
| Digital Reality Lab website downloads                       | Custom license: credit is required, and redistribution on other websites is forbidden. Use the Sketchfab CC BY copy of Marcus instead.                           |
| facecap.glb (three.js)                                      | Credited only as "model by Face Cap" (bannaflak.com), with no explicit license found. It is also a low-detail ARKit blendshape head.                             |
| kira.glb, Michelle.glb, Xbot, readyplayer.me.glb (three.js) | Full-body or stylised characters, not heads. Licenses are mixed (Unity Asset Store and Mixamo sources).                                                          |

## Demo UI attribution requirements

- **CC BY / CC BY-SA** assets need visible credit wherever the model is shown: title, author, license name with a link, and a note that it was modified (for example, decimated, re-encoded or re-rendered by NR). A small credits line under the viewport that updates with the model switcher is enough.
- **CC BY-SA** (none currently recommended): modified versions of the asset must be redistributed under the same license.
- **CC0** needs no attribution, but credit is still suggested.
- Keep a `LICENSE.txt` / `ATTRIBUTION.txt` next to every model file in the repo.
- Sketchfab downloads need a (free) account. Record the exact license shown on the model page at download time.
