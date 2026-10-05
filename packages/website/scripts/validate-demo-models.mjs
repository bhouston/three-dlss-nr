// Validate the new assets locally with the pinned official Khronos validator.
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import validator from 'gltf-validator';

const root = new URL('../', import.meta.url);
const manifest = JSON.parse(await readFile(new URL('demo-model-sources.json', root), 'utf8'));
const supportedExtensions = new Set([
  'KHR_texture_transform',
  'KHR_materials_clearcoat',
  'KHR_materials_transmission',
  'KHR_materials_sheen',
  'KHR_materials_specular',
  'EXT_texture_webp',
]);
for (const model of manifest.models) {
  const bytes = await readFile(new URL(`public/models/${model.id}/${model.filename}`, root));
  const hash = createHash('sha256').update(bytes).digest('hex');
  const license = await readFile(new URL(`public/models/${model.id}/SOURCE-LICENSE.txt`, root));
  if (
    createHash('sha256').update(license).digest('hex') !==
    model.sources.find((source) => source.name === 'LICENSE.md').sha256
  ) {
    throw new Error(`${model.id}: upstream license notice hash mismatch`);
  }
  if (bytes.length !== model.byteLength || hash !== model.sha256) throw new Error(`${model.id}: hash/size mismatch`);
  if (bytes.length > 12 * 1024 * 1024) throw new Error(`${model.id}: exceeds 12 MiB asset budget`);
  if (bytes.readUInt32LE(0) !== 0x46546c67 || bytes.readUInt32LE(4) !== 2 || bytes.readUInt32LE(8) !== bytes.length) {
    throw new Error(`${model.id}: invalid GLB header`);
  }
  const document = JSON.parse(bytes.subarray(20, 20 + bytes.readUInt32LE(12)).toString());
  if (document.buffers.some((buffer) => buffer.uri) || (document.images ?? []).some((image) => image.uri)) {
    throw new Error(`${model.id}: external/data resource URI is forbidden in these self-contained GLBs`);
  }
  for (const extension of document.extensionsUsed ?? []) {
    if (!supportedExtensions.has(extension))
      throw new Error(`${model.id}: unreviewed GLTFLoader extension ${extension}`);
  }
  const report = await validator.validateBytes(new Uint8Array(bytes), {
    uri: model.filename,
    externalResourceFunction: () => {
      throw new Error('unexpected external resource');
    },
  });
  console.log(
    `${model.id}: ${bytes.length.toLocaleString()} bytes; ${report.issues.numErrors} errors, ${report.issues.numWarnings} warnings`,
  );
  for (const message of report.issues.messages.filter((issue) => issue.severity <= 1))
    console.log(JSON.stringify(message));
  if (report.issues.numErrors) throw new Error(`${model.id}: glTF validation errors`);
}
