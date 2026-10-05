#!/usr/bin/env python3
"""Reproduce vetted demo assets from pinned Khronos sources. Requires Pillow==12.3.0."""
import hashlib
import io
import json
from pathlib import Path
import struct
import urllib.request

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'demo-model-sources.json'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fetch(entry):
    data = urllib.request.urlopen(entry['url'], timeout=60).read()
    if digest(data) != entry['sha256']:
        raise ValueError(f"source hash mismatch: {entry['url']}")
    return data


def pack_helmet(files):
    document = json.loads(files['SciFiHelmet.gltf'])
    binary = bytearray(files[document['buffers'][0]['uri']])
    document['buffers'][0].pop('uri')
    for image in document['images']:
        texture = Image.open(io.BytesIO(files[image.pop('uri')]))
        texture.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        encoded = io.BytesIO()
        texture.save(encoded, format='PNG', compress_level=9)
        binary.extend(b'\0' * (-len(binary) % 4))
        view = {'buffer': 0, 'byteOffset': len(binary), 'byteLength': len(encoded.getvalue())}
        image['bufferView'] = len(document['bufferViews'])
        image['mimeType'] = 'image/png'
        document['bufferViews'].append(view)
        binary.extend(encoded.getvalue())
    document['buffers'][0]['byteLength'] = len(binary)
    encoded = json.dumps(document, separators=(',', ':'), ensure_ascii=True).encode()
    encoded += b' ' * (-len(encoded) % 4)
    binary.extend(b'\0' * (-len(binary) % 4))
    length = 12 + 8 + len(encoded) + 8 + len(binary)
    return (struct.pack('<III', 0x46546C67, 2, length) + struct.pack('<II', len(encoded), 0x4E4F534A)
            + encoded + struct.pack('<II', len(binary), 0x004E4942) + binary)


def main():
    manifest = json.loads(MANIFEST.read_text())
    for model in manifest['models']:
        files = {entry['name']: fetch(entry) for entry in model['sources']}
        output = pack_helmet(files) if model['id'] == 'sci-fi-helmet' else files[model['filename']]
        if len(output) != model['byteLength'] or digest(output) != model['sha256']:
            raise ValueError(f"prepared asset mismatch: {model['id']}; use the pinned Pillow version")
        directory = ROOT / 'public' / 'models' / model['id']
        directory.mkdir(parents=True, exist_ok=True)
        (directory / model['filename']).write_bytes(output)
        (directory / 'SOURCE-LICENSE.txt').write_bytes(files['LICENSE.md'])
        print(f"{model['id']}: verified {len(output):,} bytes, SHA-256 {digest(output)}")


if __name__ == '__main__':
    main()
