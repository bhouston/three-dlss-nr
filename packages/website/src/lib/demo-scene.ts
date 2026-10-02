import * as THREE from 'three/webgpu';

export interface DemoScene {
  setNeuralRendering(enabled: boolean): void;
  dispose(): void;
}

/** A small lit scene rendered with WebGPURenderer, sized to `container`. */
export async function createDemoScene(container: HTMLElement): Promise<DemoScene> {
  const renderer = new THREE.WebGPURenderer({ antialias: true });
  await renderer.init();
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x202024);
  const camera = new THREE.PerspectiveCamera(45, 16 / 9, 0.1, 100);
  camera.position.set(0, 1.2, 4);
  camera.lookAt(0, 0, 0);

  scene.add(new THREE.HemisphereLight(0xffffff, 0x404040, 1.5));
  const sun = new THREE.DirectionalLight(0xffffff, 2);
  sun.position.set(3, 4, 2);
  scene.add(sun);

  const knot = new THREE.Mesh(
    new THREE.TorusKnotGeometry(0.8, 0.28, 160, 24),
    new THREE.MeshStandardNodeMaterial({ color: 0xd4a017, roughness: 0.3, metalness: 0.8 }),
  );
  scene.add(knot);

  const resize = () => {
    const width = container.clientWidth;
    const height = container.clientHeight;
    renderer.setSize(width, height);
    camera.aspect = width / Math.max(1, height);
    camera.updateProjectionMatrix();
  };
  const observer = new ResizeObserver(resize);
  observer.observe(container);
  resize();

  renderer.setAnimationLoop((time: number) => {
    knot.rotation.set(time * 0.0003, time * 0.0005, 0);
    renderer.render(scene, camera);
  });

  return {
    setNeuralRendering(enabled) {
      // Placeholder until the NR pass exists: a visible background tint.
      scene.background = new THREE.Color(enabled ? 0x1d2a3a : 0x202024);
    },
    dispose() {
      renderer.setAnimationLoop(null);
      observer.disconnect();
      knot.geometry.dispose();
      knot.material.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    },
  };
}
