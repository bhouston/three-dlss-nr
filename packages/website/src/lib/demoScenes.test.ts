import * as THREE from 'three/webgpu';
import { expect, it, vi } from 'vitest';
import { DemoController } from './demo';
import { headModel } from './models';
import { modelFitDistance } from './modelFraming';
import { loadHead } from './studio';

vi.mock('./studio', () => ({ createStudio: vi.fn(), loadHead: vi.fn() }));

function loaded(radius = 1.5) {
  return { object: new THREE.Group(), radius, dispose: vi.fn() };
}
function fixture() {
  const controller = new DemoController();
  const internals = controller as any;
  internals.studio = { stage: new THREE.Group(), scene: new THREE.Scene() };
  internals.camera = new THREE.PerspectiveCamera(26, 16 / 9, 0.1, 100);
  internals.controls = { target: new THREE.Vector3(), update: vi.fn(), maxDistance: 12 };
  internals.pass = { resetHistory: vi.fn() };
  return { controller, internals };
}

it('retains the last working model/selection on failure, displays an English error and permits retry', async () => {
  const { controller, internals } = fixture();
  const original = loaded();
  vi.mocked(loadHead).mockResolvedValueOnce(original);
  await controller.setModel('lee-perry-smith');
  vi.mocked(loadHead).mockRejectedValueOnce(new Error('fixture network failure'));
  await controller.setModel('toy-car');
  expect(controller.getState().modelId).toBe('lee-perry-smith');
  expect(controller.getState().requestedModelId).toBeNull();
  expect(controller.getState().modelError?.message).toMatch(/Could not load Toy car/);
  expect(original.dispose).not.toHaveBeenCalled();
  const car = loaded();
  vi.mocked(loadHead).mockResolvedValueOnce(car);
  await controller.setModel(controller.getState().modelError!.modelId);
  expect(controller.getState().modelId).toBe('toy-car');
  expect(controller.getState().modelError).toBeNull();
  expect(original.dispose).toHaveBeenCalledOnce();
  expect(internals.studio.stage.children).toEqual([car.object]);
  expect(internals.pass.resetHistory).toHaveBeenCalledTimes(2);
});

it('ignores a superseded different-scene load and resets camera to the selected framing', async () => {
  const { controller, internals } = fixture();
  let complete!: (value: ReturnType<typeof loaded>) => void;
  vi.mocked(loadHead).mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        complete = resolve;
      }),
  );
  const pending = controller.setModel('toy-car');
  const sofa = loaded(1.25);
  vi.mocked(loadHead).mockResolvedValueOnce(sofa);
  await controller.setModel('sheen-sofa');
  const stale = loaded();
  complete(stale);
  await pending;
  expect(controller.getState().modelId).toBe('sheen-sofa');
  expect(stale.dispose).toHaveBeenCalledOnce();
  expect(internals.studio.stage.children).toEqual([sofa.object]);
  const entry = headModel('sheen-sofa');
  const distance = modelFitDistance(sofa.radius, 26, 16 / 9);
  const expected = new THREE.Vector3(...entry.presentation!.cameraDirection).normalize().multiplyScalar(distance);
  expect(internals.camera.position.distanceTo(expected)).toBeLessThan(1e-8);
  internals.camera.position.set(99, 99, 99);
  controller.resetCamera();
  expect(internals.camera.position.distanceTo(expected)).toBeLessThan(1e-8);
  expect(internals.studio.scene.environmentIntensity).toBe(entry.presentation!.environmentIntensity);
});
