import { describe, expect, it, vi } from 'vitest';
import * as THREE from 'three/webgpu';

import { DemoController } from './demo';
import { DEFAULT_MODEL_ID } from './models';

const loads = vi.hoisted(() => ({ head: vi.fn() }));
vi.mock('./studio', () => ({ loadHead: loads.head, createStudio: vi.fn() }));

const head = () => ({ object: new THREE.Group(), dispose: vi.fn() });
const deferred = () => {
  let resolve!: (value: ReturnType<typeof head>) => void;
  const promise = new Promise<ReturnType<typeof head>>((done) => (resolve = done));
  return { promise, resolve };
};

const controller = () => {
  const demo: any = new DemoController();
  demo.studio = { stage: new THREE.Group(), dispose: vi.fn() };
  demo.head = head();
  demo.studio.stage.add(demo.head.object);
  demo.pass = { resetHistory: vi.fn(), dispose: vi.fn() };
  return demo;
};

describe('model load races', () => {
  it('disposes a superseded same-ID load without replacing the current model', async () => {
    const demo = controller();
    const displayed = demo.head;
    const a = deferred();
    const b = deferred();
    loads.head.mockReturnValueOnce(a.promise).mockReturnValueOnce(b.promise);
    const first = demo.setModel(DEFAULT_MODEL_ID);
    const second = demo.setModel(DEFAULT_MODEL_ID);
    const newest = head();
    b.resolve(newest);
    await second;
    expect(displayed.dispose).toHaveBeenCalledTimes(1);
    const stale = head();
    a.resolve(stale);
    await first;
    expect(stale.dispose).toHaveBeenCalledTimes(1);
    expect(newest.dispose).not.toHaveBeenCalled();
    expect(demo.head).toBe(newest);
    expect(demo.studio.stage.children).toEqual([newest.object]);
    expect(demo.pass.resetHistory).toHaveBeenCalledTimes(1);
  });

  it('keeps the displayed model alive on load failure', async () => {
    const demo = controller();
    const displayed = demo.head;
    loads.head.mockRejectedValueOnce(new Error('load failed'));
    await expect(demo.setModel(DEFAULT_MODEL_ID)).rejects.toThrow('load failed');
    expect(demo.head).toBe(displayed);
    expect(displayed.dispose).not.toHaveBeenCalled();
    expect(demo.getState().modelLoading).toBe(false);
  });

  it('releases a late load after controller disposal and does not emit state updates', async () => {
    const demo = controller();
    const pending = deferred();
    loads.head.mockReturnValueOnce(pending.promise);
    const request = demo.setModel(DEFAULT_MODEL_ID);
    const listener = vi.fn();
    demo.subscribe(listener);
    demo.dispose();
    const late = head();
    pending.resolve(late);
    await request;
    expect(late.dispose).toHaveBeenCalledTimes(1);
    expect(listener).not.toHaveBeenCalled();
  });
});
