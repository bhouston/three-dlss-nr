import { useEffect, useRef, useState } from 'react';

export type StudioMode = 'shape' | 'material' | 'light';

/** Real-time Three.js illustration; it never claims to run neural rendering. */
export function OrbitScene({
  motion,
  mode = 'material',
  hero = false,
}: {
  motion: boolean;
  mode?: StudioMode;
  hero?: boolean;
}) {
  const host = useRef<HTMLElement>(null);
  const config = useRef({ motion, mode });
  const refresh = useRef<() => void>(() => undefined);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    config.current = { motion, mode };
    refresh.current();
  }, [motion, mode]);
  useEffect(() => {
    const container = host.current;
    if (!container) return;
    let disposed = false;
    let cleanup: () => void = () => undefined;
    Promise.all([import('three'), import('three/addons/environments/RoomEnvironment.js')])
      .then(([T, { RoomEnvironment }]) => {
        if (disposed) return;
        const renderer = new T.WebGLRenderer({ alpha: true, antialias: true });
        renderer.setPixelRatio(Math.min(devicePixelRatio, 1.7));
        renderer.outputColorSpace = T.SRGBColorSpace;
        renderer.toneMapping = T.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.25;
        renderer.setClearColor(0x000000, 0);
        renderer.domElement.setAttribute('aria-hidden', 'true');
        container.append(renderer.domElement);
        const scene = new T.Scene();
        const camera = new T.PerspectiveCamera(35, 1, 0.1, 40);
        camera.position.set(0, 0.2, hero ? 8.5 : 7.6);
        const pmrem = new T.PMREMGenerator(renderer);
        const room = new RoomEnvironment();
        const environment = pmrem.fromScene(room, 0.04);
        scene.environment = environment.texture;
        room.dispose();
        pmrem.dispose();
        const group = new T.Group();
        scene.add(group);
        const geometry = new T.TorusKnotGeometry(1.14, 0.31, 180, 28, 2, 3);
        const cream = new T.MeshPhysicalMaterial({ color: 0xeae8d8, roughness: 0.23, metalness: 0.12, clearcoat: 0.6 });
        const lime = new T.MeshPhysicalMaterial({ color: 0xb0ee37, roughness: 0.14, metalness: 0.9, clearcoat: 1 });
        const first = new T.Mesh(geometry, cream);
        first.rotation.set(-0.3, 0.2, -0.32);
        group.add(first);
        const second = new T.Mesh(geometry, lime);
        second.scale.setScalar(0.79);
        second.rotation.set(0.8, -0.5, 0.7);
        group.add(second);
        const wireMaterial = new T.MeshBasicMaterial({
          color: 0x92998b,
          wireframe: true,
          transparent: true,
          opacity: 0.13,
        });
        const echoGeometry = new T.TorusKnotGeometry(1.15, 0.32, 64, 10, 2, 3);
        const echo = new T.Mesh(echoGeometry, wireMaterial);
        echo.position.set(-0.55, 0.1, -0.8);
        echo.scale.setScalar(1.35);
        scene.add(echo);
        const ambient = new T.HemisphereLight(0xffffff, 0x808767, 1.5);
        const key = new T.DirectionalLight(0xffffff, 3.8);
        key.position.set(3, 5, 4);
        scene.add(ambient, key);
        const accent = new T.PointLight(0xc0fa67, 45, 15);
        accent.position.set(-3, 1, 3);
        scene.add(accent);
        const lines: any[] = [];
        for (let index = 0; index < 2; index++) {
          const points = Array.from({ length: 129 }, (_, step) => {
            const angle = (step / 128) * Math.PI * 2;
            return new T.Vector3(Math.cos(angle) * 2.5, Math.sin(angle) * 2.5, 0);
          });
          const lineGeometry = new T.BufferGeometry().setFromPoints(points);
          const lineMaterial = new T.LineBasicMaterial({ color: 0x666b63, transparent: true, opacity: 0.27 });
          const line = new T.Line(lineGeometry, lineMaterial);
          line.rotation.set(index ? 1.1 : 0.7, index ? -0.5 : 0.5, 0.2);
          scene.add(line);
          lines.push(line);
        }
        let visible = true;
        let time = 0;
        let last = 0;
        let dragging = false;
        let dragX = 0;
        let dragY = 0;
        let rotation = 0;
        let tilt = 0;
        const render = (now = 0) => {
          if (disposed) return;
          const settings = config.current;
          if (settings.motion && last) time += Math.min((now - last) / 1000, 0.05);
          last = now;
          cream.wireframe = settings.mode === 'shape';
          lime.wireframe = settings.mode === 'shape';
          cream.color.set(settings.mode === 'light' ? 0xd9e2f1 : 0xeae8d8);
          accent.color.set(settings.mode === 'light' ? 0x7da8ff : 0xc0fa67);
          accent.position.x = settings.mode === 'light' ? Math.cos(time * 0.8) * 3 : -3;
          group.rotation.set(tilt + (settings.motion ? Math.sin(time * 0.4) * 0.09 : 0), rotation + time * 0.13, -0.08);
          echo.rotation.y = time * -0.08;
          renderer.render(scene, camera);
        };
        const sync = () => {
          last = 0;
          renderer.setAnimationLoop(visible && config.current.motion && !document.hidden ? render : null);
          render();
        };
        refresh.current = sync;
        const resize = new ResizeObserver(() => {
          const { width, height } = container.getBoundingClientRect();
          renderer.setSize(width, height, false);
          camera.aspect = width / Math.max(height, 1);
          camera.updateProjectionMatrix();
          render();
        });
        resize.observe(container);
        const observer = new IntersectionObserver(([entry]) => {
          visible = entry.isIntersecting;
          sync();
        });
        observer.observe(container);
        const down = (event: PointerEvent) => {
          dragging = true;
          dragX = event.clientX;
          dragY = event.clientY;
          container.setPointerCapture(event.pointerId);
        };
        const move = (event: PointerEvent) => {
          if (!dragging) return;
          rotation += (event.clientX - dragX) * 0.007;
          tilt = Math.max(-0.6, Math.min(0.6, tilt + (event.clientY - dragY) * 0.005));
          dragX = event.clientX;
          dragY = event.clientY;
          render();
        };
        const up = () => {
          dragging = false;
        };
        if (!hero) {
          container.addEventListener('pointerdown', down);
          container.addEventListener('pointermove', move);
          container.addEventListener('pointerup', up);
          container.addEventListener('pointercancel', up);
        }
        document.addEventListener('visibilitychange', sync);
        sync();
        cleanup = () => {
          renderer.setAnimationLoop(null);
          refresh.current = () => undefined;
          resize.disconnect();
          observer.disconnect();
          document.removeEventListener('visibilitychange', sync);
          container.removeEventListener('pointerdown', down);
          container.removeEventListener('pointermove', move);
          container.removeEventListener('pointerup', up);
          container.removeEventListener('pointercancel', up);
          geometry.dispose();
          echoGeometry.dispose();
          cream.dispose();
          lime.dispose();
          wireMaterial.dispose();
          for (const line of lines) {
            line.geometry.dispose();
            line.material.dispose();
          }
          environment.dispose();
          renderer.dispose();
          renderer.domElement.remove();
        };
      })
      .catch(() => {
        if (!disposed) setFailed(true);
      });
    return () => {
      disposed = true;
      cleanup();
    };
  }, [hero]);
  return (
    <figure
      className={`orbit-scene ${hero ? 'orbit-hero' : 'orbit-studio'}`}
      ref={host}
      aria-label="Összefonódó térbeli formák, Three.js-ben rajzolva"
    >
      {failed ? (
        <img
          src="/explainer/helmet-split.jpg"
          alt="A demó pilótasisakja; a 3D illusztráció ezen a készüléken nem indult el."
        />
      ) : null}
    </figure>
  );
}
