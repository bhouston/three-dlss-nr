import { useEffect, useRef, useState, useSyncExternalStore, type ReactNode } from 'react';

import type { DemoController, DemoState, FrameMode } from '@/lib/demo';
import { HEAD_MODELS, headModel } from '@/lib/models';
import { UPSTREAM_URL } from '@/lib/links';

const RESOLUTION_OPTIONS = [
  { width: 640, height: 360 },
  { width: 960, height: 540 },
  { width: 1280, height: 720 },
];

const STYLE_OPTIONS = [
  { value: 0, label: 'None' },
  { value: 1, label: 'Cinematic' },
  { value: 2, label: 'Natural' },
];

const noopSubscribe = () => () => undefined;

/** The whole demo: viewport, controls, status, attribution. The controller (three.js) loads on the client only. */
export function DemoApp() {
  const viewportRef = useRef<HTMLDivElement>(null);
  const [controller, setController] = useState<DemoController | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    const container = viewportRef.current;
    if (!container) return;
    let disposed = false;
    let created: DemoController | null = null;
    import('@/lib/demo')
      .then(({ DemoController: Controller }) => {
        if (disposed) return;
        created = new Controller();
        setController(created);
        (globalThis as { __nrDemo?: DemoController }).__nrDemo = created;
        void created.start(container);
      })
      .catch((error: unknown) => setLoadError(error instanceof Error ? error.message : String(error)));
    return () => {
      disposed = true;
      created?.dispose();
      setController(null);
    };
  }, []);

  const state = useSyncExternalStore(
    controller?.subscribe ?? noopSubscribe,
    () => controller?.getState() ?? null,
    () => null,
  );

  const error = loadError ?? (state?.phase === 'error' ? state.error : null);
  const errorTitle = state?.errorTitle ?? 'This demo needs WebGPU';

  return (
    <div className="mx-auto grid w-full max-w-[1500px] gap-4 p-4 lg:grid-cols-[minmax(0,1fr)_340px]">
      <div className="flex min-w-0 flex-col gap-2">
        <div
          ref={viewportRef}
          className="relative aspect-video w-full overflow-hidden rounded-md border border-border bg-black"
        >
          {state && !error ? <ViewportOverlay state={state} controller={controller!} /> : null}
          {error ? <WebGPUError title={errorTitle} message={error} /> : null}
          {!state && !error ? (
            <p className="absolute inset-0 grid place-items-center text-sm text-white/70">Loading…</p>
          ) : null}
        </div>
        <HeadCredit modelId={state?.modelId ?? HEAD_MODELS[0].id} />
        {state ? <StatusBar state={state} /> : null}
      </div>
      <aside className="flex flex-col gap-4 text-sm">
        {state && controller ? <Controls state={state} controller={controller} /> : null}
      </aside>
    </div>
  );
}

function WebGPUError({ title, message }: { title: string; message: string }) {
  return (
    <div className="absolute inset-0 grid place-items-center p-6 text-center text-white">
      <div className="max-w-lg space-y-2">
        <p className="text-base font-semibold">{title}</p>
        <p className="text-sm text-white/80">{message}</p>
        <p className="text-xs text-white/60">
          The neural network runs as WebGPU compute shaders in your browser; there is no server-side fallback.
        </p>
      </div>
    </div>
  );
}

function ViewportOverlay({ state, controller }: { state: DemoState; controller: DemoController }) {
  const building = state.network.state === 'building';
  const shownView = state.network.state === 'ready' ? state.view : 'off';
  return (
    <>
      {shownView === 'split' ? (
        <SplitHandle split={state.split} onChange={(split) => controller.setSplit(split)} />
      ) : null}
      <div className="pointer-events-none absolute top-2 left-2 flex gap-2 text-xs">
        {shownView === 'split' ? <Badge>NR off</Badge> : <Badge>{shownView === 'on' ? 'NR on' : 'NR off'}</Badge>}
        {state.weights?.kind === 'synthetic' && shownView !== 'off' ? (
          <Badge tone="warning">synthetic weights: output is not meaningful</Badge>
        ) : null}
      </div>
      {shownView === 'split' ? (
        <div className="pointer-events-none absolute top-2 right-2 text-xs">
          <Badge>NR on</Badge>
        </div>
      ) : null}
      {building ? <BuildProgress state={state} /> : null}
      {state.modelLoading ? (
        <div className="pointer-events-none absolute right-2 bottom-2 text-xs">
          <Badge>loading scene…</Badge>
        </div>
      ) : null}
    </>
  );
}

function Badge({ children, tone }: { children: ReactNode; tone?: 'warning' }) {
  return (
    <span
      className={`rounded px-2 py-0.5 backdrop-blur ${
        tone === 'warning' ? 'bg-amber-500/80 text-black' : 'bg-black/60 text-white'
      }`}
    >
      {children}
    </span>
  );
}

function SplitHandle({ split, onChange }: { split: number; onChange: (split: number) => void }) {
  const dragging = useRef(false);
  const move = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!dragging.current) return;
    const rect = (event.currentTarget.parentElement as HTMLElement).getBoundingClientRect();
    onChange((event.clientX - rect.left) / rect.width);
  };
  return (
    // Pointer-only convenience: the View section's range input is the accessible control for the same value.
    <div
      aria-hidden="true"
      className="absolute top-0 bottom-0 z-10 w-6 -translate-x-1/2 cursor-ew-resize touch-none"
      style={{ left: `${split * 100}%` }}
      onPointerDown={(event) => {
        dragging.current = true;
        event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerMove={move}
      onPointerUp={() => {
        dragging.current = false;
      }}
    >
      <div className="absolute top-0 bottom-0 left-1/2 w-0.5 -translate-x-1/2 bg-white/80 shadow" />
      <div className="absolute top-1/2 left-1/2 grid h-8 w-8 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full bg-white/90 text-xs text-black shadow">
        ⇔
      </div>
    </div>
  );
}

function BuildProgress({ state }: { state: DemoState }) {
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(performance.now()), 500);
    return () => clearInterval(timer);
  }, []);
  const seconds = Math.max(0, (now - state.network.since) / 1000);
  const match = /(\d+)\s*\/\s*(\d+)/.exec(state.network.message);
  const fraction = match ? Number(match[1]) / Math.max(1, Number(match[2])) : null;
  return (
    <div className="absolute inset-x-0 bottom-0 bg-black/70 p-3 text-xs text-white backdrop-blur">
      <div className="flex justify-between gap-2">
        <span className="truncate">Preparing the network: {state.network.message}</span>
        <span className="shrink-0 tabular-nums">{seconds.toFixed(0)} s</span>
      </div>
      <div className="mt-2 h-1 overflow-hidden rounded bg-white/20">
        {fraction !== null ? (
          <div className="h-full bg-sky-400 transition-[width]" style={{ width: `${(fraction * 100).toFixed(1)}%` }} />
        ) : (
          <div className="h-full w-1/3 animate-pulse bg-sky-400" />
        )}
      </div>
      <p className="mt-1 text-white/60">
        The first build compiles a few hundred GPU pipelines; on Windows (D3D12) this can take a minute. Later builds at
        the same resolution reuse them.
      </p>
    </div>
  );
}

function HeadCredit({ modelId }: { modelId: string }) {
  const { attribution: a } = headModel(modelId);
  return (
    <p className="text-xs text-muted-foreground">
      Scene: <Link href={a.titleUrl}>&ldquo;{a.title}&rdquo;</Link> by <Link href={a.authorUrl}>{a.author}</Link>,
      licensed under <Link href={a.licenseUrl}>{a.license}</Link>
      {a.source ? (
        <>
          . <Link href={a.sourceUrl}>{a.source}</Link>
        </>
      ) : null}
      . {a.modifications}
    </p>
  );
}

function Link({ href, children }: { href?: string; children: ReactNode }) {
  if (!href) return <>{children}</>;
  return (
    <a href={href} className="text-primary underline underline-offset-2" target="_blank" rel="noreferrer">
      {children}
    </a>
  );
}

function StatusBar({ state }: { state: DemoState }) {
  const { network } = state;
  const tone =
    network.state === 'failed'
      ? 'text-red-500'
      : network.state === 'ready'
        ? 'text-emerald-500'
        : 'text-muted-foreground';
  return (
    <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 rounded-md border border-border bg-muted px-3 py-2 text-xs">
      <span className={tone}>
        Network: {network.state === 'none' ? 'none' : network.state} {network.message ? `- ${network.message}` : ''}
      </span>
      <span className="text-muted-foreground tabular-nums">{state.fps.toFixed(state.fps < 10 ? 1 : 0)} fps</span>
      <span className="text-muted-foreground">
        internal {state.resolution.width}x{state.resolution.height}
      </span>
    </div>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="rounded-md border border-border p-3">
      <h2 className="mb-2 text-xs font-semibold tracking-wide text-muted-foreground uppercase">{title}</h2>
      <div className="flex flex-col gap-2">{children}</div>
    </section>
  );
}

function Segmented<T extends string>({
  value,
  options,
  onChange,
  disabled,
}: {
  value: T;
  options: { value: T; label: string; disabled?: boolean; title?: string }[];
  onChange: (value: T) => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex overflow-hidden rounded border border-border">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          title={option.title}
          disabled={disabled || option.disabled}
          onClick={() => onChange(option.value)}
          className={`flex-1 px-2 py-1 text-xs disabled:cursor-not-allowed disabled:opacity-40 ${
            option.value === value ? 'bg-primary text-background' : 'hover:bg-muted'
          }`}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  format?: (value: number) => string;
}) {
  return (
    <label className="grid grid-cols-[110px_1fr_44px] items-center gap-2 text-xs">
      <span>{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <span className="text-right tabular-nums text-muted-foreground">{format ? format(value) : value.toFixed(2)}</span>
    </label>
  );
}

function Controls({ state, controller }: { state: DemoState; controller: DemoController }) {
  const directoryInput = useRef<HTMLInputElement>(null);
  useEffect(() => {
    // `webkitdirectory` is not in React's input attribute types.
    directoryInput.current?.setAttribute('webkitdirectory', '');
    directoryInput.current?.setAttribute('directory', '');
  }, []);
  const hasWeights = state.weights !== null;
  const nrReady = state.network.state === 'ready';
  const s = state.settings;

  return (
    <>
      <Section title="View">
        <Segmented
          value={state.view}
          onChange={(view) => controller.setView(view)}
          options={[
            { value: 'off', label: 'NR off' },
            { value: 'on', label: 'NR on', disabled: !hasWeights },
            { value: 'split', label: 'Split', disabled: !hasWeights },
          ]}
        />
        {state.view === 'split' ? (
          <Slider
            label="Split"
            value={state.split}
            min={0}
            max={1}
            step={0.01}
            onChange={(v) => controller.setSplit(v)}
          />
        ) : null}
        {!hasWeights ? (
          <p className="text-xs text-muted-foreground">
            NR is off because no weights are loaded. This site hosts no weights: load a model directory below, or
            generate synthetic weights to see the pipeline run.
          </p>
        ) : null}
      </Section>

      <Section title="Weights">
        <p className="text-xs">
          {state.weights ? (
            <>
              Loaded: <span className="font-medium">{state.weights.label}</span>
            </>
          ) : (
            <span className="text-muted-foreground">None loaded (NR off).</span>
          )}
        </p>
        <input
          ref={directoryInput}
          type="file"
          multiple
          className="hidden"
          onChange={(event) => {
            if (event.target.files?.length) void controller.loadDirectory(event.target.files);
            event.target.value = '';
          }}
        />
        <button
          type="button"
          className="rounded border border-border px-2 py-1 text-left text-xs hover:bg-muted disabled:opacity-40"
          disabled={state.weightsBusy !== null}
          onClick={() => directoryInput.current?.click()}
        >
          Load model directory…
          <span className="block text-muted-foreground">
            The folder with manifest.json and model/stages/*. Read by your browser only; never uploaded.
          </span>
        </button>
        <button
          type="button"
          className="rounded border border-border px-2 py-1 text-left text-xs hover:bg-muted disabled:opacity-40"
          disabled={state.weightsBusy !== null}
          onClick={() => void controller.loadSyntheticWeights()}
        >
          Synthetic weights
          <span className="block text-muted-foreground">Testing only: output is not meaningful.</span>
        </button>
        {hasWeights ? (
          <button
            type="button"
            className="rounded border border-border px-2 py-1 text-left text-xs hover:bg-muted"
            onClick={() => void controller.clearWeights()}
          >
            Unload weights (NR off)
          </button>
        ) : null}
        {state.weightsBusy ? <p className="text-xs text-muted-foreground">{state.weightsBusy}</p> : null}
        {state.weightsError ? <p className="text-xs text-red-500">{state.weightsError}</p> : null}
        <details className="text-xs text-muted-foreground">
          <summary className="cursor-pointer">Where do weights come from?</summary>
          <p className="mt-1">
            The network is a reimplementation of NVIDIA&rsquo;s DLSS 5 neural rendering network. Its weights are not
            open and are not distributed here or by <Link href={UPSTREAM_URL}>OpenDLSS-NR</Link>. If you have a model
            directory in OpenDLSS-NR&rsquo;s layout (manifest.json plus eleven stage files, about 141 MiB), load it
            above.
          </p>
        </details>
      </Section>

      <Section title="Backend">
        <div className="flex flex-col gap-1">
          {state.backends.map((option) => (
            <label
              key={option.id}
              className={`flex items-start gap-2 text-xs ${option.reason ? 'opacity-50' : ''}`}
              title={option.detail ?? undefined}
            >
              <input
                type="radio"
                name="backend"
                className="mt-0.5"
                checked={state.backendId === option.id}
                disabled={option.reason !== null}
                onChange={() => void controller.setBackend(option.id)}
              />
              <span>
                {option.label}
                {option.reason ? (
                  <span className="block text-muted-foreground">Unavailable: {option.reason}</span>
                ) : null}
                <TimingLine timing={state.timings[option.id]} />
              </span>
            </label>
          ))}
        </div>
        {state.backendId === 'reference-wgsl' ? (
          <div className="flex flex-col gap-1 text-xs">
            <span className="text-muted-foreground">Frame around the reference network</span>
            <Segmented<FrameMode>
              value={state.frameMode}
              onChange={(mode) => void controller.setFrameMode(mode)}
              options={[
                { value: 'shared', label: 'TSL frame kernels', title: 'three-dlss-nr input features + compose' },
                { value: 'reference', label: 'Upstream frame.wgsl', title: 'the reference demo frame, recorded as is' },
              ]}
            />
          </div>
        ) : null}
        <p className="text-xs text-muted-foreground">
          GPU time per frame (timestamp queries), median of recent frames. Upstream frame mode also includes input
          preparation and composition. Other GPU work on this machine inflates it.
        </p>
        <details className="text-xs text-muted-foreground">
          <summary className="cursor-pointer rounded focus-visible:outline-2 focus-visible:outline-offset-2">
            About backends and frame modes
          </summary>
          <div className="mt-2 space-y-2">
            <p>
              Both backends run the same network with your loaded model weights. TSL is the native Three.js shader
              implementation; Reference WGSL runs the pinned upstream OpenDLSS-NR WebGPU implementation.
            </p>
            <p>
              TSL uses float and integer arithmetic without requiring shader-f16. Reference WGSL requires shader-f16.
              TSL needs at least 24 KiB of workgroup storage and 256 invocations per workgroup; Reference WGSL needs 32
              KiB and 512. Both need eight storage-buffer bindings per shader stage and enough buffer capacity for the
              model and resolution. Device support does not imply equal speed.
            </p>
            <p>
              With Reference WGSL selected, the frame-mode switch chooses input preparation and output composition:
              shared TSL frame kernels or upstream frame.wgsl. It keeps the Reference WGSL network selected.
            </p>
            <p>
              Compare backends with the same model, scene, camera, resolution, NR settings and shared frame mode. In
              shared frame mode, GPU time measures the network; upstream frame mode also times input preparation and
              composition. Demo FPS includes scene rendering and presentation. GPU time needs timestamp-query support;
              wall time includes waiting for submitted GPU work. Let recent samples refresh after changes, warm up first
              and compare on an idle GPU.
            </p>
            <p>
              Parity checks cover deterministic synthetic inputs and weights, including tested kernels and whole-network
              cases. They do not establish identical results or performance for every device or model.
            </p>
          </div>
        </details>
      </Section>

      <Section title="NR settings">
        <Slider
          label="Intensity"
          value={s.intensity}
          min={0}
          max={1}
          step={0.01}
          onChange={(v) => controller.setSettings({ intensity: v })}
        />
        <div className="grid grid-cols-[110px_1fr] items-center gap-2 text-xs">
          <span>Style</span>
          <Segmented
            value={String(s.style)}
            onChange={(value) => controller.setSettings({ style: Number(value) })}
            options={STYLE_OPTIONS.map((option) => ({ value: String(option.value), label: option.label }))}
          />
        </div>
        <Slider
          label="Local tone"
          value={s.localTone}
          min={0}
          max={2}
          step={0.01}
          onChange={(v) => controller.setSettings({ localTone: v })}
        />
        <Slider
          label="Local structure"
          value={s.localStructure}
          min={0}
          max={2}
          step={0.01}
          onChange={(v) => controller.setSettings({ localStructure: v })}
        />
        <Slider
          label="Skin structure"
          value={s.skinStructure}
          min={-1}
          max={2}
          step={0.01}
          format={(v) => (v < 0 ? 'auto' : v.toFixed(2))}
          onChange={(v) => controller.setSettings({ skinStructure: v < 0 ? -1 : v })}
        />
        <label className="flex items-center gap-2 text-xs">
          <input
            type="checkbox"
            checked={s.autoMask}
            onChange={(e) => controller.setSettings({ autoMask: e.target.checked })}
          />
          Auto mask
        </label>
        <Slider
          label="Colour strength"
          value={s.colorStrength}
          min={0}
          max={1}
          step={0.01}
          onChange={(v) => controller.setSettings({ colorStrength: v })}
        />
        <Slider
          label="Paper white"
          value={s.paperWhite}
          min={0.25}
          max={4}
          step={0.05}
          onChange={(v) => controller.setSettings({ paperWhite: v })}
        />
        <button
          type="button"
          className="self-start rounded border border-border px-2 py-1 text-xs hover:bg-muted disabled:opacity-40"
          disabled={!nrReady}
          onClick={() => controller.resetHistory()}
        >
          Reset history
        </button>
      </Section>

      <Section title="Scene">
        {state.modelError ? (
          <div role="alert" className="space-y-1 text-xs text-red-500">
            <p>{state.modelError.message}</p>
            <button
              type="button"
              className="underline"
              disabled={state.modelLoading}
              onClick={() => void controller.setModel(state.modelError!.modelId)}
            >
              Retry scene
            </button>
          </div>
        ) : null}
        <label className="grid grid-cols-[110px_1fr] items-center gap-2 text-xs">
          <span>Subject</span>
          <select
            className="rounded border border-border bg-background px-1 py-0.5"
            value={state.requestedModelId ?? state.modelId}
            onChange={(e) => void controller.setModel(e.target.value)}
          >
            {HEAD_MODELS.map((model) => (
              <option key={model.id} value={model.id}>
                {model.label}
              </option>
            ))}
          </select>
        </label>
        <label className="grid grid-cols-[110px_1fr] items-center gap-2 text-xs">
          <span>Resolution</span>
          <select
            className="rounded border border-border bg-background px-1 py-0.5"
            value={`${state.resolution.width}x${state.resolution.height}`}
            onChange={(e) => {
              const [width, height] = e.target.value.split('x').map(Number);
              void controller.setResolution({ width, height });
            }}
          >
            {RESOLUTION_OPTIONS.map((r) => (
              <option key={`${r.width}x${r.height}`} value={`${r.width}x${r.height}`}>
                {r.width} x {r.height}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2 text-xs">
          <input
            type="checkbox"
            checked={state.autoRotate}
            onChange={(e) => controller.setAutoRotate(e.target.checked)}
          />
          Slow auto-rotate
        </label>
        <button
          type="button"
          className="self-start rounded border border-border px-2 py-1 text-xs hover:bg-muted"
          onClick={() => controller.resetCamera()}
        >
          Reset camera
        </button>
      </Section>

      <DeviceSection state={state} />
    </>
  );
}

function TimingLine({ timing }: { timing: DemoState['timings'][keyof DemoState['timings']] }) {
  if (!timing) return null;
  return (
    <span className="block text-muted-foreground tabular-nums">
      {timing.gpuMs !== null ? `${timing.gpuMs.toFixed(1)} ms GPU` : 'GPU time n/a'} · {timing.wallMs.toFixed(1)} ms
      wall · {timing.frames} frames
    </span>
  );
}

function DeviceSection({ state }: { state: DemoState }) {
  const device = state.device;
  if (!device) return null;
  return (
    <Section title="Device">
      <p className="text-xs break-words">{device.adapter}</p>
      <ul className="text-xs">
        <li>
          shader-f16: <Flag on={device.shaderF16} />{' '}
          <span className="text-muted-foreground">(needed by the reference backend only)</span>
        </li>
        <li>
          timestamp-query: <Flag on={device.timestampQuery} />
        </li>
        {device.limits.map((limit) => (
          <li key={limit.name} className="flex justify-between gap-2">
            <span className="truncate text-muted-foreground">{limit.name}</span>
            <span className="tabular-nums">{mib(limit.value)}</span>
          </li>
        ))}
      </ul>
      {device.problem ? <p className="text-xs text-red-500">{device.problem}</p> : null}
    </Section>
  );
}

const mib = (value: number) => (value >= 1048576 ? `${(value / 1048576).toFixed(0)} MiB` : String(value));

function Flag({ on }: { on: boolean }) {
  return <span className={on ? 'text-emerald-500' : 'text-red-500'}>{on ? 'yes' : 'no'}</span>;
}
