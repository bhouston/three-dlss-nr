export function Arrow({ diagonal = false }: { diagonal?: boolean }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className="arrow-icon">
      <path
        d={diagonal ? 'M6 18 18 6M6 6h12v12' : 'M4 12h16M13 5l7 7-7 7'}
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function Cube({ network = false }: { network?: boolean }) {
  if (network)
    return (
      <svg viewBox="0 0 48 48" fill="none" aria-hidden="true" className="cube-icon">
        <path
          d="m24 4 18 10v20L24 44 6 34V14L24 4Zm0 20L6 14m18 10 18-10M24 24v20M6 34l36-20M6 14l36 20M24 4v20"
          stroke="currentColor"
          strokeWidth="1.3"
          strokeLinejoin="round"
        />
        {[
          [24, 4],
          [42, 14],
          [42, 34],
          [24, 44],
          [6, 34],
          [6, 14],
          [24, 24],
        ].map(([cx, cy]) => (
          <circle
            key={`${cx}-${cy}`}
            cx={cx}
            cy={cy}
            r="3"
            fill="var(--paper)"
            stroke="currentColor"
            strokeWidth="1.5"
          />
        ))}
      </svg>
    );
  return (
    <svg viewBox="0 0 48 48" fill="none" aria-hidden="true" className="cube-icon">
      <path
        d="m24 4 18 10v21L24 45 6 35V14L24 4Zm0 20L6 14m18 10 18-10M24 24v21"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
    </svg>
  );
}
