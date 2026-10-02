// Node side of the hardware-f16 FP8 GEMM check (see run-fp8-gemm-chrome.mjs): the CPU oracle for the same cases.
//
// Part of three-dlss-nr (a port to three.js of OpenDLSS-NR by maan, MIT).

import { oracleGemmFp8 } from '../oracle/gemm.js';
import { fp8ReferenceCases } from './fp8Cases.js';

export interface OracleCaseResult {
  label: string;
  e4?: number[];
  half?: number[];
}

export function fp8OracleResults(): OracleCaseResult[] {
  return fp8ReferenceCases().map((c) => {
    const r = oracleGemmFp8({
      ...c,
      residual: c.residualFormat ? { format: c.residualFormat, data: c.residualData! } : undefined,
      scale: c.scale,
    });
    return {
      label: c.label,
      e4: r.e4 ? Array.from(r.e4) : undefined,
      half: r.half ? Array.from(r.half) : undefined,
    };
  });
}
