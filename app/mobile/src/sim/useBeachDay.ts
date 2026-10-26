/**
 * Plays the simulated beach day through the risk engine on a sped-up clock.
 * Swap this hook for one fed by the BLE stream once the device exists; the screen only needs
 * a Snapshot.
 */

import {
  ExposureTracker, expectedUviCurve, seededRng, type BodyPart, type RiskOutput, type RiskStep,
} from "@sunrisk/risk-engine";
import { useCallback, useEffect, useState } from "react";

import {
  FITZPATRICK, LAT, LON, MORNING_SUNSCREEN, nextChange, segmentAt, stepAt, T_END, T_START,
  type Segment,
} from "./beachDay";

const TICK_MS = 500;
export const SPEEDS = [60, 300, 900] as const; // sim seconds per real second

export interface Snapshot {
  tUnixS: number;
  step: RiskStep;
  segment: Segment;
  output: RiskOutput;
  finished: boolean;
}

interface Sim {
  tracker: ExposureTracker;
  t: number;
}

function newSim(): Sim {
  const tracker = new ExposureTracker({ fitzpatrick: FITZPATRICK }, { rng: seededRng(1) });
  tracker.applySunscreen(MORNING_SUNSCREEN);
  tracker.step(stepAt(T_START));
  return { tracker, t: T_START };
}

/** Step the tracker forward, stopping at every schedule change so conditions switch on time. */
function advance(sim: Sim, dtS: number): void {
  const target = Math.min(sim.t + dtS, T_END);
  while (sim.t < target) {
    sim.t = Math.min(target, nextChange(sim.t));
    sim.tracker.step(stepAt(sim.t));
  }
}

function snapshot(sim: Sim): Snapshot {
  const step = stepAt(sim.t);
  // forecast-aware projection: clear-sky shape over the rest of the day, starting from what
  // the sensor sees now (no weather forecast feed yet)
  const curve = expectedUviCurve(sim.t, LAT, LON, { uviNow: step.uvi_ambient, environment: step.environment });
  return {
    tUnixS: sim.t,
    step,
    segment: segmentAt(sim.t),
    output: sim.tracker.output(curve),
    finished: sim.t >= T_END,
  };
}

export function useBeachDay() {
  // the tracker mutates in place; a new Sim object only on reset
  const [sim, setSim] = useState(newSim);
  const [snap, setSnap] = useState<Snapshot>(() => snapshot(sim));
  const [running, setRunning] = useState(true);
  const [speed, setSpeed] = useState<number>(SPEEDS[0]);

  useEffect(() => {
    if (!running || snap.finished) return;
    const id = setInterval(() => {
      advance(sim, (TICK_MS / 1000) * speed);
      setSnap(snapshot(sim));
    }, TICK_MS);
    return () => clearInterval(id);
  }, [sim, running, speed, snap.finished]);

  const reset = useCallback(() => {
    const s = newSim();
    setSim(s);
    setSnap(snapshot(s));
    setRunning(true);
  }, []);

  const applySunscreen = useCallback((spf: number, bodyParts?: BodyPart[]) => {
    if (bodyParts?.length === 0) return;
    sim.tracker.applySunscreen({ t_unix_s: sim.t, spf_label: spf, body_parts: bodyParts });
    setSnap(snapshot(sim));
  }, [sim]);

  return { snap, running, setRunning, speed, setSpeed, reset, applySunscreen };
}
