import { UTC_OFFSET_H } from "./sim/beachDay";

export const PART_LABELS: Record<string, string> = {
  forearm: "Forearm",
  face: "Face",
  neck: "Neck",
  shoulders: "Shoulders",
  upper_back: "Upper back",
  chest: "Chest",
  thighs: "Thighs",
  shins: "Shins",
};

/** Minutes left for display. The engine's horizon is 12 h; beyond it is Infinity. */
export function formatMinutes(min: number): string {
  if (!Number.isFinite(min)) return "12 h+";
  const m = Math.floor(min);
  if (m < 60) return `${m} min`;
  return `${Math.floor(m / 60)} h ${String(m % 60).padStart(2, "0")} min`;
}

export function formatRange(p10: number, p90: number): string {
  if (!Number.isFinite(p10)) return "beyond 12 h";
  if (!Number.isFinite(p90)) return `${formatMinutes(p10)} or more`;
  return `${formatMinutes(p10)} – ${formatMinutes(p90)}`;
}

/** Clock time at the simulated location, independent of the phone's time zone. */
export function formatClock(tUnixS: number): string {
  const d = new Date((tUnixS + UTC_OFFSET_H * 3600) * 1000);
  const h = d.getUTCHours();
  const m = String(d.getUTCMinutes()).padStart(2, "0");
  return `${h % 12 || 12}:${m} ${h < 12 ? "AM" : "PM"}`;
}

// WHO UV Index categories
export function uvCategory(uvi: number): { label: string; color: string } {
  if (uvi < 3) return { label: "Low", color: "#3a8a3a" };
  if (uvi < 6) return { label: "Moderate", color: "#b08a00" };
  if (uvi < 8) return { label: "High", color: "#d0661c" };
  if (uvi < 11) return { label: "Very high", color: "#c42a2a" };
  return { label: "Extreme", color: "#7b3fa0" };
}
