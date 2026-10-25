/**
 * Sunscreen protection: linear in applied amount (model/sunrisk/sunscreen.py).
 * SPF_eff = 1 + (SPF_label - 1) * amount / A_lab. The tracker keeps its own per-particle state;
 * this helper is for display (e.g. "your SPF 50 is acting like SPF 20").
 */

import * as config from "./config";

export function effectiveSpf(spfLabel: number, amountMgCm2 = config.TYPICAL_APPLIED_AMOUNT): number {
  return 1.0 + ((spfLabel - 1.0) * amountMgCm2) / config.LAB_SUNSCREEN_AMOUNT;
}
