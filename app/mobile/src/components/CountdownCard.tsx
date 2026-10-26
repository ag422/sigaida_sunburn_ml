import type { BodyPartRisk, RiskOutput } from "@sunrisk/risk-engine";
import { StyleSheet, Text, View } from "react-native";

import { formatMinutes, formatRange, PART_LABELS } from "../format";
import { colors } from "../theme";

/**
 * The headline. Per docs/contracts.md the big number is p10 (the cautious end) and the
 * p10-p90 range is always shown, never a single precise-looking number.
 */
export function CountdownCard({ output }: { output: RiskOutput }) {
  const first = output.parts.find((p) => p.body_part === output.first_to_burn) as BodyPartRisk;
  const { p10, p90 } = first.minutes_left;
  const name = PART_LABELS[first.body_part] ?? first.body_part;
  const burned = p10 === 0;
  const tone = burned || output.alert ? colors.danger : colors.text;

  return (
    <View style={[styles.card, (burned || output.alert) && styles.cardAlert]}>
      {output.alert && (
        <Text style={styles.alert}>
          {burned ? "Burn threshold may be reached" : "Cover up or find shade soon"}
        </Text>
      )}
      {first.beyond_horizon ? (
        <>
          <Text style={styles.label}>Time until sunburn</Text>
          <Text style={styles.big}>12 h+</Text>
          <Text style={styles.sub}>No burn expected at today&apos;s UV</Text>
        </>
      ) : (
        <>
          <Text style={styles.label}>{burned ? `${name} may already be burning` : `${name} could burn in`}</Text>
          <Text style={[styles.big, { color: tone }]}>{burned ? "Now" : formatMinutes(p10)}</Text>
          <Text style={styles.sub}>
            {!burned ? `Likely ${formatRange(p10, p90)}`
              : p90 === 0 ? "Past the burn dose in every estimate"
              : `Likely range ${formatRange(p10, p90)}`}
          </Text>
        </>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.card,
    borderRadius: 20,
    padding: 24,
    alignItems: "center",
    borderWidth: 2,
    borderColor: "transparent",
  },
  cardAlert: { borderColor: colors.danger, backgroundColor: colors.dangerBg },
  alert: {
    color: colors.danger,
    fontWeight: "700",
    fontSize: 15,
    marginBottom: 12,
  },
  label: { fontSize: 16, color: colors.muted },
  big: { fontSize: 64, fontWeight: "800", color: colors.text, marginVertical: 4, fontVariant: ["tabular-nums"] },
  sub: { fontSize: 15, color: colors.muted },
});
