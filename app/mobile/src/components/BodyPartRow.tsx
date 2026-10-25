import type { BodyPartRisk } from "@sunrisk/risk-engine";
import { StyleSheet, Text, View } from "react-native";

import { formatMinutes, formatRange, PART_LABELS } from "../format";
import { colors } from "../theme";

/** One body part: cautious time left, the likely range, and how much of today's dose is used. */
export function BodyPartRow({ part, alertLeadMin }: { part: BodyPartRisk; alertLeadMin: number }) {
  const { p10, p90 } = part.minutes_left;
  const used = part.fraction_med_used.p50;
  const urgent = p10 < alertLeadMin;
  const barColor = used >= 1 ? colors.danger : used >= 0.7 ? colors.warn : colors.ok;

  return (
    <View style={styles.row}>
      <View style={styles.top}>
        <Text style={styles.name}>{PART_LABELS[part.body_part] ?? part.body_part}</Text>
        {part.effective_spf > 1.5 && (
          <Text style={styles.badge}>acting like SPF {Math.round(part.effective_spf)}</Text>
        )}
        <Text style={[styles.time, urgent && { color: colors.danger }]}>
          {p10 === 0 ? "Now" : formatMinutes(p10)}
        </Text>
      </View>
      <View style={styles.track}>
        <View style={[styles.fill, { width: `${Math.min(used, 1) * 100}%`, backgroundColor: barColor }]} />
      </View>
      <Text style={styles.sub}>
        {Math.round(used * 100)}% of burn dose used · {p90 === 0 ? "past it in every estimate" : `likely ${formatRange(p10, p90)}`}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  row: { paddingVertical: 12, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: colors.border },
  top: { flexDirection: "row", alignItems: "center", gap: 8 },
  name: { fontSize: 16, fontWeight: "600", color: colors.text },
  badge: {
    fontSize: 12, color: colors.accent, backgroundColor: colors.accentBg,
    paddingHorizontal: 6, paddingVertical: 2, borderRadius: 6, overflow: "hidden",
  },
  time: { marginLeft: "auto", fontSize: 16, fontWeight: "700", color: colors.text, fontVariant: ["tabular-nums"] },
  track: { height: 6, borderRadius: 3, backgroundColor: colors.border, marginTop: 8, overflow: "hidden" },
  fill: { height: 6, borderRadius: 3 },
  sub: { fontSize: 13, color: colors.muted, marginTop: 6 },
});
