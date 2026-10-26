import { config } from "@sunrisk/risk-engine";
import { Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { BodyPartRow } from "./components/BodyPartRow";
import { CountdownCard } from "./components/CountdownCard";
import { NotificationTest } from "./components/NotificationTest";
import { SunscreenControls } from "./components/SunscreenControls";
import { formatClock, uvCategory } from "./format";
import { useSunscreenReminders } from "./hooks/useSunscreenReminders";
import { FITZPATRICK } from "./sim/beachDay";
import { SPEEDS, useBeachDay } from "./sim/useBeachDay";
import { colors } from "./theme";

const SKIN_TYPES = ["", "I", "II", "III", "IV", "V", "VI"];

export function CountdownScreen() {
  const insets = useSafeAreaInsets();
  const { snap, running, setRunning, speed, setSpeed, reset, applySunscreen } = useBeachDay();
  useSunscreenReminders(snap.output.parts, snap.tUnixS, speed, running && !snap.finished);
  const uvi = snap.step.uvi_ambient;
  const cat = uvCategory(uvi);
  // most urgent first
  const parts = [...snap.output.parts].sort((a, b) => a.minutes_left.p10 - b.minutes_left.p10);

  return (
    <ScrollView
      style={styles.screen}
      contentContainerStyle={[styles.content, { paddingTop: insets.top + 12, paddingBottom: insets.bottom + 24 }]}
    >
      <View style={styles.header}>
        <Text style={styles.title}>SunRisk</Text>
        <Text style={styles.clock}>{formatClock(snap.tUnixS)}</Text>
      </View>
      <Text style={styles.simNote}>Simulated beach day · Santa Monica · skin type {SKIN_TYPES[FITZPATRICK]}</Text>

      <View style={styles.conditions}>
        <View style={[styles.uvPill, { backgroundColor: cat.color }]}>
          <Text style={styles.uvValue}>UV {uvi.toFixed(1)}</Text>
          <Text style={styles.uvLabel}>{cat.label}</Text>
        </View>
        <Text style={styles.activity}>{snap.segment.label}</Text>
      </View>

      <CountdownCard output={snap.output} />

      <View style={styles.controls}>
        <Pressable style={styles.button} onPress={() => (snap.finished ? reset() : setRunning(!running))}>
          <Text style={styles.buttonText}>{snap.finished ? "Restart" : running ? "Pause" : "Play"}</Text>
        </Pressable>
        {SPEEDS.map((s) => (
          <Pressable key={s} style={[styles.chip, speed === s && styles.chipOn]} onPress={() => setSpeed(s)}>
            <Text style={[styles.chipText, speed === s && styles.chipTextOn]}>{s}×</Text>
          </Pressable>
        ))}
        {!snap.finished && (
          <Pressable style={styles.chip} onPress={reset}>
            <Text style={styles.chipText}>Reset</Text>
          </Pressable>
        )}
      </View>

      <SunscreenControls onApply={applySunscreen} />
      {__DEV__ && <NotificationTest />}

      <Text style={styles.section}>Body parts</Text>
      <View style={styles.list}>
        {parts.map((p) => (
          <BodyPartRow key={p.body_part} part={p} alertLeadMin={config.ALERT_LEAD_MIN} />
        ))}
      </View>

      <Text style={styles.disclaimer}>
        An awareness tool, not a medical device. Times are estimates with real uncertainty; the big
        number is the cautious end of the range.
      </Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.bg },
  content: { paddingHorizontal: 16, gap: 14 },
  header: { flexDirection: "row", alignItems: "baseline", justifyContent: "space-between" },
  title: { fontSize: 28, fontWeight: "800", color: colors.text },
  clock: { fontSize: 20, fontWeight: "600", color: colors.text, fontVariant: ["tabular-nums"] },
  simNote: { fontSize: 13, color: colors.muted, marginTop: -10 },
  conditions: { flexDirection: "row", alignItems: "center", gap: 12 },
  uvPill: { borderRadius: 12, paddingHorizontal: 12, paddingVertical: 6, alignItems: "center" },
  uvValue: { color: "#fff", fontWeight: "800", fontSize: 16, fontVariant: ["tabular-nums"] },
  uvLabel: { color: "#fff", fontSize: 11 },
  activity: { flex: 1, fontSize: 15, color: colors.text },
  controls: { flexDirection: "row", alignItems: "center", gap: 8, flexWrap: "wrap" },
  button: { backgroundColor: colors.accent, borderRadius: 12, paddingHorizontal: 18, paddingVertical: 10 },
  buttonText: { color: "#fff", fontWeight: "700", fontSize: 15, textAlign: "center" },
  chip: { borderRadius: 10, borderWidth: 1, borderColor: colors.border, paddingHorizontal: 12, paddingVertical: 8, backgroundColor: colors.card },
  chipOn: { backgroundColor: colors.text, borderColor: colors.text },
  chipText: { color: colors.text, fontWeight: "600" },
  chipTextOn: { color: "#fff" },
  section: { fontSize: 18, fontWeight: "700", color: colors.text, marginTop: 4 },
  list: { backgroundColor: colors.card, borderRadius: 16, paddingHorizontal: 16 },
  disclaimer: { fontSize: 12, color: colors.muted, textAlign: "center", marginTop: 8 },
});
