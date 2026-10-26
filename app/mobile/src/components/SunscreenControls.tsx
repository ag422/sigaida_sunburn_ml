import { BODY_PARTS, type BodyPart } from "@sunrisk/risk-engine";
import { useState } from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";

import { PART_LABELS } from "../format";
import { colors } from "../theme";

const SPF_OPTIONS = [15, 20, 25, 30, 40, 45, 50, 60, 70, 100];

export function SunscreenControls({ onApply }: { onApply: (spf: number, parts?: BodyPart[]) => void }) {
  const [spf, setSpf] = useState(30);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [everywhere, setEverywhere] = useState(true);
  const [selectedParts, setSelectedParts] = useState<BodyPart[]>([]);
  const [confirmation, setConfirmation] = useState("");
  const disabled = !everywhere && selectedParts.length === 0;

  function togglePart(part: BodyPart) {
    setSelectedParts((previous) => previous.includes(part)
      ? previous.filter((p) => p !== part) : [...previous, part]);
    setConfirmation("");
  }

  return (
    <View style={styles.card}>
      <Text style={styles.heading}>Sunscreen</Text>
      <Pressable
        accessibilityRole="button"
        accessibilityLabel={`Sunscreen SPF, ${spf}`}
        accessibilityState={{ expanded: dropdownOpen }}
        style={styles.option}
        onPress={() => setDropdownOpen(!dropdownOpen)}
      >
        <Text style={styles.optionText}>SPF {spf} {dropdownOpen ? "▲" : "▼"}</Text>
      </Pressable>
      {dropdownOpen && (
        <View style={styles.options}>
          {SPF_OPTIONS.map((value) => (
            <Pressable
              key={value}
              accessibilityRole="radio"
              accessibilityState={{ checked: spf === value }}
              style={[styles.option, spf === value && styles.selected]}
              onPress={() => { setSpf(value); setDropdownOpen(false); setConfirmation(""); }}
            >
              <Text style={styles.optionText}>SPF {value}</Text>
            </Pressable>
          ))}
        </View>
      )}
      <View style={styles.options}>
        {[true, false].map((all) => (
          <Pressable
            key={String(all)}
            accessibilityRole="radio"
            accessibilityState={{ checked: everywhere === all }}
            style={[styles.option, everywhere === all && styles.selected]}
            onPress={() => { setEverywhere(all); setConfirmation(""); }}
          >
            <Text style={styles.optionText}>{all ? "Everywhere" : "Select body parts"}</Text>
          </Pressable>
        ))}
      </View>
      {!everywhere && (
        <View style={styles.options}>
          {BODY_PARTS.map((part) => (
            <Pressable
              key={part}
              accessibilityRole="checkbox"
              accessibilityState={{ checked: selectedParts.includes(part) }}
              style={[styles.option, selectedParts.includes(part) && styles.selected]}
              onPress={() => togglePart(part)}
            >
              <Text style={styles.optionText}>{selectedParts.includes(part) ? "✓ " : ""}{PART_LABELS[part]}</Text>
            </Pressable>
          ))}
        </View>
      )}
      {disabled && <Text style={styles.note}>Select at least one body part.</Text>}
      <Pressable
        accessibilityRole="button"
        accessibilityState={{ disabled }}
        disabled={disabled}
        style={[styles.apply, disabled && styles.disabled]}
        onPress={() => {
          onApply(spf, everywhere ? undefined : selectedParts);
          setConfirmation(`Logged SPF ${spf} ${everywhere ? "everywhere" : `on ${selectedParts.map((p) => PART_LABELS[p]).join(", ")}`}.`);
        }}
      >
        <Text style={styles.applyText}>Applied SPF {spf} {everywhere ? "everywhere" : "to selected parts"}</Text>
      </Pressable>
      {!!confirmation && <Text accessibilityLiveRegion="polite" style={styles.note}>{confirmation}</Text>}
    </View>
  );
}

const styles = StyleSheet.create({
  card: { backgroundColor: colors.card, padding: 16, borderRadius: 16, gap: 12 },
  heading: { color: colors.text, fontSize: 18, fontWeight: "700" },
  options: { flexDirection: "row", flexWrap: "wrap", gap: 8 },
  option: { minHeight: 44, justifyContent: "center", borderWidth: 1, borderColor: colors.border, borderRadius: 10, paddingHorizontal: 12, paddingVertical: 10 },
  selected: { borderColor: colors.accent, backgroundColor: colors.accentBg },
  optionText: { color: colors.text, fontSize: 15, fontWeight: "600" },
  apply: { minHeight: 44, backgroundColor: colors.accent, borderRadius: 12, paddingHorizontal: 18, paddingVertical: 12 },
  disabled: { opacity: 0.45 },
  applyText: { color: "#fff", fontWeight: "700", fontSize: 15, textAlign: "center" },
  note: { color: colors.muted, fontSize: 13 },
});
