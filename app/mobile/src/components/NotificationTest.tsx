import { useState } from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";

import { cancelReminder, requestReminderPermissions, scheduleReminder } from "../notifications";
import { colors } from "../theme";

/** Development-only device check, independent of predictions and simulation controls. */
export function NotificationTest() {
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  async function test() {
    setBusy(true);
    try {
      if (!await requestReminderPermissions()) {
        setMessage("Notifications unavailable or blocked. Enable notifications for Expo Go in your phone settings, then try again.");
        return;
      }
      await cancelReminder("sunrisk-reapply-device-test");
      await scheduleReminder("device-test", 5);
      setMessage("Test scheduled for 5 seconds. Check your notification tray; you can also put Expo Go in the background. If nothing arrives, check Expo Go notification settings, the Sunscreen reminders channel, and Do Not Disturb.");
    } catch (error) {
      setMessage(`Could not schedule notification: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <View style={styles.container}>
      <Pressable accessibilityRole="button" disabled={busy} onPress={test} style={styles.button}>
        <Text style={styles.label}>{busy ? "Scheduling…" : "Test notification in 5 seconds"}</Text>
      </Pressable>
      {!!message && <Text accessibilityLiveRegion="polite" style={styles.message}>{message}</Text>}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { gap: 8 },
  button: { minHeight: 44, borderRadius: 10, borderWidth: 1, borderColor: colors.border, padding: 12 },
  label: { color: colors.text, fontWeight: "600", textAlign: "center" },
  message: { color: colors.muted, fontSize: 13 },
});
