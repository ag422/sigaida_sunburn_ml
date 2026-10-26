import { Platform } from "react-native";

import * as Notifications from "./localNotifications";
import { AndroidImportance } from "./localNotifications";

const CHANNEL = "sunscreen-reminders";
const PREFIX = "sunrisk-reapply-";
const supported = Platform.OS === "android" || Platform.OS === "ios";
let channelSetup: Promise<string | undefined> | undefined;

function ensureReminderChannel(): Promise<string | undefined> {
  if (Platform.OS !== "android") return Promise.resolve(undefined);
  // Share setup between the test button and automatic reminders, including failures.
  channelSetup ??= (async () => {
    try {
      if (await Notifications.getNotificationChannelAsync(CHANNEL)) return CHANNEL;
      await Notifications.setNotificationChannelAsync(CHANNEL, {
        name: "Sunscreen reminders", importance: AndroidImportance.HIGH,
      });
      return CHANNEL;
    } catch (error) {
      // Expo Go can reject custom-channel operations. Omit channelId so Expo's
      // native scheduler uses its fallback channel, rather than blocking permission setup.
      console.warn("Custom sunscreen channel unavailable; using the default notification channel", error);
      return undefined;
    }
  })();
  return channelSetup;
}

if (supported) {
  Notifications.setNotificationHandler({
    handleNotification: async () => ({
      shouldShowBanner: true, shouldShowList: true, shouldPlaySound: true, shouldSetBadge: false,
    }),
  });
}

/** Channel creation must precede the Android 13+ permission prompt. */
export async function requestReminderPermissions(): Promise<boolean> {
  if (!supported) return false;
  try {
    await ensureReminderChannel();
    const permission = await Notifications.getPermissionsAsync();
    if (permission.granted) return true;
    if (!permission.canAskAgain) return false;
    return (await Notifications.requestPermissionsAsync()).granted;
  } catch (error) {
    console.warn("Unable to enable sunscreen reminders", error);
    return false;
  }
}

export async function scheduleReminder(part: string, seconds: number): Promise<string> {
  const channelId = await ensureReminderChannel();
  return Notifications.scheduleNotificationAsync({
    identifier: `${PREFIX}${part}`,
    content: {
      title: "Time to Reapply Sunscreen",
      body: "Reapply sunscreen to stay protected outdoors.",
      sound: "default", data: { bodyPart: part },
    },
    trigger: {
      type: Notifications.SchedulableTriggerInputTypes.TIME_INTERVAL,
      seconds: Math.max(1, Math.ceil(seconds)), ...(channelId ? { channelId } : {}),
    },
  });
}

export async function cancelReminder(identifier: string): Promise<void> {
  await Notifications.cancelScheduledNotificationAsync(identifier);
}

interface Timer { id: string; due: number; notified: boolean; thresholdReached: boolean }

/** Owns notification state only; receives countdowns without accessing the risk engine. */
export function createReminderService() {
  const timers = new Map<string, Timer>();
  let initialized = false;
  let previousTime = Infinity;
  let queue = Promise.resolve();
  let permission: Promise<boolean> | undefined;

  return {
    update(parts: { body_part: string; minutes_left: { p10: number } }[],
      simulatedTime: number, speed: number, running: boolean): Promise<void> {
      // Serialize OS operations so fast UI updates cannot create duplicate requests.
      queue = queue.then(async () => {
        if (!supported) return;
        permission ??= requestReminderPermissions();
        if (!await permission) {
          // Recheck settings on later updates, without repeatedly showing a prompt.
          permission = Notifications.getPermissionsAsync().then((result) => result.granted);
          return;
        }
        const reset = simulatedTime < previousTime;
        previousTime = simulatedTime;
        if (!initialized || reset) {
          const pending = await Notifications.getAllScheduledNotificationsAsync();
          for (const request of pending) {
            if (request.identifier.startsWith(PREFIX)) await cancelReminder(request.identifier);
          }
          timers.clear();
          initialized = true;
        }
        const now = Date.now();
        for (const part of parts) {
          const minutes = part.minutes_left.p10;
          let timer = timers.get(part.body_part);
          if (timer && timer.due <= now) timer.notified = true;
          // A countdown above the threshold re-arms this part after reapplication/recovery.
          if (timer?.notified && minutes <= 10) continue;
          if (timer?.notified) { timers.delete(part.body_part); timer = undefined; }
          if (!running || !Number.isFinite(minutes)) {
            if (timer) { await cancelReminder(timer.id); timers.delete(part.body_part); }
            continue;
          }
          if (minutes <= 10) {
            // A simulation tick may jump straight past 10 minutes. Trigger once
            // now instead of relying on a previously projected alarm time.
            if (timer?.thresholdReached) continue;
            if (timer) await cancelReminder(timer.id);
            const id = await scheduleReminder(part.body_part, 1);
            timers.set(part.body_part, { id, due: now + 1000, notified: false, thresholdReached: true });
            continue;
          }
          const seconds = Math.max(1, (minutes - 10) * 60 / speed);
          const due = now + Math.ceil(seconds) * 1000;
          // Small prediction drift does not require rewriting the same OS alarm every tick.
          if (timer && Math.abs(timer.due - due) < 2000) continue;
          if (timer) await cancelReminder(timer.id);
          const id = await scheduleReminder(part.body_part, seconds);
          timers.set(part.body_part, { id, due, notified: false, thresholdReached: false });
        }
      }).catch((error) => { console.warn("Unable to update sunscreen reminders", error); });
      return queue;
    },
  };
}
