// SDK 58's package entry point eagerly loads ExpoTopicSubscriptionModule, which
// is absent in some Expo Go Android builds. Import only local-notification APIs.
// Keep these version-specific imports here until the upstream fix is released.
export { setNotificationHandler } from "expo-notifications/build/NotificationsHandler";
export { getPermissionsAsync, requestPermissionsAsync } from "expo-notifications/build/NotificationPermissions";
export { setNotificationChannelAsync } from "expo-notifications/build/setNotificationChannelAsync";
export { getNotificationChannelAsync } from "expo-notifications/build/getNotificationChannelAsync";
export { scheduleNotificationAsync } from "expo-notifications/build/scheduleNotificationAsync";
export { cancelScheduledNotificationAsync } from "expo-notifications/build/cancelScheduledNotificationAsync";
export { getAllScheduledNotificationsAsync } from "expo-notifications/build/getAllScheduledNotificationsAsync";
export { AndroidImportance } from "expo-notifications/build/NotificationChannelManager.types";
export { SchedulableTriggerInputTypes } from "expo-notifications/build/Notifications.types";
