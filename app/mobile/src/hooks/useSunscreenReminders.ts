import type { BodyPartRisk } from "@sunrisk/risk-engine";
import { useEffect, useState } from "react";

import { createReminderService } from "../notifications";

export function useSunscreenReminders(parts: BodyPartRisk[], time: number, speed: number, running: boolean) {
  const [service] = useState(createReminderService);
  useEffect(() => {
    void service.update(parts, time, speed, running);
    // Do not cancel on unmount: the OS must retain reminders after the app closes.
  }, [service, parts, time, speed, running]);
}
