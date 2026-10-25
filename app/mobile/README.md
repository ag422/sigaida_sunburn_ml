# SunRisk mobile app

Expo (React Native) app showing a time-until-sunburn countdown. An awareness tool, not a
medical device.

Right now it plays a **simulated beach day** (`src/sim/`) through the TypeScript risk engine
(`../risk-engine`), because the wearable isn't built yet.

## Run it

From `app/` (the npm workspace root, so the risk engine gets linked):

```bash
npm install
npm start          # then scan the QR code with Expo Go on your phone, or press w for web
```

If Expo Go hangs on the loading screen, the phone probably can't reach your computer (common
on school Wi-Fi). Use a tunnel instead, from `app/mobile`:

```bash
npx expo start --tunnel ngrok
```

On Windows PowerShell, if `npm`/`npx` fail with "running scripts is disabled", use `npm.cmd` /
`npx.cmd` instead.

Checks, from `app/mobile`: `npm run typecheck`, `npm run lint`.

## How it fits together

| File | What it does |
|---|---|
| `src/CountdownScreen.tsx` | the screen: UV now, headline countdown, controls, body-part list |
| `src/components/CountdownCard.tsx` | headline: p10 minutes for the first body part to burn, plus the p10–p90 range |
| `src/components/BodyPartRow.tsx` | one body part: p10, range, share of burn dose used, effective SPF |
| `src/sim/beachDay.ts` | the scripted day: schedule, morning sunscreen, placeholder exposure ratios |
| `src/sim/useBeachDay.ts` | runs the scripted day through `ExposureTracker` on a sped-up clock |

UI rule from `docs/contracts.md`: the big number is always **p10** (the cautious end), and the
p10–p90 range is always shown.

When the device works, replace `useBeachDay` with a hook that turns BLE packets into
`RiskStep`s; the screen only needs a `Snapshot`.

## Not done yet

- Phone notifications when p10 drops below 15 minutes (the alert currently shows on screen only)
- Real exposure ratios per body part (needs `geometry.py` / `pipeline.py` ported)
- Weather forecast feed, sunscreen logging with SPF/body-part choice, skin-type setting
- Bluetooth (needs an Expo development build; Expo Go can't do BLE)
