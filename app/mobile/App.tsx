import { StatusBar } from "expo-status-bar";
import { SafeAreaProvider } from "react-native-safe-area-context";

import { CountdownScreen } from "./src/CountdownScreen";

export default function App() {
  return (
    <SafeAreaProvider>
      <CountdownScreen />
      <StatusBar style="dark" />
    </SafeAreaProvider>
  );
}
