import { AppShell } from "./components/Layout/AppShell";
import { RepoProvider } from "./state/RepoContext";

export default function App() {
  return (
    <RepoProvider>
      <AppShell />
    </RepoProvider>
  );
}
