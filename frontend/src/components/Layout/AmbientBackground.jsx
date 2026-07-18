/**
 * Subtle animated gradient-mesh background for LandingView - three large,
 * heavily-blurred, low-opacity indigo/violet blobs that slowly drift
 * (see the `rm-drift-*` keyframes in styles/global.css), just enough to
 * make the landing screen feel alive without competing with the
 * centered content. Pure CSS animation, no canvas/particle library.
 */
export function AmbientBackground() {
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden">
      <div className="animate-rm-drift-a absolute left-[10%] top-[15%] h-72 w-72 rounded-full bg-indigo-600/20 blur-3xl" />
      <div className="animate-rm-drift-b absolute right-[12%] top-[35%] h-80 w-80 rounded-full bg-purple-600/15 blur-3xl" />
      <div className="animate-rm-drift-c absolute bottom-[10%] left-[35%] h-64 w-64 rounded-full bg-indigo-500/15 blur-3xl" />
    </div>
  );
}
