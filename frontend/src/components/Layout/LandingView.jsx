import { useState } from "react";
import { ChatIcon, GraphIcon, ShieldIcon } from "../common/icons";
import { IndexingProgress } from "../Sidebar/IndexingProgress";
import { RepoInputPanel } from "../Sidebar/RepoInputPanel";
import { AmbientBackground } from "./AmbientBackground";

const FEATURES = [
  { icon: ChatIcon, label: "Chat with any codebase" },
  { icon: GraphIcon, label: "Visualize the call graph" },
  { icon: ShieldIcon, label: "Catch bugs before they ship" },
];

const EXAMPLE_REPOS = [
  { label: "facebook/react", url: "https://github.com/facebook/react" },
  { label: "expressjs/express", url: "https://github.com/expressjs/express" },
  { label: "navdeep-G/samplemod", url: "https://github.com/navdeep-G/samplemod" },
];

/**
 * Pre-workspace landing state: centered title + repo URL input, no
 * sidebar - plus the content additions that fill the surrounding space
 * (badge, feature highlights, example-repo chips, ambient background,
 * footer). Reuses `RepoInputPanel`/`IndexingProgress` unchanged (the
 * same components the workspace sidebar renders) driven by the same
 * lifted `indexing` state from AppShell - not a second copy of the
 * indexing logic, just a second render site for the same state.
 *
 * `prefillUrl` (local to this component, purely additive) feeds
 * `RepoInputPanel`'s new optional `prefillUrl` prop - clicking an
 * example chip only seeds the field's text, the existing submit/
 * validation/indexing flow is untouched and still requires the normal
 * "Index" click (or Enter) to actually trigger anything.
 */
export function LandingView({ indexing }) {
  const [prefillUrl, setPrefillUrl] = useState(undefined);
  const isIdle = indexing.phase === "idle";

  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center overflow-hidden">
      <AmbientBackground />

      <div className="relative z-10 flex w-full max-w-3xl flex-col items-center gap-10 px-6 text-center">
        <span className="rounded-full border border-indigo-400/30 bg-indigo-500/10 px-3 py-1 text-[11px] font-medium tracking-wide text-indigo-300">
          Adversarial Code Review Engine
        </span>

        <div>
          <h1 className="bg-gradient-to-r from-white to-slate-300 bg-clip-text text-7xl font-bold tracking-tight text-transparent">
            Verdict AI
          </h1>
          <p className="mx-auto mt-5 max-w-xl text-xl leading-relaxed text-slate-400">
            Point it at a GitHub repository to explore its call graph and chat with its code.
          </p>
        </div>

        {isIdle && (
          <div className="flex items-start justify-center gap-14">
            {FEATURES.map(({ icon: Icon, label }) => (
              <div key={label} className="flex w-28 flex-col items-center gap-3">
                <Icon className="h-8 w-8 text-indigo-400" />
                <span className="text-sm leading-tight text-slate-400">{label}</span>
              </div>
            ))}
          </div>
        )}

        <div className="w-full max-w-2xl">
          <RepoInputPanel indexing={indexing} onSubmit={indexing.start} prefillUrl={prefillUrl} size="lg" />
        </div>

        {isIdle && (
          <div className="flex flex-col items-center gap-3">
            <span className="text-sm text-slate-500">Try an example:</span>
            <div className="flex flex-wrap justify-center gap-3">
              {EXAMPLE_REPOS.map((repo) => (
                <button
                  key={repo.url}
                  type="button"
                  onClick={() => setPrefillUrl(repo.url)}
                  className="rounded-full border border-white/10 bg-slate-900/40 px-5 py-2.5 font-mono text-sm text-slate-300 backdrop-blur-xl transition hover:border-indigo-400/40 hover:text-indigo-300"
                >
                  {repo.label}
                </button>
              ))}
            </div>
          </div>
        )}

        {!isIdle && (
          <div className="w-full max-w-2xl">
            <IndexingProgress indexing={indexing} />
          </div>
        )}
      </div>

      <footer className="absolute inset-x-0 bottom-4 text-center text-[11px] text-slate-600">
        Built with FastAPI, Gemini, and a real call graph.
      </footer>
    </div>
  );
}
