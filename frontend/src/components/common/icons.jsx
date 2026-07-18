/**
 * Small inline SVG icons - no icon-library dependency, kept self-contained.
 * Warning/Check/Spinner: Alert's warning icon, IndexingProgress's
 * checkmark/spinner. Chat/Graph/Shield: LandingView's feature-highlight
 * row. Repo/Search: RepoInputPanel's summary card, ChatInputBar's input.
 * XIcon/GavelIcon: ReviewView's claim/Judge cards.
 */

export function WarningIcon({ className = "h-4 w-4" }) {
  return (
    <svg viewBox="0 0 20 20" fill="none" className={className} aria-hidden="true">
      <path
        d="M8.257 3.099c.765-1.36 2.72-1.36 3.486 0l6.28 11.18c.75 1.334-.213 2.987-1.744 2.987H3.72c-1.53 0-2.493-1.653-1.744-2.987l6.28-11.18Z"
        stroke="currentColor"
        strokeWidth="1.5"
      />
      <path d="M10 7.5v4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      <circle cx="10" cy="14" r="0.9" fill="currentColor" />
    </svg>
  );
}

export function CheckIcon({ className = "h-3.5 w-3.5" }) {
  return (
    <svg viewBox="0 0 20 20" fill="none" className={className} aria-hidden="true">
      <path d="M4 10.5l4 4 8-9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function SpinnerIcon({ className = "h-3.5 w-3.5" }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" className={`animate-spin ${className}`} aria-hidden="true">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="3" className="opacity-25" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" className="opacity-90" />
    </svg>
  );
}

export function ChatIcon({ className = "h-4 w-4" }) {
  return (
    <svg viewBox="0 0 20 20" fill="none" className={className} aria-hidden="true">
      <path
        d="M3 5.5A2.5 2.5 0 0 1 5.5 3h9A2.5 2.5 0 0 1 17 5.5v5A2.5 2.5 0 0 1 14.5 13H9l-4 3.2V13H5.5A2.5 2.5 0 0 1 3 10.5v-5Z"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function GraphIcon({ className = "h-4 w-4" }) {
  return (
    <svg viewBox="0 0 20 20" fill="none" className={className} aria-hidden="true">
      <circle cx="4.5" cy="5" r="2" stroke="currentColor" strokeWidth="1.4" />
      <circle cx="15.5" cy="5" r="2" stroke="currentColor" strokeWidth="1.4" />
      <circle cx="10" cy="15" r="2" stroke="currentColor" strokeWidth="1.4" />
      <path d="M6.3 6.1 8.5 13M13.7 6.1 11.5 13M6.5 5h7" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}

export function ShieldIcon({ className = "h-4 w-4" }) {
  return (
    <svg viewBox="0 0 20 20" fill="none" className={className} aria-hidden="true">
      <path
        d="M10 2.5 16 4.5v4.7c0 4-2.6 6.7-6 8.3-3.4-1.6-6-4.3-6-8.3V4.5L10 2.5Z"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
      <path d="M7.3 10 9.2 11.8 12.8 8.2" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function RepoIcon({ className = "h-3.5 w-3.5" }) {
  return (
    <svg viewBox="0 0 20 20" fill="none" className={className} aria-hidden="true">
      <path
        d="M5 2.5h8A1.5 1.5 0 0 1 14.5 4v13l-4.5-2.2L5.5 17V4A1.5 1.5 0 0 1 5 2.5Z"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function SearchIcon({ className = "h-3.5 w-3.5" }) {
  return (
    <svg viewBox="0 0 20 20" fill="none" className={className} aria-hidden="true">
      <circle cx="8.5" cy="8.5" r="5" stroke="currentColor" strokeWidth="1.4" />
      <path d="m16 16-3.8-3.8" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}

export function XIcon({ className = "h-3.5 w-3.5" }) {
  return (
    <svg viewBox="0 0 20 20" fill="none" className={className} aria-hidden="true">
      <path d="M5 5l10 10M15 5 5 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

export function GavelIcon({ className = "h-4 w-4" }) {
  return (
    <svg viewBox="0 0 20 20" fill="none" className={className} aria-hidden="true">
      <path d="m8 4 4 4-6 6-4-4 6-6Z" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" />
      <path d="m10.5 6.5 4-4 3 3-4 4M2 18h9" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
