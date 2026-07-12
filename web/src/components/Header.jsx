import { useEffect, useState } from "react";

export default function Header({ stats }) {
  const [theme, setTheme] = useState(null); // null = follow system

  useEffect(() => {
    if (theme) document.documentElement.dataset.theme = theme;
    else delete document.documentElement.dataset.theme;
  }, [theme]);

  const isDark =
    theme === "dark" ||
    (!theme && window.matchMedia("(prefers-color-scheme: dark)").matches);

  return (
    <header>
      <div className="wrap header-row">
        <div className="seal" aria-hidden="true">
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">
            {/* compass needle — guidance, language-neutral */}
            <path d="M10 1 L13 10 L10 19 L7 10 Z" fill="currentColor" />
            <circle cx="10" cy="10" r="1.6" fill="var(--teal)" />
          </svg>
        </div>
        <div>
          <div className="brand-name">PolicyPilot</div>
          <div className="brand-sub">
            Public Scheme Eligibility Assistant · myScheme corpus
          </div>
        </div>
        <div className="header-stats">
          {stats ? (
            <>
              <span className="num">
                <b>{stats.schemes.toLocaleString()}</b> schemes
              </span>{" "}
              ·{" "}
              <span className="num">
                <b>{stats.rule_checked.toLocaleString()}</b> rule-checked
              </span>
              <br />
              <span>0 verdicts guessed by AI</span>
            </>
          ) : (
            <span>connecting…</span>
          )}
        </div>
        <button
          className="theme-btn"
          onClick={() => setTheme(isDark ? "light" : "dark")}
          aria-label="Toggle theme"
        >
          {isDark ? "Light" : "Dark"}
        </button>
      </div>
    </header>
  );
}
