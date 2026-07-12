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
          हक़
        </div>
        <div>
          <div className="brand-name">Haqdar</div>
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
