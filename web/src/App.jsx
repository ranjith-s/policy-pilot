import { useCallback, useEffect, useState } from "react";
import Header from "./components/Header.jsx";
import Ledger from "./components/Ledger.jsx";
import QuestionCard from "./components/QuestionCard.jsx";
import RecordPanel from "./components/RecordPanel.jsx";

const EXAMPLES = [
  "I'm a farmer in Tamil Nadu looking for support",
  "I'm a 62 year old artist, what support can I get?",
  "I'm a widow living in Delhi",
  "I want to start a small business as an SC woman entrepreneur",
];

export default function App() {
  const [sessionId, setSessionId] = useState(null);
  const [stats, setStats] = useState(null);
  const [utterance, setUtterance] = useState("");
  const [data, setData] = useState(null); // funnel payload for current view
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let tries = 0;
    let timer;
    const poll = () =>
      fetch("/api/stats")
        .then((r) => (r.ok ? r.json() : Promise.reject()))
        .then(setStats)
        .catch(() => {
          if (++tries < 15) timer = setTimeout(poll, 2000);
        });
    poll();
    return () => clearTimeout(timer);
  }, []);

  const send = useCallback(
    async (message) => {
      setLoading(true);
      setError(null);
      const isMore = /^more$/i.test(message.trim());
      if (!isMore) setUtterance(message);
      try {
        const r = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message, session_id: sessionId }),
        });
        if (!r.ok) throw new Error(`server returned ${r.status}`);
        const out = await r.json();
        setSessionId(out.session_id);
        if (out.data) {
          setData((prev) => {
            const d = out.data;
            // 'more' pages arrive with a non-zero cursor: append, don't replace
            if (prev && (d.page_start.eligible > 0 || d.page_start.candidates > 0)) {
              return {
                ...d,
                eligible: [...prev.eligible, ...d.eligible],
                candidates: [...prev.candidates, ...d.candidates],
                page_start: prev.page_start,
              };
            }
            return d;
          });
        }
      } catch (e) {
        setError(
          `Couldn't reach the assistant (${e.message}). ` +
            "Is the API running? Start it with: uvicorn api.server:app --port 8000"
        );
      } finally {
        setLoading(false);
      }
    },
    [sessionId]
  );

  const reset = useCallback(async () => {
    if (sessionId) {
      await fetch("/api/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "", session_id: sessionId }),
      }).catch(() => {});
    }
    setSessionId(null);
    setData(null);
    setUtterance("");
    setError(null);
  }, [sessionId]);

  return (
    <>
      <Header stats={stats} />
      <div className="wrap main-grid">
        <main>
          {utterance && (
            <div className="you-line">
              <span className="you-label">You</span>“{utterance}”
            </div>
          )}

          {error && <div className="error-note">{error}</div>}

          {!data && !loading && (
            <section className="welcome">
              <h1>Know what you're entitled to.</h1>
              <p>
                Tell me about yourself in your own words. I'll check the
                eligibility rules of {stats ? stats.rule_checked.toLocaleString() : "3,810"}{" "}
                Indian government schemes and show you what you can claim —
                and exactly which facts confirm the rest.
              </p>
              <div className="examples">
                {EXAMPLES.map((ex) => (
                  <button key={ex} className="example" onClick={() => send(ex)}>
                    {ex}
                  </button>
                ))}
              </div>
            </section>
          )}

          {loading && (
            <div className="thinking">
              <span className="dot" />
              Reading your message, then checking every scheme rule…
            </div>
          )}

          {data && (
            <>
              <Ledger
                variant="eligible"
                title="Eligible by the rules check"
                total={data.counts.eligible}
                rows={data.eligible}
                remaining={data.counts.eligible - data.shown.eligible}
                onMore={() => send("more")}
                busy={loading}
              />
              <Ledger
                variant="candidates"
                title="Likely matches — answer below to confirm"
                total={data.counts.candidates}
                rows={data.candidates}
                remaining={data.counts.candidates - data.shown.candidates}
                onMore={() => send("more")}
                busy={loading}
              />
            </>
          )}

          <QuestionCard
            nextQuestion={data?.next_question}
            started={!!data}
            busy={loading}
            onSubmit={send}
          />
        </main>

        <RecordPanel
          profile={data?.profile}
          askedField={data?.next_question?.field}
          onReset={reset}
        />
      </div>
      <footer>
        <div className="wrap">
          Haqdar is an independent assistant built on public myScheme data.
          Results are indicative only — always verify on the{" "}
          <a href="https://www.myscheme.gov.in" target="_blank" rel="noreferrer">
            official myScheme portal
          </a>{" "}
          before applying.
        </div>
      </footer>
    </>
  );
}
