import { useEffect, useRef, useState } from "react";

export default function QuestionCard({ nextQuestion, started, busy, onSubmit }) {
  const [value, setValue] = useState("");
  const inputRef = useRef(null);

  useEffect(() => {
    // preventScroll: focusing the input must not yank the page down to it
    if (!busy) inputRef.current?.focus({ preventScroll: true });
  }, [busy, nextQuestion?.question]);

  const submit = (e) => {
    e.preventDefault();
    const text = value.trim();
    if (!text || busy) return;
    setValue("");
    onSubmit(text);
  };

  const bar = (placeholder) => (
    <form className="answer-bar" onSubmit={submit}>
      <input
        ref={inputRef}
        type="text"
        value={value}
        placeholder={placeholder}
        onChange={(e) => setValue(e.target.value)}
        disabled={busy}
        aria-label="Your answer"
      />
      <button type="submit" disabled={busy || !value.trim()}>
        {started ? "Answer" : "Start"}
      </button>
    </form>
  );

  if (nextQuestion) {
    return (
      <section className="question" aria-label="Next question">
        <span className="eyebrow">
          {nextQuestion.unlocks
            ? `One question unlocks ${nextQuestion.unlocks.toLocaleString()} of these`
            : "To narrow this down"}
        </span>
        <h2>{nextQuestion.question}</h2>
        <p>Answer it — or tell me anything else about yourself.</p>
        {bar("Type your answer…")}
      </section>
    );
  }

  return (
    <section className="plain-ask" aria-label="Message">
      {started && (
        <p style={{ margin: "0 0 4px", fontSize: 13, color: "var(--ink-2)" }}>
          Your record is complete for these rules. Add anything else, or refine
          what you told me.
        </p>
      )}
      {bar(
        started
          ? "Tell me more…"
          : "e.g. I'm a farmer in Tamil Nadu looking for support"
      )}
    </section>
  );
}
