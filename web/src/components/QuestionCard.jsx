import { useEffect, useRef, useState } from "react";

// fields with a defined answer list get controls, not free text —
// these answers skip the LLM entirely (instant + can't be misread)
const OPTIONS = {
  gender: ["male", "female"],
  category: ["General", "SC", "ST", "OBC", "EWS", "BPL"],
  marital_status: ["single", "married", "widow"],
  has_bank_account: ["yes", "no"],
  land_owner: ["yes", "no"],
  // most common occupation tokens in the rules vocabulary
  occupation: ["farmer", "student", "entrepreneur", "construction worker",
    "artist", "fisherman", "artisan", "ex-serviceman", "faculty",
    "sportsperson", "weaver", "journalist"],
};

const STATES = [
  "Andaman and Nicobar Islands", "Andhra Pradesh", "Arunachal Pradesh",
  "Assam", "Bihar", "Chandigarh", "Chhattisgarh",
  "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Goa", "Gujarat",
  "Haryana", "Himachal Pradesh", "Jammu and Kashmir", "Jharkhand",
  "Karnataka", "Kerala", "Ladakh", "Lakshadweep", "Madhya Pradesh",
  "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha",
  "Puducherry", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana",
  "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
];

const NUMERIC = { age: "e.g. 40", annual_income: "e.g. 90000 (rupees per year)" };

export default function QuestionCard({
  nextQuestion,
  started,
  busy,
  onSubmit,
  onAnswerField,
}) {
  const [value, setValue] = useState("");
  const [numValue, setNumValue] = useState("");
  const [stateValue, setStateValue] = useState("");
  const inputRef = useRef(null);
  const field = nextQuestion?.field;

  useEffect(() => {
    setNumValue("");
    setStateValue("");
    // preventScroll: focusing the input must not yank the page down to it
    if (!busy && !field) inputRef.current?.focus({ preventScroll: true });
  }, [busy, field]);

  const submitText = (e) => {
    e.preventDefault();
    const text = value.trim();
    if (!text || busy) return;
    setValue("");
    onSubmit(text);
  };

  const submitNum = (e) => {
    e.preventDefault();
    const n = numValue.trim();
    if (!n || busy) return;
    onAnswerField(field, n);
  };

  const freeText = (placeholder) => (
    <form className="answer-bar" onSubmit={submitText}>
      <input
        ref={inputRef}
        type="text"
        value={value}
        placeholder={placeholder}
        onChange={(e) => setValue(e.target.value)}
        disabled={busy}
        aria-label="Your message"
      />
      <button type="submit" disabled={busy || !value.trim()}>
        {started ? "Send" : "Start"}
      </button>
    </form>
  );

  if (nextQuestion) {
    let control;
    if (OPTIONS[field]) {
      control = (
        <div className="option-row" role="group" aria-label="Answer options">
          {OPTIONS[field].map((opt) => (
            <button
              key={opt}
              className="option-chip"
              disabled={busy}
              onClick={() => onAnswerField(field, opt)}
            >
              {opt}
            </button>
          ))}
        </div>
      );
    } else if (field === "state") {
      // explicit confirm — a change event alone must never submit (autofill
      // or programmatic changes would silently answer for the user)
      control = (
        <div className="answer-bar" style={{ marginTop: 16 }}>
          <select
            className="state-select"
            style={{ marginTop: 0, flex: 1 }}
            disabled={busy}
            value={stateValue}
            autoComplete="off"
            aria-label="Select your state"
            onChange={(e) => setStateValue(e.target.value)}
          >
            <option value="" disabled>
              Select your state or union territory…
            </option>
            {STATES.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
          <button
            type="button"
            disabled={busy || !stateValue}
            onClick={() => onAnswerField("state", stateValue)}
          >
            Answer
          </button>
        </div>
      );
    } else {
      control = (
        <form className="answer-bar" onSubmit={submitNum}>
          <input
            type="number"
            min="0"
            inputMode="numeric"
            value={numValue}
            placeholder={NUMERIC[field] || "Type your answer…"}
            onChange={(e) => setNumValue(e.target.value)}
            disabled={busy}
            aria-label="Your answer"
          />
          <button type="submit" disabled={busy || !numValue.trim()}>
            Answer
          </button>
        </form>
      );
    }

    return (
      <section className="question" aria-label="Next question">
        <span className="eyebrow">
          {nextQuestion.unlocks
            ? `One question unlocks ${nextQuestion.unlocks.toLocaleString()} of these`
            : "To narrow this down"}
        </span>
        <h2>{nextQuestion.question}</h2>
        {control}
        <div className="or-else">{freeText("Or tell me anything else about yourself…")}</div>
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
      {freeText(
        started
          ? "Tell me more…"
          : "e.g. I'm a farmer in Tamil Nadu looking for support"
      )}
    </section>
  );
}
