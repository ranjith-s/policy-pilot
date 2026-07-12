const FIELDS = [
  ["occupation", "Occupation"],
  ["state", "State"],
  ["age", "Age"],
  ["annual_income", "Annual income"],
  ["gender", "Gender"],
  ["category", "Category"],
  ["marital_status", "Marital status"],
  ["has_bank_account", "Bank account"],
  ["land_owner", "Owns land"],
];

const fmt = (k, v) => {
  if (v === undefined || v === null || v === "") return null;
  if (k === "annual_income") {
    const n = Number(String(v).replace(/,/g, ""));
    return Number.isFinite(n) ? `₹${n.toLocaleString("en-IN")}/yr` : String(v);
  }
  if (typeof v === "boolean") return v ? "Yes" : "No";
  return String(v);
};

export default function RecordPanel({ profile, askedField, onReset }) {
  return (
    <aside className="record">
      <div className="record-card">
        <h3>Your record</h3>
        <p className="hint">Fills in as you talk. Nothing leaves this session.</p>
        {FIELDS.map(([key, label]) => {
          const v = fmt(key, profile?.[key]);
          return (
            <div key={key} className={"fact" + (v ? "" : " empty")}>
              <span className="k">{label}</span>
              <span className="dots" />
              <span className="v num">{v ?? "—"}</span>
              {!v && askedField === key && <span className="next">asked now</span>}
            </div>
          );
        })}
        <p className="record-note">
          Each fact you add moves schemes from “likely” to “eligible” — or
          rules them out honestly.
        </p>
        <button className="reset-btn" onClick={onReset}>
          Start over
        </button>
      </div>
      <div className="trust">
        <b>Why you can trust this:</b> every verdict comes from a deterministic
        rules engine reading official eligibility criteria — the AI only reads
        your message, it never decides. Results are indicative; verify on the{" "}
        <a href="https://www.myscheme.gov.in" target="_blank" rel="noreferrer">
          myScheme portal
        </a>
        .
      </div>
    </aside>
  );
}
