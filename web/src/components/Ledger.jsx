const human = (f) => f.replace(/_/g, " ");

function EligibleRow({ row, index }) {
  const why = [
    ...row.reasons,
    row.documents_required.length
      ? "documents: " + row.documents_required.slice(0, 3).join(", ")
      : null,
  ]
    .filter(Boolean)
    .join(" · ");
  return (
    <div className="row">
      <span className="idx num">{index}</span>
      <span className="name">{row.scheme_name}</span>
      <span className="stamp" title="Eligible">
        ✓
      </span>
      {why && <span className="why">{why}</span>}
    </div>
  );
}

function CandidateRow({ row, index }) {
  const fields = row.missing_fields.slice(0, 3);
  const extra = row.missing_fields.length - fields.length;
  return (
    <div className="row">
      <span className="idx num">{index}</span>
      <span className="name">{row.scheme_name}</span>
      <span className="needs">
        {fields.map((f, i) => (
          <span key={f} className="need">
            {i === 0 ? "needs: " : ""}
            {human(f)}
          </span>
        ))}
        {extra > 0 && <span className="need">+{extra}</span>}
      </span>
    </div>
  );
}

export default function Ledger({ variant, title, total, rows, remaining, onMore, busy }) {
  if (!total) return null;
  const Row = variant === "eligible" ? EligibleRow : CandidateRow;
  return (
    <section className="ledger" aria-label={title}>
      <div className="ledger-head">
        <span className="eyebrow">{title}</span>
        <span className="count num">{total.toLocaleString()}</span>
        <span className="showing num">showing 1–{rows.length}</span>
      </div>
      {rows.map((r, i) => (
        <Row key={r.scheme_id} row={r} index={i + 1} />
      ))}
      {remaining > 0 && (
        <button className="more-btn" onClick={onMore} disabled={busy}>
          Show {Math.min(5, remaining)} more of {remaining.toLocaleString()} remaining ↓
        </button>
      )}
    </section>
  );
}
