export function StatCard({ value, label }) {
  return (
    <article className="stat-card">
      <p className="stat-card__value">{value}</p>
      <p className="stat-card__label">{label}</p>
    </article>
  );
}
