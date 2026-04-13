export function ServiceCard({ title, description }) {
  return (
    <article className="service-card">
      <h3>{title}</h3>
      <p>{description}</p>
    </article>
  );
}
