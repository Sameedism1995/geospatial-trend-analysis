import { Link } from 'react-router-dom';

export function ProductCard({ product }) {
  return (
    <Link className="product-card" to={product.href}>
      <div className="product-card__image" aria-hidden="true">
        <span>Image</span>
      </div>
      <div className="product-card__body">
        <h3>{product.name}</h3>
        <p>{product.description}</p>
        <span className="product-card__cta">{product.cta}</span>
      </div>
    </Link>
  );
}
