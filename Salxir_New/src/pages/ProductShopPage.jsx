import { Navigate, useParams } from 'react-router-dom';
import { productBySlug } from '../content/products';

export function ProductShopPage() {
  const { slug } = useParams();
  const product = productBySlug[slug];

  if (!product) {
    return <Navigate to="/shop" replace />;
  }

  return (
    <section className="page-section page-section--narrow">
      <div className="container">
        <p className="eyebrow">Shop</p>
        <h1>{product.title}</h1>
      </div>
    </section>
  );
}
