import { Navigate, useParams } from 'react-router-dom';
import { ButtonLink } from '../components/ButtonLink';
import { productBySlug } from '../content/products';

export function ProductDetailPage() {
  const { slug } = useParams();
  const product = productBySlug[slug];

  if (!product) {
    return <Navigate to="/" replace />;
  }

  return (
    <section className="page-section page-section--narrow">
      <div className="container">
        <p className="eyebrow">Product</p>
        <h1>{product.title}</h1>
        <div className="hero__actions">
          <ButtonLink href={`/shop/${product.slug}`}>Shop now</ButtonLink>
        </div>
      </div>
    </section>
  );
}
