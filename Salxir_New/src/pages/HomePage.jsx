import { Link } from 'react-router-dom';
import { ButtonLink } from '../components/ButtonLink';
import { SectionIntro } from '../components/SectionIntro';
import { consumerContent } from '../content/siteContent';

export function HomePage() {
  return (
    <>
      <section className="hero page-section">
        <div className="container hero__grid">
          <div>
            <p className="eyebrow">{consumerContent.hero.eyebrow}</p>
            <h1>{consumerContent.hero.title}</h1>
            <p className="hero__copy">{consumerContent.hero.description}</p>
            <div className="hero__actions">
              <ButtonLink href={consumerContent.hero.primaryCta.href}>
                {consumerContent.hero.primaryCta.label}
              </ButtonLink>
              <ButtonLink href="/subscribe" variant="secondary">
                Subscribe now
              </ButtonLink>
            </div>
          </div>
          <aside className="hero-panel hero-panel--accent" aria-label="Featured products">
            <p className="hero-panel__title">Product Facts</p>
            <dl className="facts-list">
              <div>
                <dt>Origin</dt>
                <dd>{consumerContent.stats.origin}</dd>
              </div>
              <div>
                <dt>Price</dt>
                <dd>{consumerContent.stats.productPrice}</dd>
              </div>
              <div>
                <dt>Net Weight</dt>
                <dd>{consumerContent.stats.netWeight}</dd>
              </div>
              <div>
                <dt>Recommended Use</dt>
                <dd>{consumerContent.stats.usage}</dd>
              </div>
            </dl>
            <Link className="hero-panel__cta" to="/shop">
              Explore all products
            </Link>
          </aside>
        </div>
      </section>

      <section className="page-section page-section--compact page-section--tinted">
        <div className="container">
          <SectionIntro
            eyebrow="From Source to You"
            title="Crafted with care at every step"
            description="Transparent sourcing and handling standards designed for confidence."
          />
          <div className="journey-grid">
            {consumerContent.journeySteps.map((step) => (
              <article key={step.title} className="journey-card">
                <div className="journey-card__icon" aria-hidden="true">
                  {step.icon}
                </div>
                <h3>{step.title}</h3>
                <p>{step.description}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="page-section">
        <div className="container">
          <SectionIntro
            eyebrow="What Science Says"
            title="Research highlights and educational references"
            description="Open any research card to view the main research page."
          />
          <div className="science-grid">
            {consumerContent.scienceCards.map((item) => (
              <Link key={item.title} to={item.href} className="science-card">
                <h3>{item.title}</h3>
                <p>{item.description}</p>
                <span>Open research page</span>
              </Link>
            ))}
          </div>
        </div>
      </section>

      <section className="page-section">
        <div className="container">
          <SectionIntro
            eyebrow="Shop"
            title="Explore products"
            description="Hover cards from left to right. Click any card to open its product page."
          />
          <div className="slider-row" role="list" aria-label="Products">
            {consumerContent.products.map((product) => (
              <Link key={product.name} to={product.href} className="slider-card" role="listitem">
                <h3>{product.name}</h3>
                <p>{product.description}</p>
                <span>{product.cta}</span>
              </Link>
            ))}
          </div>
        </div>
      </section>

      <section className="page-section page-section--compact">
        <div className="container">
          <SectionIntro
            eyebrow="What Our Customers Say"
            title="Real stories from daily users"
            description="Swipe through review boxes from left to right."
          />
          <div className="slider-row" role="list" aria-label="Reviews">
            {consumerContent.reviews.map((review) => (
              <article key={review.author} className="slider-card slider-card--review" role="listitem">
                <p>
                  <span aria-hidden="true">&ldquo;</span>
                  {review.quote}
                  <span aria-hidden="true">&rdquo;</span>
                </p>
                <strong>{review.author}</strong>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="page-section page-section--tinted">
        <div className="container newsletter-box">
          <div className="newsletter-content">
            <p className="eyebrow">Stay Connected</p>
            <h2>Join our community and be the first to know about new products.</h2>
            <p className="section-copy">
              Join our community and be the first to know about new products, wellness tips, and
              exclusive offers.
            </p>
            <form className="newsletter-form">
              <label htmlFor="newsletter-email">Enter your email</label>
              <div className="newsletter-form__controls">
                <input id="newsletter-email" type="email" placeholder="you@example.com" />
                <button type="button">Subscribe</button>
              </div>
            </form>
          </div>
        </div>
      </section>
    </>
  );
}
