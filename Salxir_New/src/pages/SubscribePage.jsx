import { ButtonLink } from '../components/ButtonLink';

const plans = [
  {
    id: 'monthly',
    title: 'Monthly',
    description:
      'Flexible and simple. Superfoods delivered to your door every month — pause or cancel whenever you want.',
    note: 'You can stop anytime.',
  },
  {
    id: 'quarterly',
    title: 'Quarterly',
    description:
      'Three months of curated superfoods in one rhythm. A steady supply shipped to your address with a little extra value.',
    note: 'Billed every three months.',
  },
  {
    id: 'yearly',
    title: 'Yearly',
    description:
      'Our best value for people who want to go all in. A full year of wellness, delivered on schedule to your home.',
    note: 'Billed annually.',
  },
];

export function SubscribePage() {
  return (
    <>
      <section className="page-section page-section--narrow">
        <div className="container legal-page">
          <p className="eyebrow">Become our family</p>
          <h1>Subscribe</h1>
          <p className="section-copy subscribe-intro">
            Join the Salxir family. We send premium superfoods straight to your address — so you can focus on feeling good,
            not reordering. Pick the rhythm that fits your life.
          </p>
        </div>
      </section>

      <section className="page-section page-section--compact page-section--tinted">
        <div className="container">
          <div className="subscribe-grid">
            {plans.map((plan) => (
              <article key={plan.id} className="subscribe-card">
                <h2>{plan.title}</h2>
                <p className="subscribe-card__body">{plan.description}</p>
                <p className="subscribe-card__note">{plan.note}</p>
                <ButtonLink href="/contact" className="subscribe-card__cta">
                  Choose {plan.title}
                </ButtonLink>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="page-section page-section--compact">
        <div className="container">
          <p className="section-copy subscribe-footer-note">
            Questions about shipping or what&apos;s inside each box?{' '}
            <a href="/contact">Contact us</a> — we&apos;re happy to help you find the right plan.
          </p>
        </div>
      </section>
    </>
  );
}
