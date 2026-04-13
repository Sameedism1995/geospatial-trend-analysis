import { ButtonLink } from './ButtonLink';
import { SectionIntro } from './SectionIntro';

export function PlaceholderPage({ eyebrow, title, description, cta }) {
  return (
    <section className="page-section page-section--narrow">
      <div className="container">
        <SectionIntro eyebrow={eyebrow} title={title} description={description} />
        {cta ? <ButtonLink href={cta.href}>{cta.label}</ButtonLink> : null}
      </div>
    </section>
  );
}
