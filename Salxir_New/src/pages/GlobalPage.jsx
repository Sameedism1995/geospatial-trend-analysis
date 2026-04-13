import PartnerSectionBackground from '../components/ui/background-components';
import { PartnerHero } from '../components/PartnerHero';
import { SectionIntro } from '../components/SectionIntro';
import { ServiceCard } from '../components/ServiceCard';
import { StatCard } from '../components/StatCard';
import { globalContent } from '../content/siteContent';

export function GlobalPage() {
  return (
    <>
      <PartnerHero />

      <div className="partner-page-body">
        <div className="partner-page-body__shader" aria-hidden="true">
          <PartnerSectionBackground />
        </div>
        <div className="partner-page-body__content">
          <section className="page-section page-section--compact">
            <div className="container">
              <div className="stats-grid">
                {globalContent.orderStats.map((item) => (
                  <StatCard key={item.label} value={item.value} label={item.label} />
                ))}
              </div>
            </div>
          </section>

          <section className="page-section">
            <div className="container split-section">
              <SectionIntro
                eyebrow="Presence"
                title="Countries reached"
                description="Update this grid as new markets, importers, and distributors are added."
              />
              <div className="country-grid">
                {globalContent.countries.map((country) => (
                  <div key={country} className="country-pill">
                    {country}
                  </div>
                ))}
              </div>
            </div>
          </section>

          <section className="page-section" id="partner-services">
            <div className="container">
              <SectionIntro
                eyebrow="Services"
                title="Supply services"
                description="A restrained overview of the core B2B offer."
              />
              <div className="services-grid">
                {globalContent.services.map((service) => (
                  <ServiceCard key={service.title} {...service} />
                ))}
              </div>
            </div>
          </section>

          <section className="page-section">
            <div className="container">
              <SectionIntro
                eyebrow="Partners"
                title="Partner placeholders"
                description="Text-based placeholders keep the first version clean while partner assets are being finalized."
              />
              <div className="partners-grid">
                {globalContent.partners.map((partner) => (
                  <div key={partner} className="partner-block">
                    {partner}
                  </div>
                ))}
              </div>
            </div>
          </section>
        </div>
      </div>
    </>
  );
}
