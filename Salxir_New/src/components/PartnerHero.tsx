import { useNavigate } from 'react-router-dom';
import { AuroraHero } from '@/components/ui/aurora-hero-bg';
import { globalContent } from '@/content/siteContent';

export function PartnerHero() {
  const navigate = useNavigate();
  const { hero } = globalContent;

  return (
    <AuroraHero
      className="pt-24 pb-12 sm:pt-28"
      title={hero.title}
      description={hero.description}
      primaryAction={{
        label: hero.primaryCta.label,
        onClick: () => navigate(hero.primaryCta.href),
      }}
      secondaryAction={{
        label: hero.secondaryCta.label,
        onClick: () => {
          document.getElementById('partner-services')?.scrollIntoView({ behavior: 'smooth' });
        },
      }}
    />
  );
}
