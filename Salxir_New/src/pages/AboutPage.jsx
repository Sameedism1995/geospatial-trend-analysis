import { PlaceholderPage } from '../components/PlaceholderPage';

export function AboutPage() {
  return (
    <PlaceholderPage
      eyebrow="About"
      title="A premium natural wellness brand in development"
      description="This route is ready for a fuller brand story, sourcing background, certificates, and trust details without changing the shared layout."
      cta={{ href: '/contact', label: 'Contact Salxir' }}
    />
  );
}
