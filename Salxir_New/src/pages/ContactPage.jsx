import { PlaceholderPage } from '../components/PlaceholderPage';

export function ContactPage() {
  return (
    <PlaceholderPage
      eyebrow="Contact"
      title="Start a conversation"
      description="Use this page for retail questions, wholesale inquiries, private label opportunities, and distribution partnerships."
      cta={{ href: '/partner', label: 'View Partner Page' }}
    />
  );
}
