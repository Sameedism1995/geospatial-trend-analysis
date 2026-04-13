import { NavLink } from 'react-router-dom';
import {
  SiAmericanexpress,
  SiApplepay,
  SiFacebook,
  SiGooglepay,
  SiInstagram,
  SiKlarna,
  SiMastercard,
  SiPinterest,
  SiTiktok,
  SiVisa,
} from 'react-icons/si';
import { FaLinkedin } from 'react-icons/fa6';
import { MdEmail, MdPhoneIphone } from 'react-icons/md';
import { contactDetails, footerLinks } from '../content/siteContent';

export function Footer() {
  const socialItems = [
    { label: 'Instagram', icon: SiInstagram },
    { label: 'Facebook', icon: SiFacebook },
    { label: 'LinkedIn', icon: FaLinkedin },
    { label: 'Pinterest', icon: SiPinterest },
    { label: 'TikTok', icon: SiTiktok },
    { label: 'Mail', icon: MdEmail },
  ];

  const paymentItems = [
    { label: 'Visa', icon: SiVisa },
    { label: 'Mastercard', icon: SiMastercard },
    { label: 'Amex', icon: SiAmericanexpress },
    { label: 'Klarna', icon: SiKlarna },
    { label: 'MobilePay', icon: MdPhoneIphone },
    { label: 'Apple Pay', icon: SiApplepay },
    { label: 'Google Pay', icon: SiGooglepay },
  ];

  return (
    <footer className="footer">
      <div className="container footer__grid footer__grid--top">
        <div>
          <p className="footer__title">Quick Links</p>
          <div className="footer__links">
            {footerLinks.map((link) => (
              <NavLink key={link.label} to={link.href} className="footer__link">
                {link.label}
              </NavLink>
            ))}
          </div>
        </div>
        <div>
          <p className="footer__title">Contact</p>
          <p className="footer__text">{contactDetails.email}</p>
          <p className="footer__text">{contactDetails.phone}</p>
          <p className="footer__text">{contactDetails.location}</p>
        </div>
        <div>
          <p className="footer__title">Find Us On</p>
          <div className="chip-row">
            {socialItems.map((item) => {
              const Icon = item.icon;
              return (
                <span key={item.label} className="icon-chip" aria-label={item.label} title={item.label}>
                  <Icon aria-hidden="true" />
                </span>
              );
            })}
          </div>
          <p className="footer__title footer__title--spaced">Payment Methods</p>
          <div className="chip-row">
            {paymentItems.map((item) => {
              const Icon = item.icon;
              return (
                <span key={item.label} className="icon-chip" aria-label={item.label} title={item.label}>
                  <Icon aria-hidden="true" />
                </span>
              );
            })}
          </div>
        </div>
      </div>
      <div className="container footer-bottom">
        <p>© 2026 Salxir.com. All rights reserved.</p>
        <div className="footer-bottom__links">
          <NavLink to="/privacy-policy">Privacy Policy</NavLink>
          <NavLink to="/terms-of-service">Terms of Service</NavLink>
        </div>
      </div>
    </footer>
  );
}
