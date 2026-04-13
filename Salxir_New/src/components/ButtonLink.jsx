import { Link } from 'react-router-dom';

export function ButtonLink({ href, children, variant = 'primary' }) {
  return (
    <Link to={href} className={`button button--${variant}`}>
      {children}
    </Link>
  );
}
