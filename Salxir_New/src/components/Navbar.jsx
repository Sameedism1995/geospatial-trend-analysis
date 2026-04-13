import { useEffect, useRef, useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';

const productMenu = [
  { label: 'Shilajit Resin', items: [{ label: 'Shilajit Resin', href: '/products/shilajit-resin' }] },
  {
    label: 'Shilajit Caps',
    items: [
      { label: 'Shilajit Caps', href: '/products/shilajit-caps' },
      { label: 'Shilajit + Ashwagandha Caps', href: '/products/shilajit-ashwagandha-caps' },
    ],
  },
  {
    label: 'Honey Infused Shilajit',
    items: [
      { label: 'Shilajit + Honey Sticks', href: '/products/shilajit-honey-sticks' },
      {
        label: 'Shilajit + Ashwagandha Honey Sticks',
        href: '/products/shilajit-ashwagandha-honey-sticks',
      },
    ],
  },
  { label: 'Pink Salt', items: [{ label: 'Pink Salt', href: '/products/pink-salt' }] },
  { label: 'Herbal Tea', items: [{ label: 'Herbal Tea', href: '/products/herbal-tea' }] },
  {
    label: 'Royal Honey Blends',
    items: [{ label: 'Royal Honey Blends', href: '/products/royal-honey-blends' }],
  },
  { label: 'Turmeric', items: [{ label: 'Turmeric', href: '/products/turmeric' }] },
];

const shopNav = [
  { label: 'Research', href: '/research' },
  { label: 'About us', href: '/about' },
];

const partnerNav = [
  { label: 'Partner', href: '/partner' },
  { label: 'Contact', href: '/contact' },
];

export function Navbar() {
  const location = useLocation();
  const [isProductsOpen, setIsProductsOpen] = useState(false);
  const [activeProductIndex, setActiveProductIndex] = useState(0);
  const dropdownRef = useRef(null);
  const isPartnerMode =
    location.pathname.startsWith('/partner') || location.pathname.startsWith('/global');

  useEffect(() => {
    function handleOutsideClick(event) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsProductsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleOutsideClick);
    return () => document.removeEventListener('mousedown', handleOutsideClick);
  }, []);

  const currentProductGroup = productMenu[activeProductIndex];

  return (
    <header className="navbar">
      <div className="container navbar__inner">
        <NavLink className="brand-mark" to="/" aria-label="Salxir Home">
          <span>Salxir</span>
        </NavLink>
        <nav aria-label="Primary">
          <ul className="nav-list">
            {!isPartnerMode && (
              <li className="nav-dropdown" ref={dropdownRef}>
                <button
                  type="button"
                  className={`nav-list__link nav-list__button${isProductsOpen ? ' is-active' : ''}`}
                  onClick={() => setIsProductsOpen((prev) => !prev)}
                >
                  <span>Products</span>
                  <span className={`nav-list__arrow${isProductsOpen ? ' is-open' : ''}`} aria-hidden="true">
                    ▾
                  </span>
                </button>
                {isProductsOpen && (
                  <div className="products-panel">
                    <div className="products-panel__left">
                      {productMenu.map((group, index) => (
                        <button
                          key={group.label}
                          type="button"
                          className={`products-panel__group${
                            index === activeProductIndex ? ' is-active' : ''
                          }`}
                          onClick={() => setActiveProductIndex(index)}
                        >
                          {group.label}
                        </button>
                      ))}
                    </div>
                    <div className="products-panel__right">
                      <p className="products-panel__heading">{currentProductGroup.label}</p>
                      {currentProductGroup.items.map((item) => (
                        <NavLink
                          key={item.label}
                          to={item.href}
                          className="products-panel__item"
                          onClick={() => setIsProductsOpen(false)}
                        >
                          {item.label}
                        </NavLink>
                      ))}
                    </div>
                  </div>
                )}
              </li>
            )}

            {(isPartnerMode ? partnerNav : shopNav).map((item) => (
              <li key={item.label}>
                <NavLink to={item.href} className="nav-list__link">
                  {item.label}
                </NavLink>
              </li>
            ))}

            <li className="nav-mode-switch">
              <div className="mode-toggle mode-toggle--small" aria-label="Site mode switch">
                <NavLink
                  to="/"
                  end
                  className={({ isActive }) => `mode-toggle__item${isActive ? ' is-active' : ''}`}
                >
                  Shop
                </NavLink>
                <NavLink
                  to="/partner"
                  className={({ isActive }) => `mode-toggle__item${isActive ? ' is-active' : ''}`}
                >
                  Partner
                </NavLink>
              </div>
            </li>
          </ul>
        </nav>
      </div>
    </header>
  );
}
