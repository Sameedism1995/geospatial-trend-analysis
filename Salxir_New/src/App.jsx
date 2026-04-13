import { Route, Routes } from 'react-router-dom';
import { Layout } from './components/Layout';
import { AboutPage } from './pages/AboutPage';
import { ContactPage } from './pages/ContactPage';
import { GlobalPage } from './pages/GlobalPage';
import { HomePage } from './pages/HomePage';
import { PrivacyPolicyPage } from './pages/PrivacyPolicyPage';
import { ProductDetailPage } from './pages/ProductDetailPage';
import { ProductShopPage } from './pages/ProductShopPage';
import { ResearchPage } from './pages/ResearchPage';
import { ShopPage } from './pages/ShopPage';
import { SubscribePage } from './pages/SubscribePage';
import { TermsPage } from './pages/TermsPage';

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<HomePage />} />
        <Route path="/partner" element={<GlobalPage />} />
        <Route path="/shop" element={<ShopPage />} />
        <Route path="/shop/:slug" element={<ProductShopPage />} />
        <Route path="/products/:slug" element={<ProductDetailPage />} />
        <Route path="/global" element={<GlobalPage />} />
        <Route path="/research" element={<ResearchPage />} />
        <Route path="/about" element={<AboutPage />} />
        <Route path="/subscribe" element={<SubscribePage />} />
        <Route path="/contact" element={<ContactPage />} />
        <Route path="/privacy-policy" element={<PrivacyPolicyPage />} />
        <Route path="/terms-of-service" element={<TermsPage />} />
      </Route>
    </Routes>
  );
}
