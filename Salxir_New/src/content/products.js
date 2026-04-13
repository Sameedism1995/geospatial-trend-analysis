export const productPages = [
  { slug: 'shilajit-resin', title: 'Shilajit Resin' },
  { slug: 'shilajit-caps', title: 'Shilajit Caps' },
  { slug: 'shilajit-ashwagandha-caps', title: 'Shilajit + Ashwagandha Caps' },
  { slug: 'shilajit-honey-sticks', title: 'Shilajit + Honey Sticks' },
  { slug: 'shilajit-ashwagandha-honey-sticks', title: 'Shilajit + Ashwagandha Honey Sticks' },
  { slug: 'pink-salt', title: 'Pink Salt' },
  { slug: 'herbal-tea', title: 'Herbal Tea' },
  { slug: 'royal-honey-blends', title: 'Royal Honey Blends' },
  { slug: 'turmeric', title: 'Turmeric' },
];

export const productBySlug = Object.fromEntries(productPages.map((item) => [item.slug, item]));
