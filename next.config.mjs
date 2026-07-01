import createMDX from '@next/mdx';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypePrettyCode from 'rehype-pretty-code';

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  pageExtensions: ['ts', 'tsx', 'js', 'jsx', 'md', 'mdx'],
  experimental: {
    optimizePackageImports: ['lucide-react'],
  },
  redirects: async () => {
    return [
      {
        source: '/knowledge-graph',
        destination: '/knowledge-intelligence',
        permanent: false,
      },
    ]
  },
  headers: async () => {
    return [
      {
        source: '/:path*',
        headers: [
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-XSS-Protection',
            value: '1; mode=block',
          },
          {
            key: 'Strict-Transport-Security',
            value: 'max-age=63072000; includeSubDomains; preload',
          },
          {
            key: 'Referrer-Policy',
            value: 'strict-origin-when-cross-origin',
          },
          {
            key: 'Permissions-Policy',
            value: 'camera=(), microphone=(), geolocation=()',
          },
          {
            key: 'Content-Security-Policy',
            value: [
              "default-src 'self'",
              "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
              "style-src 'self' 'unsafe-inline'",
              // Allow images from self, data URIs, CDNs (Lottie, diagrams)
              "img-src 'self' data: blob: https://assets.lottiefiles.com https://assets9.lottiefiles.com",
              "font-src 'self' https://cdn.jsdelivr.net",
              // API calls to backend + Observable + PostHog
              "connect-src 'self' https://paper2code-1-81y5.onrender.com https://observablehq.com https://us.i.posthog.com",
              // YouTube and Observable iframe embeds
              "frame-src https://www.youtube.com https://www.youtube-nocookie.com https://observablehq.com",
              // This page must not be framed (XFO already set above)
              "frame-ancestors 'none'",
            ].join('; '),
          },
        ],
      },
    ]
  },
};

const withMDX = createMDX({
  extension: /\.mdx?$/,
  options: {
    remarkPlugins: [remarkGfm, remarkMath],
    rehypePlugins: [
      rehypeKatex,
      [rehypePrettyCode, { theme: 'github-dark' }]
    ],
  },
});

export default withMDX(nextConfig);
