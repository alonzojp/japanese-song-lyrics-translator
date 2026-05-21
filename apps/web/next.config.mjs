/** @type {import('next').NextConfig} */
const nextConfig = {
  transpilePackages: [
    "@japanese-lyrics/shared",
    "@japanese-lyrics/japanese-processing",
    "@japanese-lyrics/providers",
    "@japanese-lyrics/anki",
    "@japanese-lyrics/alignment",
  ],
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "i.ytimg.com",
        pathname: "/vi/**",
      },
      {
        protocol: "https",
        hostname: "img.youtube.com",
        pathname: "/vi/**",
      },
    ],
  },
};

export default nextConfig;
