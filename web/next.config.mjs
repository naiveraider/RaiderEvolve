/** @type {import('next').NextConfig} */
const nextConfig = {
  // Standalone output → smaller Docker image (only necessary files copied).
  output: "standalone",

  // Dev rewrites proxy to FastAPI; default timeout covers long SSE runs.
  experimental: {
    proxyTimeout: 900_000,
  },

  async rewrites() {
    // In Docker compose the backend service is reachable via BACKEND_URL.
    // Locally it falls back to 127.0.0.1:8000.
    const backendUrl =
      process.env.BACKEND_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";
    return [
      {
        source: "/api/backend/:path*",
        destination: `${backendUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;
