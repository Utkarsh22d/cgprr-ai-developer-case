/**
 * Static asset worker for the demo site.
 * All routes are served from the built public/ directory.
 */
export default {
  async fetch(request, env) {
    if (!env || !env.ASSETS) {
      return new Response("Asset binding missing", { status: 500 });
    }
    return env.ASSETS.fetch(request);
  },
};
