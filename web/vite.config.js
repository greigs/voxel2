import { defineConfig } from "vite";

// Hosted at https://greigs.github.io/voxel2/ , so all asset URLs are prefixed with the
// repo name. Override with VITE_BASE if you fork to a differently named repo.
export default defineConfig({
  base: process.env.VITE_BASE || "/voxel2/",
  build: {
    target: "es2020",
    outDir: "dist",
  },
});
