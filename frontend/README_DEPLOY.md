Deploying this Vite React app to GitHub Pages

This project is configured to deploy to GitHub Pages using the `gh-pages` package.

Before you start
- This repository is configured to publish to: `https://vpatturi3.github.io/PulmoSimHost`
- `vite.config.ts` base is set to: `'/PulmoSimHost/'`

PowerShell commands (run inside the `frontend` folder)

# Install dependencies (will add gh-pages)
npm install

# Build and publish to the gh-pages branch
npm run deploy

# If npm is not available on your system
# 1) Install Node.js (recommended: download from https://nodejs.org/) which includes npm
# 2) Re-open PowerShell (or restart your terminal) so npm is available

Enable GitHub Pages
1. Push the repository to GitHub (if not already):
   git add .
   git commit -m "Prepare frontend for GitHub Pages"
   git push origin main

2. On GitHub, go to your repository Settings -> Pages.
   Set the source to the `gh-pages` branch and save.

Notes & troubleshooting
- Routing: this app uses BrowserRouter. GitHub Pages does not automatically fallback to `index.html` on direct route navigation. If you see 404s when navigating directly to a sub-route, you have two options:
  1) Use HashRouter instead of BrowserRouter (no server support needed).
  2) Add a `404.html` that redirects to the index page and preserves the path. See: https://github.com/rafgraph/spa-github-pages

- If you prefer automatic deployments from CI, consider adding a GitHub Action that runs `npm ci && npm run build && npx gh-pages -d dist` on push to `main`.

If you want, I can:
- Replace placeholders in files with your GitHub username and repo name if you provide them.
- Add a `404.html` redirect helper automatically.
- Switch the app to use HashRouter to avoid route 404s on GitHub Pages.

