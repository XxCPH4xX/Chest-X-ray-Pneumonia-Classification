# Deployment preparation

The application runs locally by default. Nothing is deployed automatically.

`deploy/vercel/` contains an optional CPU inference entrypoint and dependency
configuration. It requires explicit deployment hostnames, disables history,
and limits uploads to 3 MiB. The local API retains its localhost restrictions
and 8 MiB limit.

To check the bundle locally, activate the Python environment and make Node.js
available on `PATH`, then run from the repository root:

```bash
python -m scripts.prepare_vercel
```

This builds a static frontend and writes `artifacts/deploy/vercel/`. It includes
the exported model and matching aggregate evaluation reports. Dataset images,
training checkpoints, notebook outputs, and `.env` files are excluded. The
inventory is saved as `artifacts/deploy/inventory.json`.

The hosted frontend says that uploads are sent to a server. Local builds keep
their local-processing wording. Run `npm run build` in `frontend/` afterward
to restore the regular Next.js build used by `npm start`.

The preparation command does not upload files or create a hosting project.
Vercel authentication and hosted runtime verification remain separate steps.
