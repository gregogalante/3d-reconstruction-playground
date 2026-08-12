# viewer

The browser end of `viewer_server.py`: datasets, their point clouds, the cameras that
made them and the photos localised against them.

React with `@react-three/fiber`, built by Vite into `dist/`, which the server mounts at
`/`. The API it talks to is documented where it is implemented — start at
`viewer_server.py` and `src/api.js`.

```bash
yarn install
yarn dev     # vite on 5173, calling the API on 8000
yarn build   # into dist/, which viewer_server.py serves
```

The other UI in this repo is `capture/`, served by `capture_server.py`, where the
datasets come from in the first place. It is plain ES modules with no build step, because
it has to run on a phone that has just accepted a self signed certificate.
