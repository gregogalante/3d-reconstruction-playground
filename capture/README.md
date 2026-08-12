# capture

The phone end of `capture_server.py`: a guided capture that writes photos and ARCore
poses straight into `storage/datasets/<name>/train/`, ready for `pipeline.py`.

No build step and no dependencies — plain ES modules, served as they are on disk. That is
deliberate: this page is opened on a phone over the local network, usually just after
clicking through a certificate warning, and anything that needed bundling would be one
more thing to go wrong at that moment.

```
index.html      the three screens: setup, the AR heads up display, the summary
app.js          wiring: capabilities, the frame loop, and the two degraded paths
lib/guidance.js when to take a shot and where to send the person next — the only part
                with tests, `node --test 'capture/lib/*.test.js'`
lib/xr.js       the WebXR session: ARCore pose in, camera pixels out
lib/radar.js    the coverage dial and the breadcrumb map
lib/simulate.js a phone that is not there, for `?simulate=1`
lib/api.js      talking to the server, with an upload queue the shutter never waits on
lib/vec.js      the vector maths the guidance is written in
```

Why it works the way it does — the overlap rule, the blur bar that yields to the chain,
what a real capture measured — is in [AGENTS.md](../AGENTS.md).

The other UI in this repo is `viewer/`, served by `viewer_server.py`, which shows what
these captures became.
