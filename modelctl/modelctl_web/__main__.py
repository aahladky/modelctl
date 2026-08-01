"""python -m modelctl_web -- run the console. Bind via MODELCTL_WEB_BIND
(default 0.0.0.0:9293), token via MODELCTL_WEB_TOKEN or the stored file."""
import os

import uvicorn

from .app import create_app
from . import telemetry


class _GracefulServer(uvicorn.Server):
    """Signals open SSE generators before uvicorn starts draining.

    uvicorn's graceful shutdown waits for in-flight responses; an SSE
    tick loop never finishes on its own, so without this a SIGTERM hung
    until SIGKILL. handle_exit runs in signal context -- setting the
    event is all it does."""

    def handle_exit(self, sig, frame):
        telemetry.request_shutdown()
        super().handle_exit(sig, frame)


def main():
    bind = os.environ.get("MODELCTL_WEB_BIND", "0.0.0.0:9293")
    host, _, port = bind.rpartition(":")
    config = uvicorn.Config(create_app(), host=host or "0.0.0.0",
                            port=int(port or 9293), log_level="info")
    _GracefulServer(config).run()


if __name__ == "__main__":
    main()
