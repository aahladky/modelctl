"""python -m modelctl_web -- run the console. Bind via MODELCTL_WEB_BIND
(default 0.0.0.0:9293), token via MODELCTL_WEB_TOKEN or the stored file."""
import os

import uvicorn

from .app import create_app


def main():
    bind = os.environ.get("MODELCTL_WEB_BIND", "0.0.0.0:9293")
    host, _, port = bind.rpartition(":")
    uvicorn.run(create_app(), host=host or "0.0.0.0", port=int(port or 9293),
                log_level="info")


if __name__ == "__main__":
    main()
