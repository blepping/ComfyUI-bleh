from time import time
from typing import Any

from aiohttp import web
from server import PromptServer

dum_page = """<!DOCTYPE html>
<html>
<head>
    <title>bleh preview</title>
    <style>
        body { background-color: #303030; margin: 0; }
        img { width: 100%; height: auto; max-height: 100vh; object-fit: contain; }
    </style>
</head>
<body>
    <a id="preview-link" target="_blank" href="/bleh/last_preview.html?no_refresh">
        <img id="preview-img">
    </a>

    <script>
        const urlParams = new URLSearchParams(window.location.search);
        const endpoint = '/bleh/last_preview';
        const default_refresh = 10000;
        let currentBuffer = null;
        let currentObjectURL = null;

        function isDataEqual(buf1, buf2) {
            if (!buf1 || !buf2 || buf1.byteLength !== buf2.byteLength) return false;
            const v1 = new Uint8Array(buf1);
            const v2 = new Uint8Array(buf2);
            for (let i = 0; i < v1.length; i++) {
                if (v1[i] !== v2[i]) return false;
            }
            return true;
        }

        async function checkUpdate() {
            let nextPollMs = null; // Default to 10 seconds
            try {
                const response = await fetch(endpoint, { cache: 'no-store' });
                if (!response.ok) return;
                const durationHeader = response.headers.get('x-bleh-animation-duration');
                if (durationHeader) {
                    const durationMs = parseInt(durationHeader, 10);
                    if (!isNaN(durationMs)) {
                        nextPollMs = Math.max(1000, durationMs);
                    }
                }
                const newBuffer = await response.arrayBuffer();
                if (!isDataEqual(currentBuffer, newBuffer)) {
                    currentBuffer = newBuffer;

                    const contentType = response.headers.get('content-type') || '';
                    const blob = new Blob([newBuffer], { type: contentType });
                    const newUrl = URL.createObjectURL(blob);

                    document.getElementById('preview-img').src = newUrl;

                    currentObjectURL = newUrl;
                }
            } catch (err) {
                console.error("Bleh last preview: fetch failed:", err);
            } finally {
                if (urlParams.has('no_refresh')) return;
                if (nextPollMs === null) { nextPollMs = default_refresh }
                setTimeout(checkUpdate, nextPollMs);
            }
        }
        checkUpdate();
    </script>
</body>
</html>"""


class LastPreview:
    image: bytes | None
    stamp: float | None
    content_type: str | None
    duration: int

    def __init__(self, *, min_refresh: int = 5):
        self.min_refresh = min_refresh
        self.image = None
        self.stamp = None
        self.content_type = None
        self.duration = 10

    def update(
        self,
        *,
        image_bytes: bytes,
        content_type: str,
        stamp: float | None = None,
        duration: int | None = None,
    ):
        self.image = image_bytes
        self.stamp = time() if stamp is None else stamp
        self.content_type = content_type
        self.duration = int(
            max(self.min_refresh, duration if duration is not None else 0)
        )

    async def __call__(self, request: web.Request):
        if request.path.endswith(".html"):
            return web.Response(body=dum_page, content_type="text/html")
        if self.image is None or self.content_type is None:
            raise web.HTTPNotFound(reason="OHNO")
        return web.Response(
            body=self.image,
            content_type=self.content_type,
            headers={"x-bleh-animation-duration": str(int(self.duration * 1000))},
        )


LAST_PREVIEW = None


def init_routes(**kwargs: Any):
    global LAST_PREVIEW  # noqa: PLW0603
    if LAST_PREVIEW is not None:
        return
    LAST_PREVIEW = LastPreview(**kwargs)
    PromptServer.instance.routes.get("/bleh/last_preview")(LAST_PREVIEW)
    PromptServer.instance.routes.get("/bleh/last_preview.html")(LAST_PREVIEW)
