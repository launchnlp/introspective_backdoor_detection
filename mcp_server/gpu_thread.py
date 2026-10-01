import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor


class GPUThread:
    """Runs all model work on one dedicated thread, one call at a time.

    vLLM's offline engine and the HF model must not be driven from several
    threads at once, while MCP tool handlers are async and can overlap. Every
    model call (including loading) goes through this single worker thread.
    """

    def __init__(self):
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")

    def submit_sync(self, fn, *args, **kwargs):
        return self._pool.submit(fn, *args, **kwargs).result()

    async def run(self, fn, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, functools.partial(fn, *args, **kwargs))
