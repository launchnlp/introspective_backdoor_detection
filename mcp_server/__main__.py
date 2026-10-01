"""Serve the Case I tools for one organism over streamable HTTP.

    cd /home/meghss/introspective_backdoor_detection
    python -m mcp_server --organism flattery --port 8765

Clients connect to http://<host>:<port>/mcp. Uses GPUs AUDITOR_GPU and
ASSISTANT_GPU from agentic_setup/case_i.py.
"""
import argparse

from .gpu_thread import GPUThread
from .server import create_mcp_server
from .tools.double_agent import ALPHA_RANGE


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--organism", required=True, help="a key of MODELS in agentic_setup/case_i.py, e.g. flattery")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--default-alpha", type=float, default=0.5,
                   help="LDA strength when the caller doesn't pass alpha")
    args = p.parse_args()

    if not ALPHA_RANGE[0] <= args.default_alpha <= ALPHA_RANGE[1]:
        p.error(f"--default-alpha must be between {ALPHA_RANGE[0]} and {ALPHA_RANGE[1]}")

    # Heavy imports (torch, vLLM) only when actually serving.
    from .backend import CaseIBackend

    gpu = GPUThread()
    backend = gpu.submit_sync(CaseIBackend, args.organism)
    app = create_mcp_server(backend, gpu, default_alpha=args.default_alpha)
    print(f"Serving {args.organism} on http://{args.host}:{args.port}/mcp", flush=True)
    app.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
