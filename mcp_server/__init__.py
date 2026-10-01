"""MCP server exposing Case I (double agent + LDA) and plain chat for one organism.

The backend (torch/vLLM) is only imported by __main__, so importing this package
stays lightweight.
"""
from .gpu_thread import GPUThread
from .server import create_mcp_server

__all__ = ["GPUThread", "create_mcp_server"]
