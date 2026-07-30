"""Application services for modelctl.

Service functions receive typed inputs and return typed results.
They do not print, call sys.exit(), or touch the UI directly.
CLI handlers and HTTP routes both call these.

Service functions that are long-running accept a `ctx` parameter
(JobContext from the web layer) for logging, cancellation, and
subprocess registration. When called from the CLI, ctx can be a
simple CLIContext that prints to stderr.
"""
