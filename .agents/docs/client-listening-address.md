# Client listening address

## Decision

`python -m client` defaults to `127.0.0.1:42731`. Optional `EVERYGPU_HOST=host:port` changes the bind address and port without changing the command. Bracket IPv6 addresses, for example `[::1]:42731`.

`EVERYGPU_HOST=0.0.0.0:42731 python -m client` binds all IPv4 interfaces. Other devices use the server machine's reachable address, not `0.0.0.0`, to send requests.

The service prints its listening address. Invalid configuration and bind failures are explicit errors; it does not silently choose another port. Configuration parsing and binding live in `client/server/config.py`; startup and cleanup live in `client/server/serve.py`.

## Why

One stable default keeps local startup simple. An environment override follows the configuration approach the user requested after reviewing `ollama serve`, and allows deployment on another interface or port without adding CLI commands.

The port is a project default, not a registered or guaranteed-free port. Bind failures, startup and shutdown behavior remain unverified at runtime.
