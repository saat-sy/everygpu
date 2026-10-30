# Client server framework

The client HTTP server uses Starlette with uvicorn, not FastAPI.

Why: the contract requires 400 responses in OpenAI's `error` object shape and no extra routes. FastAPI's pydantic validation returns its own 422 shape and adds `/docs` and `/openapi.json`, so both would need overriding. Two routes with hand-written validation (`client/server/validation.py`) need only Starlette.

Pre-binding the listening socket and passing it to `uvicorn.Server.run(sockets=[...])` gives explicit invalid-port and occupied-port errors instead of uvicorn's own exit.
