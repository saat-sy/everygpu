"""Network service: a long-lived DHT client; each request discovers, admits a route and generates."""

import trio

from client.generation import assets
from client.generation import loop
from client.generation import models as generation_models
from client.generation import text
from client.network import admission
from client.network import checkpoint
from client.network import discovery
from client.network import models
from client.network import routing
from client.network import runtime as runtime_lib
from client.utils import constants
from client.utils import models as client_models
from worker.discovery import node


class Service:
    """Keeps a DHT client joined through the bootstrap peers; `generate` admits a route per request."""

    def __init__(self, bootstrap: list[str] | None):
        self.bootstrap = bootstrap
        self.runtime = runtime_lib.TrioRuntime()
        self._dht = None
        self._error = "joining the network"
        self._checkpoints: checkpoint.Cache = {}
        self._lookup = trio.Lock()
        self._assets = trio.Lock()
        self._tokenizers: dict[tuple[str, str], generation_models.Tokenizer] = {}

    async def start(self) -> None:
        """Begin joining in the background; never waits on the network."""
        if self.bootstrap:
            self.runtime.start()
            self.runtime.spawn(self._maintain)

    async def stop(self) -> None:
        if self.bootstrap:
            await self.runtime.stop()

    def generate(self, request: client_models.ChatRequest) -> runtime_lib.Stream:
        """Start a generation; its items are text deltas, then one Finish."""
        if not self.bootstrap:
            raise models.RouteError("no bootstrap peer is configured")
        return self.runtime.stream(self._generate, request)

    async def _maintain(self) -> None:
        while True:
            try:
                async with node.open_node(self.bootstrap, client=True) as dht:
                    self._dht = dht
                    while dht.get_routing_table_size():
                        await trio.sleep(constants.RECONNECT_SECONDS)
                    self._error = "lost every peer"
            except Exception as error:
                self._error = f"{type(error).__name__}: {error}"
            finally:
                self._dht = None
            await trio.sleep(constants.RECONNECT_SECONDS)

    async def _generate(self, emit, request: client_models.ChatRequest) -> None:
        pinned, route = await self._route()
        tokenizer = await self._tokenizer(pinned)
        prompt = await trio.to_thread.run_sync(
            text.encode_chat, tokenizer, request.messages, abandon_on_cancel=True
        )
        limit = pinned.config["max_position_embeddings"]
        if len(prompt) >= limit:
            raise client_models.ApiError(
                400,
                f"This model's maximum context length is {limit} tokens, but the "
                f"messages resulted in {len(prompt)} tokens, leaving no room "
                "to generate.",
                param="messages",
                code=constants.CONTEXT_LENGTH_CODE,
            )
        detokenizer = text.Detokenizer(tokenizer)
        async with admission.open_session(route) as session:
            reason, count = await loop.run(
                session,
                prompt,
                min(request.max_tokens, limit - len(prompt)),
                request.temperature,
                tokenizer.eos,
                pinned.config["vocab_size"],
                emit,
                detokenizer,
            )
        if tail := detokenizer.flush():
            await emit(tail)
        await emit(generation_models.Finish(reason, len(prompt), count))

    async def _tokenizer(
        self, pinned: models.Checkpoint
    ) -> generation_models.Tokenizer:
        """The pinned revision's tokenizer; fetched and verified once per pin."""
        key = (pinned.repo, pinned.revision)
        async with self._assets:
            if key not in self._tokenizers:
                self._tokenizers[key] = await trio.to_thread.run_sync(
                    assets.load, pinned, abandon_on_cancel=True
                )
            return self._tokenizers[key]

    async def _route(self) -> tuple[models.Checkpoint, models.Route]:
        dht = self._dht
        if dht is None:
            raise models.RouteError(f"not connected to the network ({self._error})")
        try:
            async with self._lookup:
                with trio.fail_after(constants.DISCOVERY_TIMEOUT_SECONDS):
                    discovery.forget_cached(dht)
                    pinned = await checkpoint.load(dht, self._checkpoints)
                    hops = await discovery.find_hops(dht, pinned)
        except trio.TooSlowError:
            raise models.RouteError("network discovery timed out") from None
        except models.RouteError:
            raise
        except Exception as error:
            raise models.RouteError(
                f"network discovery failed: {type(error).__name__}: {error}"
            ) from None
        return pinned, routing.select(pinned, hops)
