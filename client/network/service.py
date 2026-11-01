"""Network service: a long-lived DHT client; each request discovers, admits a route and generates."""

import logging
import time

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
from monitoring.logging import logger as log_utils
from worker.discovery import node

logger = logging.getLogger(__name__)


class Service:
    """Keeps a DHT client joined through the bootstrap peers; `generate` admits a route per request."""

    def __init__(self, bootstrap: list[str] | None):
        self.bootstrap = bootstrap
        self.runtime = runtime_lib.TrioRuntime()
        self._dht = None
        self._error = "joining the network"
        self._reported = ""
        self._checkpoints: checkpoint.Cache = {}
        self._lookup = trio.Lock()
        self._assets = trio.Lock()
        self._tokenizers: dict[tuple[str, str], generation_models.Tokenizer] = {}

    async def start(self) -> None:
        """Begin joining in the background; never waits on the network."""
        if self.bootstrap:
            logger.info(
                "network service starting bootstrap_peers=%d", len(self.bootstrap)
            )
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
                    self._reported = ""
                    while dht.get_routing_table_size():
                        await trio.sleep(constants.RECONNECT_SECONDS)
                    self._degrade("lost every peer")
            except Exception as error:
                self._degrade(f"{type(error).__name__}: {error}", error)
            finally:
                self._dht = None
            await trio.sleep(constants.RECONNECT_SECONDS)

    def _degrade(self, reason: str, error: Exception | None = None) -> None:
        """Record why the network is unavailable, logging each new reason once."""
        self._error = reason
        if reason != self._reported:
            logger.warning("network unavailable: %s", reason, exc_info=error)
            self._reported = reason

    async def _generate(self, emit, request: client_models.ChatRequest) -> None:
        started = time.monotonic()
        pinned, route = await self._route()
        tokenizer = await self._tokenizer(pinned)
        prompt = await trio.to_thread.run_sync(
            text.encode_chat, tokenizer, request.messages, abandon_on_cancel=True
        )
        logger.debug("prompt encoded prompt_tokens=%d", len(prompt))
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
        logger.info(
            "generation finished reason=%s prompt_tokens=%d completion_tokens=%d "
            "elapsed_ms=%d",
            reason,
            len(prompt),
            count,
            log_utils.elapsed_ms(started),
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
                started = time.monotonic()
                logger.info("tokenizer loading revision=%s", pinned.revision)
                self._tokenizers[key] = await trio.to_thread.run_sync(
                    assets.load, pinned, abandon_on_cancel=True
                )
                logger.info(
                    "tokenizer loaded revision=%s elapsed_ms=%d",
                    pinned.revision,
                    log_utils.elapsed_ms(started),
                )
            return self._tokenizers[key]

    async def _route(self) -> tuple[models.Checkpoint, models.Route]:
        dht = self._dht
        if dht is None:
            raise models.RouteError(f"not connected to the network ({self._error})")
        phase = "checkpoint lookup"
        started = time.monotonic()
        logger.info("route discovery started")
        try:
            async with self._lookup:
                with trio.fail_after(constants.DISCOVERY_TIMEOUT_SECONDS):
                    discovery.forget_cached(dht)
                    pinned = await checkpoint.load(dht, self._checkpoints)
                    phase = "worker provider lookup"
                    hops = await discovery.find_hops(dht, pinned)
        except trio.TooSlowError:
            raise models.RouteError(
                f"network discovery timed out during {phase} "
                f"after {constants.DISCOVERY_TIMEOUT_SECONDS}s"
            ) from None
        except models.RouteError:
            raise
        except Exception as error:
            raise models.RouteError(
                f"network discovery failed: {type(error).__name__}: {error}"
            ) from error
        route = routing.select(pinned, hops)
        logger.info(
            "route selected hops=%d workers=%d elapsed_ms=%d",
            len(route.hops),
            len(hops),
            log_utils.elapsed_ms(started),
        )
        for hop in route.hops:
            logger.debug(
                "route hop peer=%s layers=%d-%d embedding=%s final=%s endpoint=%s",
                hop.peer,
                hop.stage.first,
                hop.stage.stop,
                hop.stage.embedding,
                hop.stage.final,
                hop.endpoint,
            )
        return pinned, route
