"""Bootstrap configuration: --bootstrap beats EVERYGPU_BOOTSTRAP, and addresses are validated."""

import collections.abc
import re

from libp2p.tools.utils import info_from_p2p_addr
from multiaddr import Multiaddr

from client.utils import constants


def resolve(
    argument: list[str] | None, env: collections.abc.Mapping[str, str]
) -> list[str] | None:
    """The validated bootstrap addresses, or None when neither source sets any."""
    values = argument or [
        value
        for value in re.split(r"[\s,]+", env.get(constants.BOOTSTRAP_ENV, ""))
        if value
    ]
    for value in values:
        try:
            info_from_p2p_addr(Multiaddr(value))
        except Exception as error:
            raise ValueError(f"invalid bootstrap address {value!r}: {error}") from error
    return values or None
