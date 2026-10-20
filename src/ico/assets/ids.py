"""``AssetId``: the type of a parameter or resource field that names an
asset, so that tools find the assets a spec needs::

    @resource("wan21.umt5", 1)
    @dataclass(frozen=True)
    class Umt5(Resource[UMT5EncoderModel]):
        weights: AssetId = "wan21-umt5-xxl"

It is a ``str`` for type checkers and at run time.
"""

from __future__ import annotations

from typing import Annotated, get_args, get_origin


class _AssetMarker:
    def __repr__(self) -> str:
        return "ASSET"


ASSET = _AssetMarker()

AssetId = Annotated[str, ASSET]
"""An asset id (a ``str``)."""


def is_asset_hint(hint: object) -> bool:
    """Whether a resolved annotation (``include_extras=True``) is
    :data:`AssetId`."""
    return get_origin(hint) is Annotated and any(m is ASSET for m in get_args(hint)[1:])


def strip_annotated(hint: object) -> object:
    """The type under ``Annotated[...]``, or ``hint`` itself."""
    return get_args(hint)[0] if get_origin(hint) is Annotated else hint
