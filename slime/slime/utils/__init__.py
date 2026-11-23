from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .arguments import parse_args

__all__ = ["parse_args"]


def __getattr__(name: str):
    if name == "parse_args":
        from .arguments import parse_args

        return parse_args
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
