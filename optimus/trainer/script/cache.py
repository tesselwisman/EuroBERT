import typing as tp


class Cache:
    """Simple cache to store values for a given key.

    The values in the cache should not be mutated,
    as they are shared across all instances.
    """

    def __init__(self):
        self._values = {}

    def maybe_init(
        self,
        key: str,
        init_fn: tp.Callable[..., tp.Any],
        *init_args: tp.Any,
        **init_kwargs: tp.Any,
    ):
        if key not in self._values:
            self._values[key] = init_fn(*init_args, **init_kwargs)

    def get(self, key: str) -> tp.Any:
        return self._values[key]
