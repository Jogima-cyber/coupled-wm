# SPDX-License-Identifier: Apache-2.0
"""YAML configs with ``--set a.b.c=value`` overrides and ``${ENV_VAR}`` interpolation.

* Values are parsed with a YAML 1.2-style float rule, so ``1e-4`` is a float (plain PyYAML
  reads it as a string).
* ``--set a.b.c=value`` replaces an existing key; the value is parsed as YAML
  (``--set opt.lr=3e-4``, ``--set data.dirs=[a,b]``, ``--set x=null``). A path that does not
  exist in the config (after the trainer's ``defaults`` are merged in) raises; prefix it with
  ``+`` to add a new key on purpose (``--set +a.new_key=1``). Integer path segments index
  lists (``env.action_scale.3=0.5``).
* String values may contain ``${VAR}`` or ``${VAR:-default}`` with the shell's meaning:
  ``${VAR:-default}`` takes the default when ``VAR`` is unset OR empty; ``${VAR}`` is the
  value as set (an empty string if set but empty) and raises when ``VAR`` is unset. A string
  that is exactly one ``${...}`` expression is YAML-parsed after substitution, so
  ``num_workers: ${NUM_WORKERS:-4}`` yields an int.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


class _Loader(yaml.SafeLoader):
    pass


_FLOAT_RE = re.compile(
    r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
    |[-+]?\.(?:inf|Inf|INF)
    |\.(?:nan|NaN|NAN))$""",
    re.X,
)
_Loader.yaml_implicit_resolvers = {
    k: [(tag, rx) for tag, rx in v if tag != "tag:yaml.org,2002:float"]
    for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_Loader.add_implicit_resolver("tag:yaml.org,2002:float", _FLOAT_RE, list("-+0123456789."))

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def parse_yaml(text: str) -> Any:
    return yaml.load(text, Loader=_Loader)


def _interpolate_str(value: str) -> Any:
    def repl(m: re.Match) -> str:
        name, default = m.group(1), m.group(2)
        current = os.environ.get(name)
        if default is not None and not current:
            return default
        if current is not None:
            return current
        raise KeyError(f"environment variable ${{{name}}} is not set and has no default (in {value!r})")

    whole = _ENV_RE.fullmatch(value)
    out = _ENV_RE.sub(repl, value)
    if whole is not None:
        return parse_yaml(out) if out.strip() else out
    return out


def interpolate_env(obj: Any) -> Any:
    """Return a copy of ``obj`` with ``${VAR}``/``${VAR:-default}`` substituted in strings."""
    if isinstance(obj, dict):
        return {k: interpolate_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [interpolate_env(v) for v in obj]
    if isinstance(obj, str):
        return _interpolate_str(obj)
    return obj


def _split_path(path: str) -> list[str]:
    parts = path.split(".")
    if not path or any(p == "" for p in parts):
        raise ValueError(f"invalid config key path {path!r}")
    return parts


def set_by_path(cfg: dict, path: str, value: Any, allow_new: bool = False) -> None:
    parts = _split_path(path)
    node: Any = cfg
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        where = ".".join(parts[: i + 1])
        if isinstance(node, list):
            if not (part.lstrip("-").isdigit() and -len(node) <= int(part) < len(node)):
                raise KeyError(f"override {path!r}: {where!r} is not a valid list index")
            if last:
                node[int(part)] = value
            else:
                node = node[int(part)]
        elif isinstance(node, dict):
            if part not in node:
                if not allow_new:
                    raise KeyError(
                        f"override {path!r}: unknown key {where!r} (prefix the override with '+' to add it)"
                    )
                if not last:
                    node[part] = {}
            if last:
                node[part] = value
            else:
                node = node[part]
        else:
            raise KeyError(f"override {path!r}: {'.'.join(parts[:i])!r} is not a mapping or list")


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    """Apply ``key.path=value`` overrides in place and return ``cfg``."""
    for item in overrides or ():
        if "=" not in item:
            raise ValueError(f"override {item!r} must have the form key.path=value")
        key, raw = item.split("=", 1)
        key = key.strip()
        allow_new = key.startswith("+")
        set_by_path(cfg, key.lstrip("+"), parse_yaml(raw) if raw.strip() else "", allow_new=allow_new)
    return cfg


def _copy_tree(obj: Any) -> Any:
    """Copy dicts and lists; leaves are shared (keeps sentinel objects identical)."""
    if isinstance(obj, Mapping):
        return {k: _copy_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_copy_tree(v) for v in obj]
    return obj


def _merge_defaults(cfg: Mapping[str, Any], defaults: Mapping[str, Any]) -> dict:
    """A copy of ``cfg`` with the keys it lacks filled from ``defaults``, recursively.

    Values in ``cfg`` win; keys only in ``cfg`` are kept (the caller validates them). A
    ``null`` block whose default is a mapping takes the default block.
    """
    out = _copy_tree(dict(defaults))
    for key, value in cfg.items():
        default = out.get(key)
        if isinstance(default, Mapping) and (value is None or isinstance(value, Mapping)):
            out[key] = _merge_defaults(value or {}, default)
        else:
            out[key] = _copy_tree(value)
    return out


def load_config(
    path: str | os.PathLike,
    overrides: Iterable[str] | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> dict:
    """Load a YAML config, merge ``defaults`` under it, apply ``overrides``, then interpolate
    environment variables. With a trainer's defaults, ``--set`` reaches every key the
    trainer knows, including those the YAML leaves at their default."""
    text = Path(path).read_text()
    cfg = parse_yaml(text)
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: top level of a config must be a mapping")
    if defaults is not None:
        cfg = _merge_defaults(cfg, defaults)
    apply_overrides(cfg, overrides or [])
    return interpolate_env(cfg)


def save_config(cfg: dict, path: str | os.PathLike) -> None:
    """Write ``cfg`` as YAML (key order preserved, written atomically)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(yaml.safe_dump(_plain(cfg), sort_keys=False, default_flow_style=None))
    os.replace(tmp, path)


def _plain(obj: Any) -> Any:
    """Convert tuples / numpy scalars / Paths so that ``yaml.safe_dump`` accepts the tree."""
    if isinstance(obj, dict):
        return {str(k) if not isinstance(k, (str, int, float, bool)) else k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and callable(obj.item) and getattr(obj, "ndim", None) == 0:
        return obj.item()
    return obj
