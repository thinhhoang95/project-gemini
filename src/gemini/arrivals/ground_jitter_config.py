from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from numpy.random import Generator, default_rng

logger = logging.getLogger(__name__)


def _time_str_to_minutes(token: str) -> int:
    """Convert a HH:MM string into minutes since midnight."""
    token = (token or "").strip()
    if not token:
        raise ValueError("Time specification cannot be empty")
    if ":" not in token:
        hours = int(token)
        minutes = 0
    else:
        hours_str, minutes_str = token.split(":", 1)
        hours = int(hours_str)
        minutes = int(minutes_str)
    if hours == 24 and minutes == 0:
        return 1440
    return ((hours % 24) * 60 + minutes) % 1440


def _derive_lognormal_params(mean_minutes: float, std_minutes: float) -> Tuple[float, float]:
    """Convert arithmetic mean/std to lognormal parameters."""
    mean = max(float(mean_minutes), 1e-3)
    std = max(float(std_minutes), 1e-6)
    variance = std * std
    sigma_sq = math.log(1.0 + (variance / (mean * mean)))
    sigma = math.sqrt(max(sigma_sq, 1e-12))
    mu = math.log(mean) - 0.5 * sigma_sq
    return mu, sigma


def _lognormal_cdf(x: float, log_mu: float, log_sigma: float) -> float:
    """CDF of a lognormal distribution."""
    if x <= 0:
        return 0.0
    sigma = max(log_sigma, 1e-9)
    z = (math.log(x) - log_mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def _gpd_cdf(x: float, shape: float, scale: float) -> float:
    """CDF of the Generalized Pareto Distribution."""
    if x <= 0:
        return 0.0
    scale = max(scale, 1e-9)
    if abs(shape) < 1e-12:
        # Exponential limit.
        return 1.0 - math.exp(-x / scale)
    term = 1.0 + shape * x / scale
    if term <= 0:
        return 1.0
    return 1.0 - term ** (-1.0 / shape)


@dataclass(frozen=True)
class HBSParameters:
    """Hurdle-Bulk-Splice parameters for a time-of-day block."""

    p_hurdle: float
    mean_minutes: float
    std_minutes: float
    threshold_minutes: float
    shift_minutes: float = 0.0
    tail_shape: float = 0.25
    tail_scale: float = 15.0
    log_mu: Optional[float] = None
    log_sigma: Optional[float] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, float]) -> "HBSParameters":
        """Build parameters from a config mapping."""
        if not isinstance(data, Mapping):
            raise TypeError("HBS parameter block must be a mapping")
        p_hurdle = float(data.get("p_hurdle", 0.15))
        mean_minutes = float(data.get("mu", data.get("mean", 8.0)))
        std_minutes = float(data.get("sigma", data.get("std", 4.0)))
        threshold_minutes = float(data.get("threshold", 45.0))
        shift_minutes = float(data.get("shift", data.get("c", 0.0)))
        tail_shape = float(data.get("tail_shape", 0.25))
        tail_scale = float(data.get("tail_scale", data.get("beta", 20.0)))
        log_mu = data.get("log_mu")
        log_sigma = data.get("log_sigma")
        if log_mu is not None:
            log_mu = float(log_mu)
        if log_sigma is not None:
            log_sigma = float(log_sigma)
        return cls(
            p_hurdle=p_hurdle,
            mean_minutes=mean_minutes,
            std_minutes=std_minutes,
            threshold_minutes=threshold_minutes,
            shift_minutes=shift_minutes,
            tail_shape=tail_shape,
            tail_scale=tail_scale,
            log_mu=log_mu,
            log_sigma=log_sigma,
        )

    def lognormal_params(self) -> Tuple[float, float]:
        """Return (mu, sigma) for the underlying lognormal law."""
        if self.log_mu is not None and self.log_sigma is not None:
            return self.log_mu, max(self.log_sigma, 1e-9)
        return _derive_lognormal_params(self.mean_minutes, self.std_minutes)


@dataclass(frozen=True)
class _TimeWindowRule:
    start_minute: int
    end_minute: int
    params: HBSParameters

    def matches(self, minute_of_day: int) -> bool:
        minute = minute_of_day % 1440
        if self.start_minute == self.end_minute:
            return True  # Catch-all bucket.
        if self.start_minute < self.end_minute:
            return self.start_minute <= minute < self.end_minute
        # Wrap-around case (e.g., 22:00-02:00).
        return minute >= self.start_minute or minute < self.end_minute


class GroundJitterDistribution:
    """Continuous-time helper that exposes CDF / interval probabilities."""

    def __init__(self, params: HBSParameters):
        self.params = params
        self.p_hurdle = min(max(params.p_hurdle, 0.0), 1.0)
        self.shift_minutes = max(params.shift_minutes, 0.0)
        self.threshold_minutes = max(params.threshold_minutes, 0.0)
        self.tail_shape = params.tail_shape
        self.tail_scale = max(params.tail_scale, 1e-6)
        self._log_mu, self._log_sigma = params.lognormal_params()
        self._bulk_limit = max(self.threshold_minutes - self.shift_minutes, 0.0)
        self._bulk_limit = float(self._bulk_limit)
        self._bulk_cdf_at_limit = (
            _lognormal_cdf(self._bulk_limit, self._log_mu, self._log_sigma)
            if self._bulk_limit > 0
            else 0.0
        )

    def positive_cdf(self, minutes: float) -> float:
        """CDF of the positive (continuous) part ignoring the hurdle."""
        if minutes <= self.shift_minutes:
            return 0.0
        adj = minutes - self.shift_minutes
        if self._bulk_limit <= 0:
            return min(1.0, _gpd_cdf(adj, self.tail_shape, self.tail_scale))
        if adj <= self._bulk_limit:
            return _lognormal_cdf(adj, self._log_mu, self._log_sigma)
        tail_prob = 1.0 - self._bulk_cdf_at_limit
        if tail_prob <= 0:
            return 1.0
        excess = adj - self._bulk_limit
        gpd_val = _gpd_cdf(excess, self.tail_shape, self.tail_scale)
        blended = self._bulk_cdf_at_limit + tail_prob * gpd_val
        return min(blended, 1.0)

    def cdf(self, minutes: float) -> float:
        """Return P(delta <= minutes)."""
        if minutes < 0:
            return 0.0
        if minutes == 0:
            return 1.0 - self.p_hurdle
        cont = self.p_hurdle * self.positive_cdf(minutes)
        base = 1.0 - self.p_hurdle
        return min(base + cont, 1.0)

    def tail_probability(self, minutes: float) -> float:
        """Return 1 - CDF(minutes)."""
        return max(0.0, 1.0 - self.cdf(minutes))

    def interval_probability(self, start_min: float, end_min: float) -> float:
        """Probability that delta lies in [start_min, end_min)."""
        if end_min <= start_min:
            return 0.0
        if end_min <= 0:
            return 0.0
        prob = 0.0
        if start_min <= 0 < end_min:
            prob += 1.0 - self.p_hurdle
        pos_start = max(start_min, 0.0)
        pos_end = max(end_min, 0.0)
        if pos_end <= pos_start:
            return min(max(prob, 0.0), 1.0)
        delta = self.p_hurdle * max(
            self.positive_cdf(pos_end) - self.positive_cdf(pos_start),
            0.0,
        )
        prob += delta
        return min(max(prob, 0.0), 1.0)

    def sample_minutes(self, rng: Optional[Generator] = None) -> float:
        """Sample a delay (minutes) from the hurdle/bulk/splice distribution."""
        generator = rng or default_rng()
        if generator.random() >= self.p_hurdle:
            return 0.0
        delay = self.shift_minutes
        if self._bulk_limit <= 0 or self._bulk_cdf_at_limit <= 0.0:
            return delay + self._sample_tail_minutes(generator)
        if generator.random() <= self._bulk_cdf_at_limit:
            return delay + self._sample_lognormal_truncated(generator)
        tail_excess = self._sample_tail_minutes(generator)
        return delay + self._bulk_limit + tail_excess

    # --- sampling helpers -----------------------------------------------------
    def _sample_lognormal_truncated(self, rng: Generator) -> float:
        """Rejection sample lognormal variates capped at the splice threshold."""
        if self._bulk_limit <= 0:
            return 0.0
        for _ in range(10000):
            draw = float(rng.lognormal(mean=self._log_mu, sigma=self._log_sigma))
            if draw <= self._bulk_limit:
                return draw
        logger.warning(
            "Falling back to bulk_limit draw for truncated lognormal (limit=%s)", self._bulk_limit
        )
        return self._bulk_limit

    def _sample_tail_minutes(self, rng: Generator) -> float:
        """Inverse-CDF sampling from the GPD tail."""
        u = 1.0 - max(rng.random(), 1e-12)
        scale = self.tail_scale
        shape = self.tail_shape
        if abs(shape) < 1e-12:
            return -scale * math.log(u)
        return scale * ((u ** (-shape)) - 1.0) / shape


class GroundJitterConfig:
    """Container for airport/time-of-day specific HBS parameters."""

    def __init__(
        self,
        default_rules: List[_TimeWindowRule],
        airport_rules: Optional[Dict[str, List[_TimeWindowRule]]] = None,
    ):
        if not default_rules:
            raise ValueError("GroundJitterConfig requires at least one default rule")
        self._default_rules = list(default_rules)
        self._airport_rules = {k.upper(): list(v) for k, v in (airport_rules or {}).items()}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Mapping[str, Mapping[str, float]]]) -> "GroundJitterConfig":
        if not isinstance(mapping, Mapping):
            raise TypeError("GroundJitterConfig expects a mapping at the top level")
        if "default" not in mapping:
            raise ValueError("GroundJitterConfig mapping must contain a 'default' block")
        default_rules = cls._parse_time_block(mapping["default"])
        airport_rules: Dict[str, List[_TimeWindowRule]] = {}
        for airport, block in mapping.items():
            if airport.lower() == "default":
                continue
            airport_rules[airport.upper()] = cls._parse_time_block(block)
        return cls(default_rules=default_rules, airport_rules=airport_rules)

    @classmethod
    def from_json(cls, path: str) -> "GroundJitterConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_mapping(data)

    @staticmethod
    def _parse_time_block(block: Mapping[str, Mapping[str, float]]) -> List[_TimeWindowRule]:
        if not isinstance(block, Mapping):
            raise TypeError("Each airport block must be a mapping of time ranges")
        rules: List[_TimeWindowRule] = []
        for window_spec, param_block in block.items():
            if not isinstance(param_block, Mapping):
                raise TypeError("Parameter block must be a mapping")
            try:
                start_str, end_str = window_spec.split("-", 1)
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid time window specification '{window_spec}'") from exc
            start_min = _time_str_to_minutes(start_str)
            end_min = _time_str_to_minutes(end_str)
            params = HBSParameters.from_mapping(param_block)
            rules.append(_TimeWindowRule(start_minute=start_min, end_minute=end_min, params=params))
        if not rules:
            raise ValueError("At least one time window must be provided")
        # Ensure deterministic ordering for reproducibility.
        rules.sort(key=lambda rule: (rule.start_minute, rule.end_minute))
        return rules

    def get_distribution(self, airport: Optional[str], minute_of_day: int) -> GroundJitterDistribution:
        params = self.get_params(airport, minute_of_day)
        return GroundJitterDistribution(params)

    def get_params(self, airport: Optional[str], minute_of_day: int) -> HBSParameters:
        airport_key = (airport or "").upper()
        rules = self._airport_rules.get(airport_key, self._default_rules)
        minute = minute_of_day % 1440
        for rule in rules:
            if rule.matches(minute):
                return rule.params
        # Fallback to first default rule if nothing matches.
        logger.debug(
            "No ground jitter params for airport=%s minute=%d; using first default block",
            airport_key or "default",
            minute,
        )
        return self._default_rules[0].params

    @property
    def airports(self) -> List[str]:
        return sorted(self._airport_rules.keys())
