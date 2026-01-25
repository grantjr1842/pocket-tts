"""
Model caching utilities for improved loading performance.
"""

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pocket_tts.models.tts_model import TTSModel

logger = logging.getLogger(__name__)


class ModelCache:
    """Simple in-memory cache for TTS model instances.

    This cache stores loaded model instances in memory to avoid
    repeated loading overhead when the same model is requested
    multiple times within a single Python process.

    Note: This is a process-local cache. Models are not persisted
    across different Python invocations.
    """

    def __init__(self) -> None:
        """Initialize model cache."""
        self._cache = {}
        self._lock = threading.Lock()
        self._enabled = True

    def get(
        self,
        variant: str,
        temp: float,
        lsd_decode_steps: int,
        noise_clamp: float | None,
        eos_threshold: float,
    ):
        """Get model from cache if available.

        Args:
            variant: Model variant
            temp: Temperature parameter
            lsd_decode_steps: LSD decode steps
            noise_clamp: Noise clamp value
            eos_threshold: EOS threshold

        Returns:
            Cached model instance, or None if not in cache
        """
        if not self._enabled:
            return None

        cache_key = self._make_key(
            variant, temp, lsd_decode_steps, noise_clamp, eos_threshold
        )

        with self._lock:
            return self._cache.get(cache_key)

    def put(
        self,
        variant: str,
        temp: float,
        lsd_decode_steps: int,
        noise_clamp: float | None,
        eos_threshold: float,
        model,
    ) -> None:
        """Store model in cache.

        Args:
            variant: Model variant
            temp: Temperature parameter
            lsd_decode_steps: LSD decode steps
            noise_clamp: Noise clamp value
            eos_threshold: EOS threshold
            model: Model instance to cache
        """
        if not self._enabled:
            return

        cache_key = self._make_key(
            variant, temp, lsd_decode_steps, noise_clamp, eos_threshold
        )

        with self._lock:
            # Simple LRU: if cache gets too large, clear it
            if len(self._cache) >= 3:
                logger.info("Model cache full, clearing")
                self._cache.clear()

            self._cache[cache_key] = model
            logger.info(f"Cached model with key: {cache_key}")

    def clear(self) -> None:
        """Clear all cached models."""
        with self._lock:
            self._cache.clear()
            logger.info("Model cache cleared")

    def enable(self) -> None:
        """Enable model caching."""
        self._enabled = True
        logger.info("Model caching enabled")

    def disable(self) -> None:
        """Disable model caching."""
        self._enabled = False
        logger.info("Model caching disabled")

    def is_enabled(self) -> bool:
        """Check if caching is enabled.

        Returns:
            True if caching is enabled
        """
        return self._enabled

    @staticmethod
    def _make_key(
        variant: str,
        temp: float,
        lsd_decode_steps: int,
        noise_clamp: float | None,
        eos_threshold: float,
    ) -> str:
        """Create cache key from model parameters.

        Args:
            variant: Model variant
            temp: Temperature parameter
            lsd_decode_steps: LSD decode steps
            noise_clamp: Noise clamp value
            eos_threshold: EOS threshold

        Returns:
            Cache key string
        """
        clamp_str = f"{noise_clamp:.2f}" if noise_clamp is not None else "None"
        return (
            f"{variant}_{temp:.2f}_{lsd_decode_steps}_{clamp_str}_{eos_threshold:.2f}"
        )


# Global model cache instance
_global_cache = ModelCache()


def get_global_cache() -> ModelCache:
    """Get the global model cache instance.

    Returns:
        Global ModelCache instance
    """
    return _global_cache
