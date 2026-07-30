"""LLM backends for the System 2 planner.

Provides an ``llm_fn: Callable[[str], str]`` that :class:`planner.Planner` consumes. Wraps the
Qwen API on Alibaba Cloud Model Studio (DashScope), accessed through its OpenAI-compatible
chat-completions endpoint. The API key is read from the environment variable
``DASHSCOPE_API_KEY`` — it is **never** hardcoded or committed. Set it via your shell or a
gitignored ``.env``:

    # Windows (PowerShell):   $env:DASHSCOPE_API_KEY = "sk-..."
    # Linux/macOS:            export DASHSCOPE_API_KEY=sk-...

Docs: https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions
(OpenAI-compatible; base_url https://dashscope.aliyuncs.com/compatible-mode/v1)
"""

import base64
import logging
import os
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["qwen_llm_fn", "QwenError", "DEFAULT_MODEL", "DEFAULT_VISION_MODEL"]

# Beijing region (China mainland) by default; override via DASHSCOPE_BASE_URL for other
# regions, e.g. https://dashscope-intl.aliyuncs.com/compatible-mode/v1 (Singapore/US/Germany).
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_BASE_URL = os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL)
DEFAULT_MODEL = "qwen-plus"                # text reasoning; balanced speed/quality
DEFAULT_VISION_MODEL = "qwen-vl-max"       # vision+reasoning tasks

def _to_image_url(img) -> Optional[str]:
    """Best-effort convert an image into an OpenAI-style image_url string.

    Accepts an already-formed url/data-uri string, or an HWC uint8 numpy array in **RGB** order
    (as produced by the pipeline's frame helpers). Encodes to a base64 PNG data URI.
    Returns None if it cannot encode, so callers can skip it gracefully.
    """
    if isinstance(img, str):
        return img
    try:
        import numpy as np

        arr = np.asarray(img)
        if arr.dtype != np.uint8:
            arr = arr.astype("uint8")
        try:
            import cv2

            # Input is RGB (from _frame_from_image / _to_hwc_uint8_rgb), but cv2.imencode treats
            # its input as BGR — so we MUST swap RGB->BGR before encoding, otherwise the PNG comes
            # out with red and blue channels swapped (red caps would look blue to the VLM).
            bgr = arr[:, :, ::-1] if arr.ndim == 3 and arr.shape[2] == 3 else arr
            ok, buf = cv2.imencode(".png", bgr)
            if not ok:
                return None
            data = buf.tobytes()
        except Exception:  # pragma: no cover - cv2 optional
            from PIL import Image
            import io

            # PIL.Image.fromarray expects RGB — pass the original array unchanged.
            bio = io.BytesIO()
            Image.fromarray(arr).save(bio, format="PNG")
            data = bio.getvalue()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception as exc:  # pragma: no cover - reserved path
        logger.warning("Could not encode planner image (%s); skipping.", exc)
        return None


class QwenError(RuntimeError):
    """Raised when the Qwen (DashScope) API call fails."""


def _get_api_key(explicit: Optional[str] = None) -> str:
    key = explicit or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
    if not key:
        raise QwenError(
            "DASHSCOPE_API_KEY not set. Export it in your shell or a gitignored .env "
            "before using the Qwen planner backend."
        )
    return key


def qwen_llm_fn(
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: int = 60,
    base_url: str = QWEN_BASE_URL,
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    enable_thinking: Optional[bool] = None,
) -> Callable[[str], str]:
    """Return a ``Callable[[str], str]`` that sends a prompt to Qwen and returns the text.

    Args:
        model: Qwen model name (e.g. ``DEFAULT_MODEL`` for text, ``DEFAULT_VISION_MODEL`` for
            vision+reasoning tasks).
        temperature: sampling temperature; 0.0 for deterministic planning.
        max_tokens: response cap.
        timeout: per-request timeout (seconds).
        base_url: OpenAI-compatible API base url.
        api_key: optional explicit key; defaults to env var ``DASHSCOPE_API_KEY``.
        system_prompt: optional system message prepended to every call.
        enable_thinking: Qwen3-specific extended-thinking flag. Pass ``False`` for planning /
            perception tasks to get deterministic JSON output without a reasoning preamble.
            ``None`` leaves the model's default unchanged.
    """
    import requests  # local import so the module loads even if requests is absent

    key = _get_api_key(api_key)
    endpoint = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def _call(prompt: str, images=None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if images:
            # Vision content: image parts are placed BEFORE the text prompt so the model sees
            # the image in context first (recommended by Qwen-VL docs).
            parts: List[dict] = []
            for img in images:
                url = _to_image_url(img)
                if url:
                    parts.append({"type": "image_url", "image_url": {"url": url}})
            parts.append({"type": "text", "text": prompt})
            messages.append({"role": "user", "content": parts})
        else:
            messages.append({"role": "user", "content": prompt})
        payload: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        # Qwen3 extended-thinking control: pass only when explicitly set so the call is
        # compatible with older Qwen models that don't accept this parameter.
        if enable_thinking is not None:
            payload["enable_thinking"] = enable_thinking
        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            # When enable_thinking=True, Qwen3 returns reasoning in <think>...</think> tags
            # before the actual answer. Strip the thinking block so callers always receive clean
            # output (the planner's JSON parsers would otherwise trip on the preamble).
            if enable_thinking and content and "<think>" in content:
                end = content.find("</think>")
                if end != -1:
                    content = content[end + len("</think>"):].strip()
            return content
        except requests.RequestException as exc:
            raise QwenError(f"Qwen request failed: {exc}") from exc
        except (KeyError, IndexError, ValueError) as exc:
            raise QwenError(f"Unexpected Qwen response format: {exc}") from exc

    return _call


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Tiny connectivity check (requires DASHSCOPE_API_KEY + network).
    try:
        fn = qwen_llm_fn(enable_thinking=False)
        print(fn('Reply with ONLY this JSON array: ["ok"]'))
    except QwenError as e:
        print(f"[skip] {e}")
