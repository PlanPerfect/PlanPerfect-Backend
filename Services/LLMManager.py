from typing import Optional, Dict, List, Any
from groq import Groq
from google import genai
from google.genai import types
import os
import re
import time
import httpx
import json
import ast
import base64
from datetime import datetime, timedelta
from Services import Logger
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

"""
    LLMManager is a service which serves as a high-level wrapper for all LLM calls, rate-limit management and smart model selection.
    It supports both Groq and Gemini LLMs, automatically switching between models based on rate-limit status.
    It handles rate-limit parsing, cooldown tracking, logging, and provides a unified chat pipeline which integrates with RAGManager.

    NOW INCLUDES: Agent-specific models with tool calling support for autonomous agents.
"""

class RateLimitCapturingClient(httpx.Client): # custom HTTPX client to capture x-rate-limit response headers
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_response_headers = {}

    def send(self, request, *args, **kwargs):
        response = super().send(request, *args, **kwargs)
        self.last_response_headers = dict(response.headers)
        return response

class _RateLimitManager: # manages rate limit info for a single model
    def __init__(self):
        self.requests_remaining: Optional[int] = None
        self.tokens_remaining: Optional[int] = None
        self.requests_reset_time: Optional[datetime] = None
        self.tokens_reset_time: Optional[datetime] = None
        self.requests_limit: Optional[int] = None
        self.tokens_limit: Optional[int] = None

    def update_headers(self, headers: Dict[str, str]):
        try:
            if 'x-ratelimit-remaining-requests' in headers:
                self.requests_remaining = int(headers['x-ratelimit-remaining-requests'])

            if 'x-ratelimit-remaining-tokens' in headers:
                self.tokens_remaining = int(headers['x-ratelimit-remaining-tokens'])

            if 'x-ratelimit-limit-requests' in headers:
                self.requests_limit = int(headers['x-ratelimit-limit-requests'])

            if 'x-ratelimit-limit-tokens' in headers:
                self.tokens_limit = int(headers['x-ratelimit-limit-tokens'])

            if 'x-ratelimit-reset-requests' in headers:
                self.requests_reset_time = self._parse_reset_time(
                    headers['x-ratelimit-reset-requests']
                )

            if 'x-ratelimit-reset-tokens' in headers:
                self.tokens_reset_time = self._parse_reset_time(
                    headers['x-ratelimit-reset-tokens']
                )
        except Exception as e:
            Logger.log(f"[LLM MANAGER] - Error parsing rate limit headers: {str(e)}")

    @staticmethod
    def _parse_retry_after(retry_after: Optional[str]) -> Optional[float]: # parse Retry-After header to seconds
        if not retry_after:
            return None
        s = str(retry_after).strip().lower()
        try:
            if re.match(r'^\d+(\.\d+)?$', s):
                return float(s)

            m = re.match(r'^(?:(\d+)(?:m))?(?:(\d+(\.\d+)?)(?:s))?$', s)
            if m:
                mins = int(m.group(1)) if m.group(1) else 0
                secs = float(m.group(2)) if m.group(2) else 0.0
                return mins * 60 + secs

            num = re.findall(r'(\d+(\.\d+)?)', s)
            if num:
                return float(num[0][0])
        except Exception:
            pass
        return None

    def _parse_reset_time(self, reset_str: str) -> datetime: # parse reset time string to datetime
        try:
            s = str(reset_str).strip().lower()

            m = re.match(r'^(\d+(\.\d+)?)(s)?$', s)
            if m:
                seconds = float(m.group(1))
                return datetime.now() + timedelta(seconds=seconds)

            m = re.match(r'^(?:(\d+)(?:m))?(?:(\d+(\.\d+)?)(?:s))?$', s)
            if m:
                mins = int(m.group(1)) if m.group(1) else 0
                secs = float(m.group(2)) if m.group(2) else 0.0
                total = mins * 60 + secs
                return datetime.now() + timedelta(seconds=total)

            digits = re.findall(r'(\d+(\.\d+)?)', s)
            if digits:
                total = float(digits[0][0])
                return datetime.now() + timedelta(seconds=total)

            return datetime.now() + timedelta(seconds=60)
        except Exception as e:
            Logger.log(f"[LLM MANAGER] - Error parsing reset time '{reset_str}': {str(e)}")
            return datetime.now() + timedelta(seconds=60)

    def get_wait_time(self) -> float: # get wait time in seconds until rate limits reset
        now = datetime.now()
        wait_times = []

        if self.tokens_reset_time:
            dt = (self.tokens_reset_time - now).total_seconds()
            if dt > 0:
                wait_times.append(dt)

        if self.requests_reset_time:
            dt = (self.requests_reset_time - now).total_seconds()
            if dt > 0:
                wait_times.append(dt)

        if wait_times:
            return float(min(wait_times)) + 1.0

        return 60.0

class _ModelManager: # manages available models and their rate-limit status. Backbone of the smart model-switching mechanism.
    CHAT_MODELS = [
        {"name": "llama-3.3-70b-versatile", "provider": "groq"},      # 30 RPM, 1K RPD, 12K TPM, 100K TPD
        {"name": "llama-3.1-8b-instant", "provider": "groq"},         # 30 RPM, 14.4K RPD, 6K TPM, 500K TPD
        {"name": "gemini-3-flash", "provider": "gemini"},             # 1K RPM, 1M TPM, 10K RPD
        {"name": "gemini-2.5-flash", "provider": "gemini"},           # 1K RPM, 1M TPM, 10K RPD
        {"name": "gemini-2.5-flash-lite", "provider": "gemini"}       # 4K RPM, 4M TPM, UNLIMITED RPD
    ]

    AGENT_MODELS = [
        {"name": "llama-3.3-70b-versatile", "provider": "groq"},                                # 30 RPM, 1K RPD, 12K TPM, 100K TPD
        {"name": "llama-3.1-8b-instant", "provider": "groq"},                                   # 30 RPM, 14.4K RPD, 6K TPM, 500K TPD
        {"name": "meta-llama/llama-4-maverick-17b-128e-instruct", "provider": "groq"},          # 30 RPM, 1K RPD, 6K TPM, 500K TPD
        {"name": "meta-llama/llama-4-scout-17b-16e-instruct", "provider": "groq"},              # 30 RPM, 1K RPD, 30K TPM, 500K TPD
        {"name": "openai/gpt-oss-20b", "provider": "groq"},                                     # 30 RPM, 1K RPD, 8K TPM, 200K TPD
        {"name": "openai/gpt-oss-120b", "provider": "groq"},                                    # 30 RPM, 1K RPD, 8K TPM, 200K TPD
        {"name": "qwen/qwen3-32b", "provider": "groq"},                                         # 60 RPM, 1K RPD, 6K TPM, 500K TPD
        {"name": "gemini-3-pro-preview", "provider": "gemini"},                                 # 25 RPM, 250 RPD, 1M TPM
        {"name": "gemini-3-flash-preview", "provider": "gemini"},                               # 1K RPM, 10K RPD, 1M TPM
        {"name": "gemini-2.5-pro", "provider": "gemini"},                                       # 150 RPM, 1K RPD, 2M TPM
        {"name": "gemini-2.5-flash", "provider": "gemini"},                                     # 1K RPM, 10K RPD, 1M TPM
    ]

    VISION_MODELS = [
        {"name": "meta-llama/llama-4-scout-17b-16e-instruct", "provider": "groq"},
        {"name": "meta-llama/llama-4-maverick-17b-128e-instruct", "provider": "groq"},
        {"name": "gemini-2.5-flash", "provider": "gemini"},
    ]

    def __init__(self, use_agent_models: bool = False, use_vision_models: bool = False):
        self.use_agent_models = use_agent_models
        self.use_vision_models = use_vision_models
        if use_vision_models:
            self.models = self.VISION_MODELS
        elif use_agent_models:
            self.models = self.AGENT_MODELS
        else:
            self.models = self.CHAT_MODELS
        self.current_model_index = 0
        self.model_rate_limits: Dict[str, _RateLimitManager] = {
            model["name"]: _RateLimitManager() for model in self.models
        }
        self.rate_limit_cooldowns: Dict[str, datetime] = {}

    def get_current_model(self) -> Dict: # get the current available model, switch if rate-limited
        original_index = self.current_model_index
        attempts = 0

        while attempts < len(self.models):
            model = self.models[self.current_model_index]

            if self._model_available(model["name"]):
                return model

            self.current_model_index = (self.current_model_index + 1) % len(self.models)
            attempts += 1

            if self.current_model_index == original_index:
                return self._get_model_with_shortest_cd()

        return self.models[self.current_model_index]

    def _model_available(self, model_name: str) -> bool: # check if model is not currently rate-limited
        if model_name not in self.rate_limit_cooldowns:
            return True

        now = datetime.now()
        if now >= self.rate_limit_cooldowns[model_name]:
            del self.rate_limit_cooldowns[model_name]
            return True

        return False

    def _get_model_with_shortest_cd(self) -> Dict: # get model with shortest cooldown time
        if not self.rate_limit_cooldowns:
            return self.models[0]

        now = datetime.now()
        min_cooldown_model_name = min(
            self.rate_limit_cooldowns.items(),
            key=lambda x: (x[1] - now).total_seconds()
        )[0]

        for model in self.models:
            if model["name"] == min_cooldown_model_name:
                return model

        return self.models[0]

    def update_rate_limit_info(self, model_name: str, headers: Dict[str, str]): # update for tracking and logging
        if model_name in self.model_rate_limits:
            self.model_rate_limits[model_name].update_headers(headers)

    def mark_rate_limited(self, model_name: str, retry_after: Optional[str] = None): # mark model as rate-limited and set cooldown
        now = datetime.now()

        cooldown_time = None
        if retry_after:
            secs = _RateLimitManager._parse_retry_after(retry_after)
            if secs:
                cooldown_time = now + timedelta(seconds=secs + 2)

        if not cooldown_time and model_name in self.model_rate_limits:
            wait_secs = self.model_rate_limits[model_name].get_wait_time()
            if wait_secs and wait_secs > 0:
                cooldown_time = now + timedelta(seconds=wait_secs)

        if not cooldown_time:
            cooldown_time = now + timedelta(seconds=60)

        self.rate_limit_cooldowns[model_name] = cooldown_time

        self.current_model_index = (self.current_model_index + 1) % len(self.models)

    def mark_model_unavailable(self, model_name: str, cooldown_seconds: int = 86400):
        """
        Mark a model as unavailable (decommissioned/unsupported) and skip it for a long cooldown.
        Reuses cooldown storage so model selection can avoid it.
        """
        self.rate_limit_cooldowns[model_name] = datetime.now() + timedelta(seconds=cooldown_seconds)
        self.current_model_index = (self.current_model_index + 1) % len(self.models)

    def all_models_rate_limited(self) -> bool: # DANGER: all models rate-limited
        return len(self.rate_limit_cooldowns) >= len(self.models)

class LLMManagerClass: # singleton class managing LLM calls, rate-limits, and model-switching
    _instance = None
    FALLBACK_MESSAGE = "I apologize, but I'm currently experiencing high traffic and all available models are rate-limited. Please try again in a moment."

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._http_client = RateLimitCapturingClient()
            cls._instance._initialized = False
            cls._instance._groq_client = None
            cls._instance._gemini_client = None
            cls._instance._model_manager = None
            cls._instance._agent_model_manager = None  # Separate manager for agent models
            cls._instance._vision_model_manager = None  # Separate manager for vision analysis
        return cls._instance

    def initialize(self):
        if self._initialized:
            return

        self._groq_client = Groq(api_key=GROQ_API_KEY, http_client=self._http_client)
        self._gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        self._model_manager = _ModelManager(use_agent_models=False)  # For regular chat
        self._agent_model_manager = _ModelManager(use_agent_models=True)  # For agent chat with tools
        self._vision_model_manager = _ModelManager(use_vision_models=True)  # For image analysis
        self._initialized = True

    @staticmethod
    def _is_rate_limit_error(error_str: str) -> bool:
        return (
            "rate_limit" in error_str
            or "429" in error_str
            or "too many requests" in error_str
            or "quota" in error_str
            or "resource_exhausted" in error_str
        )

    @staticmethod
    def _is_tool_use_failed_error(error_str: str) -> bool:
        return (
            "tool_use_failed" in error_str
            or "failed to call a function" in error_str
            or "failed_generation" in error_str
        )

    @staticmethod
    def _is_model_unavailable_error(error_str: str) -> bool:
        return (
            "model_decommissioned" in error_str
            or "decommissioned" in error_str
            or "no longer supported" in error_str
            or "model_not_found" in error_str
            or "does not exist" in error_str
            or "not found" in error_str and "model" in error_str
        )

    @staticmethod
    def _is_vision_not_supported_error(error_str: str) -> bool:
        return (
            "does not support image" in error_str
            or "image_url is not supported" in error_str
            or "multimodal is not supported" in error_str
            or "vision is not supported" in error_str
            or "unsupported image" in error_str
            or "unsupported content type" in error_str
        )

    def chat(self, prompt: str) -> str: # main chat method with smart model switching and rate-limit handling
        if not self._initialized:
            raise RuntimeError("LLMManager not initialized. Call initialize() first.")

        if prompt is None:
            return self.FALLBACK_MESSAGE

        temperature = 0.8
        max_tokens = 1024
        max_retries = 8
        attempts = 0

        while attempts < max_retries:
            if self._model_manager.all_models_rate_limited():
                self._log_all_models_exhausted()
                return self.FALLBACK_MESSAGE

            current_model = self._model_manager.get_current_model()
            model_name = current_model["name"]
            provider = current_model["provider"]

            try:
                if provider == "groq":
                    assistant_response = self._call_groq(model_name, prompt, temperature, max_tokens)
                else:
                    assistant_response = self._call_gemini(model_name, prompt, temperature, max_tokens)

                self._log_success(model_name, provider)

                return assistant_response

            except Exception as e:
                error_str = str(e).lower()

                if self._is_rate_limit_error(error_str):
                    retry_after = None
                    if hasattr(e, 'response') and hasattr(e.response, 'headers'):
                        retry_after = e.response.headers.get('retry-after')

                    self._model_manager.mark_rate_limited(model_name, retry_after)
                    self._log_rate_limit_hit(model_name, provider, str(e))

                    attempts += 1

                    if attempts < max_retries:
                        time.sleep(0.5)
                        continue
                    else:
                        self._log_all_models_exhausted()
                        return self.FALLBACK_MESSAGE
                elif self._is_model_unavailable_error(error_str):
                    self._model_manager.mark_model_unavailable(model_name)
                    attempts += 1
                    if attempts < max_retries:
                        continue
                    return self.FALLBACK_MESSAGE
                else:
                    Logger.log(f"[LLM MANAGER] - Error with {provider}/{model_name}: {str(e)}")
                    raise

        self._log_all_models_exhausted()
        return self.FALLBACK_MESSAGE

    def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """
        Agent-specific chat method with tool calling support.
        Uses Groq models with function calling capabilities.

        Args:
            messages: Conversation history in OpenAI format
            tools: List of tool definitions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Dict containing the response with potential function calls
        """
        if not self._initialized:
            raise RuntimeError("LLMManager not initialized. Call initialize() first.")

        max_retries = 8
        attempts = 0

        while attempts < max_retries:
            if self._agent_model_manager.all_models_rate_limited():
                self._log_all_models_exhausted(is_agent=True)
                raise Exception("All agent models are rate-limited. Please try again later.")

            current_model = self._agent_model_manager.get_current_model()
            model_name = current_model["name"]
            provider = current_model["provider"]

            try:
                if provider == "groq":
                    raw_response = self._call_groq_with_tools(
                        model_name, messages, tools, temperature, max_tokens
                    )
                else:
                    raw_response = self._call_gemini_with_tools(
                        model_name, messages, tools, temperature, max_tokens
                    )

                response = self._normalize_tool_response_payload(
                    response=raw_response,
                    tools=tools,
                )
                if (
                    isinstance(raw_response, dict)
                    and not raw_response.get("tool_calls")
                    and isinstance(response, dict)
                    and response.get("tool_calls")
                ):
                    Logger.log(
                        f"[LLM MANAGER] - WARNING: Parsed textual tool-call fallback for {provider}/{model_name}."
                    )

                self._log_success(model_name, provider)

                return response

            except Exception as e:
                error_str = str(e).lower()

                if self._is_rate_limit_error(error_str):
                    retry_after = None
                    if hasattr(e, 'response') and hasattr(e.response, 'headers'):
                        retry_after = e.response.headers.get('retry-after')

                    self._agent_model_manager.mark_rate_limited(model_name, retry_after)
                    self._log_rate_limit_hit(model_name, provider, str(e))

                    attempts += 1

                    if attempts < max_retries:
                        time.sleep(0.5)
                        continue
                    else:
                        self._log_all_models_exhausted(is_agent=True)
                        raise Exception("All agent models are rate-limited. Please try again later.")
                elif self._is_model_unavailable_error(error_str):
                    self._agent_model_manager.mark_model_unavailable(model_name)
                    attempts += 1
                    if attempts < max_retries:
                        continue
                    raise Exception("All configured agent models are unavailable. Please update model list.")
                elif self._is_tool_use_failed_error(error_str):
                    attempts += 1
                    self._agent_model_manager.current_model_index = (
                        self._agent_model_manager.current_model_index + 1
                    ) % len(self._agent_model_manager.models)

                    if attempts < max_retries:
                        time.sleep(0.25)
                        continue
                    else:
                        raise Exception(
                            "Tool calling failed across available agent models. "
                            "Please retry your request."
                        )
                else:
                    Logger.log(f"[LLM MANAGER] - Error with agent {provider}/{model_name}: {str(e)}")
                    raise

        self._log_all_models_exhausted(is_agent=True)
        raise Exception("All agent models are rate-limited. Please try again later.")

    def _normalize_tool_response_payload(
        self,
        response: Any,
        tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not isinstance(response, dict):
            return {
                "content": str(response or ""),
                "tool_calls": [],
                "finish_reason": "stop",
            }

        normalized: Dict[str, Any] = {
            "content": response.get("content"),
            "tool_calls": response.get("tool_calls") if isinstance(response.get("tool_calls"), list) else [],
            "finish_reason": response.get("finish_reason", "stop"),
        }

        if normalized["tool_calls"]:
            return normalized

        textual_calls = self._extract_textual_tool_calls(
            content=normalized.get("content"),
            tools=tools,
        )
        if textual_calls:
            normalized["tool_calls"] = textual_calls
            normalized["content"] = None
            normalized["finish_reason"] = "tool_calls"

        return normalized

    def _extract_textual_tool_calls(
        self,
        content: Any,
        tools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not isinstance(content, str):
            return []

        text = content.strip()
        if not text:
            return []

        allowed_name_map: Dict[str, str] = {}
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") != "function":
                continue
            function_payload = tool.get("function")
            if not isinstance(function_payload, dict):
                continue
            tool_name = function_payload.get("name")
            if not isinstance(tool_name, str) or not tool_name.strip():
                continue
            allowed_name_map[tool_name.strip().lower()] = tool_name.strip()

        if not allowed_name_map:
            return []

        calls: List[Dict[str, Any]] = []

        def _append_call(candidate_name: str, candidate_args: str) -> None:
            lower_name = str(candidate_name or "").strip().lower()
            normalized_name = allowed_name_map.get(lower_name)
            if not normalized_name:
                return

            normalized_args = self._normalize_tool_call_arguments(candidate_args)
            if not normalized_args:
                return

            call_id = f"text_call_{normalized_name}_{len(calls) + 1}"
            calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": normalized_name,
                        "arguments": normalized_args,
                    },
                }
            )

        markup_patterns = [
            r"<\s*function\s*=\s*([a-zA-Z_][\w\-]*)\s*>\s*(\{.*?\})\s*</\s*function\s*>",
            r"<\s*([a-zA-Z_][\w\-]*)\s*>\s*(\{.*?\})\s*</\s*function\s*>",
            r"<\s*function_call\s+name\s*=\s*[\"']?([a-zA-Z_][\w\-]*)[\"']?\s*>\s*(\{.*?\})\s*</\s*function_call\s*>",
        ]
        for pattern in markup_patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
                _append_call(match.group(1), match.group(2))

        if calls:
            return calls

        for lower_name, original_name in allowed_name_map.items():
            call_pattern = re.compile(
                rf"^\s*{re.escape(lower_name)}\s*\(\s*(\{{.*\}})\s*\)\s*$",
                flags=re.IGNORECASE | re.DOTALL,
            )
            match = call_pattern.match(text)
            if not match:
                continue
            _append_call(original_name, match.group(1))
            if calls:
                return calls

        return calls

    def _normalize_tool_call_arguments(self, raw_arguments: Any) -> Optional[str]:
        text = str(raw_arguments or "").strip()
        if not text:
            return None

        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text).strip()

        candidates = [text]
        start_index = text.find("{")
        end_index = text.rfind("}")
        if start_index >= 0 and end_index > start_index:
            candidates.append(text[start_index : end_index + 1])

        for candidate in candidates:
            parsed = None
            try:
                parsed = json.loads(candidate)
            except Exception:
                try:
                    parsed = ast.literal_eval(candidate)
                except Exception:
                    parsed = None

            if isinstance(parsed, dict):
                return json.dumps(parsed, ensure_ascii=False)

        return None

    def chat_with_vision(
        self,
        prompt: str,
        image_bytes: bytes,
        mime_type: str = "image/png",
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> str:
        if not self._initialized:
            raise RuntimeError("LLMManager not initialized. Call initialize() first.")

        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            raise ValueError("chat_with_vision requires non-empty image bytes.")

        vision_manager = self._vision_model_manager
        if not vision_manager:
            raise RuntimeError("Vision model manager not initialized.")

        max_retries = 8
        attempts = 0

        while attempts < max_retries:
            if vision_manager.all_models_rate_limited():
                self._log_models_exhausted(vision_manager, "VISION MODELS")
                raise Exception("All vision models are rate-limited. Please try again later.")

            current_model = vision_manager.get_current_model()
            model_name = current_model["name"]
            provider = current_model["provider"]

            try:
                if provider == "groq":
                    response_text = self._call_groq_with_vision(
                        model_name=model_name,
                        prompt=prompt,
                        image_bytes=bytes(image_bytes),
                        mime_type=mime_type,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                else:
                    response_text = self._call_gemini_with_vision(
                        model_name=model_name,
                        prompt=prompt,
                        image_bytes=bytes(image_bytes),
                        mime_type=mime_type,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                self._log_success(model_name, provider)
                return response_text

            except Exception as e:
                error_str = str(e).lower()

                if self._is_rate_limit_error(error_str):
                    retry_after = None
                    if hasattr(e, 'response') and hasattr(e.response, 'headers'):
                        retry_after = e.response.headers.get('retry-after')

                    vision_manager.mark_rate_limited(model_name, retry_after)
                    self._log_rate_limit_hit(model_name, provider, str(e))
                    attempts += 1
                    if attempts < max_retries:
                        time.sleep(0.5)
                        continue
                    self._log_models_exhausted(vision_manager, "VISION MODELS")
                    raise Exception("All vision models are rate-limited. Please try again later.")

                if self._is_model_unavailable_error(error_str) or self._is_vision_not_supported_error(error_str):
                    vision_manager.mark_model_unavailable(model_name)
                    attempts += 1
                    if attempts < max_retries:
                        continue
                    raise Exception("All configured vision models are unavailable. Please update model list.")

                Logger.log(f"[LLM MANAGER] - Error with vision {provider}/{model_name}: {str(e)}")
                raise

        self._log_models_exhausted(vision_manager, "VISION MODELS")
        raise Exception("All vision models are rate-limited. Please try again later.")

    def _call_groq(self, model_name: str, prompt: str, temperature: float, max_tokens: int) -> str:
        response = self._groq_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )

        headers = self._http_client.last_response_headers
        self._model_manager.update_rate_limit_info(model_name, headers)

        return response.choices[0].message.content

    def _call_gemini(self, model_name: str, prompt: str, temperature: float, max_tokens: int) -> str:
        response = self._gemini_client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "temperature": temperature,
                "max_output_tokens": max_tokens
            }
        )

        return response.text

    def _call_groq_with_vision(
        self,
        model_name: str,
        prompt: str,
        image_bytes: bytes,
        mime_type: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_data_url = f"data:{mime_type};base64,{image_base64}"

        response = self._groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        headers = self._http_client.last_response_headers
        if self._vision_model_manager:
            self._vision_model_manager.update_rate_limit_info(model_name, headers)

        content = response.choices[0].message.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    part_text = part.get("text")
                    if isinstance(part_text, str) and part_text.strip():
                        text_parts.append(part_text.strip())
            if text_parts:
                return "\n".join(text_parts).strip()
        return str(content or "").strip()

    def _call_gemini_with_vision(
        self,
        model_name: str,
        prompt: str,
        image_bytes: bytes,
        mime_type: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        response = self._gemini_client.models.generate_content(
            model=model_name,
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                ),
            ],
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )

        return str(response.text or "").strip()

    def _call_groq_with_tools(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Call Groq with tool support"""
        response = self._groq_client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            tool_choice="auto"
        )

        headers = self._http_client.last_response_headers
        self._agent_model_manager.update_rate_limit_info(model_name, headers)

        # Parse response
        choice = response.choices[0]
        message = choice.message

        result = {
            "content": message.content,
            "tool_calls": [],
            "finish_reason": choice.finish_reason
        }

        if message.tool_calls:
            for tool_call in message.tool_calls:
                result["tool_calls"].append({
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                })

        return result

    def _call_gemini_with_tools(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int
    ) -> Dict[str, Any]:
        """Call Gemini with tool support (fallback)"""
        # Convert OpenAI-style messages to Gemini format
        # Convert OpenAI-style tools to Gemini FunctionDeclarations
        # This is a simplified version - you may need to expand based on your needs

        # Convert tools to Gemini format
        function_declarations = []
        for tool in tools:
            if tool["type"] == "function":
                func = tool["function"]
                function_declarations.append(
                    types.FunctionDeclaration(
                        name=func["name"],
                        description=func["description"],
                        parameters=func["parameters"]
                    )
                )

        # Convert messages to Gemini format
        gemini_messages = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            gemini_messages.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })

        response = self._gemini_client.models.generate_content(
            model=model_name,
            contents=gemini_messages,
            config=types.GenerateContentConfig(
                tools=[types.Tool(function_declarations=function_declarations)],
                temperature=temperature,
                max_output_tokens=max_tokens
            )
        )

        # Parse response and convert to OpenAI-style format
        result = {
            "content": None,
            "tool_calls": [],
            "finish_reason": "stop"
        }

        for part in response.candidates[0].content.parts:
            if hasattr(part, 'text') and part.text:
                result["content"] = part.text
            elif hasattr(part, 'function_call') and part.function_call:
                result["tool_calls"].append({
                    "id": f"call_{part.function_call.name}",
                    "type": "function",
                    "function": {
                        "name": part.function_call.name,
                        "arguments": json.dumps(dict(part.function_call.args))
                    }
                })

        return result

    def _log_success(self, model_name: str, provider: str): # logging call
        if provider != "groq":
            return

        remaining_requests = "N/A"
        remaining_tokens = "N/A"
        requests_reset = "N/A"
        tokens_reset = "N/A"

        rate_limit_mgr = None

        if self._model_manager and model_name in self._model_manager.model_rate_limits:
            rate_limit_mgr = self._model_manager.model_rate_limits.get(model_name)
        elif self._agent_model_manager and model_name in self._agent_model_manager.model_rate_limits:
            rate_limit_mgr = self._agent_model_manager.model_rate_limits.get(model_name)
        elif self._vision_model_manager and model_name in self._vision_model_manager.model_rate_limits:
            rate_limit_mgr = self._vision_model_manager.model_rate_limits.get(model_name)

        if rate_limit_mgr:
            remaining_requests = (
                f"{rate_limit_mgr.requests_remaining}/{rate_limit_mgr.requests_limit}"
                if rate_limit_mgr.requests_remaining is not None
                else "N/A"
            )
            remaining_tokens = (
                f"{rate_limit_mgr.tokens_remaining}/{rate_limit_mgr.tokens_limit}"
                if rate_limit_mgr.tokens_remaining is not None
                else "N/A"
            )

            now = datetime.now()
            requests_reset = (
                f"Resetting in {int((rate_limit_mgr.requests_reset_time - now).total_seconds())}s"
                if rate_limit_mgr.requests_reset_time
                else "N/A"
            )
            tokens_reset = (
                f"Resetting in {int((rate_limit_mgr.tokens_reset_time - now).total_seconds())}s"
                if rate_limit_mgr.tokens_reset_time
                else "N/A"
            )

        with open("rate_limit.log", "a", encoding="utf-8") as log_file:
            log_file.write(f"------------------SUCCESS------------------\n")
            log_file.write(f"Provider: {provider}\n")
            log_file.write(f"Model: {model_name}\n")
            log_file.write(f"Remaining Requests (RPM): {remaining_requests}\n")
            log_file.write(f"Requests resetting in: {requests_reset}\n")
            log_file.write(f"Remaining Tokens (TPM): {remaining_tokens}\n")
            log_file.write(f"Tokens resetting in: {tokens_reset}\n")
            log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"------------------------------------------\n\n")

    def _log_rate_limit_hit(self, model_name: str, provider: str, error_message: str): # logging call
        with open("rate_limit.log", "a", encoding="utf-8") as log_file:
            log_file.write(f"--------------RATE LIMIT HIT--------------\n")
            log_file.write(f"Provider: {provider}\n")
            log_file.write(f"Model: {model_name}\n")

            if provider == "groq":
                tpd_limit = "N/A"
                tpd_used = "N/A"
                tpd_requested = "N/A"

                limit_match = re.search(r'Limit (\d+)', error_message)
                used_match = re.search(r'Used (\d+)', error_message)
                requested_match = re.search(r'Requested (\d+)', error_message)

                if limit_match:
                    tpd_limit = limit_match.group(1)
                if used_match:
                    tpd_used = used_match.group(1)
                if requested_match:
                    tpd_requested = requested_match.group(1)

                log_file.write(f"TPD Limit: {tpd_limit}, Used: {tpd_used}, Requested: {tpd_requested}\n")
            else:
                log_file.write(f"Error: {error_message}\n")

            log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"------------------------------------------\n\n")

    def _log_all_models_exhausted(self, is_agent: bool = False): # logging call
        manager = self._agent_model_manager if is_agent else self._model_manager
        model_type = "AGENT MODELS" if is_agent else "CHAT MODELS"
        self._log_models_exhausted(manager, model_type)

    def _log_models_exhausted(self, manager: Optional[_ModelManager], model_type: str): # logging call
        if manager is None:
            return

        best_model = None
        best_wait = None

        for model_dict in manager.models:
            model_name = model_dict["name"]
            if model_name in manager.model_rate_limits:
                wait = manager.model_rate_limits[model_name].get_wait_time()
                if wait is not None:
                    if best_wait is None or wait < best_wait:
                        best_wait = wait
                        best_model = model_name

        if best_model is None and manager.rate_limit_cooldowns:
            now = datetime.now()
            min_model, min_time = min(
                manager.rate_limit_cooldowns.items(),
                key=lambda x: (x[1] - now).total_seconds()
            )
            best_model = min_model
            best_wait = (min_time - now).total_seconds()

        with open("rate_limit.log", "a", encoding="utf-8") as log_file:
            if best_model and best_wait is not None:
                wait_seconds = max(0, int(best_wait))
                log_file.write(f"--------------------ALL {model_type} EXHAUSTED--------------------\n")
                log_file.write(f"{best_model} available in {wait_seconds:.1f}s\n")
                log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"------------------------------------------------------------\n\n")
            else:
                log_file.write(f"--------------------ALL {model_type} EXHAUSTED--------------------\n")
                log_file.write("No reset times or cooldowns known. Fallback wait 60s\n")
                log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"------------------------------------------------------------\n\n")

    def get_current_model(self) -> str:
        if not self._initialized:
            raise RuntimeError("LLMManager not initialized. Call initialize() first.")

        model = self._model_manager.get_current_model()
        return f"{model['provider']}/{model['name']}"

    def get_current_agent_model(self) -> str:
        if not self._initialized:
            raise RuntimeError("LLMManager not initialized. Call initialize() first.")

        model = self._agent_model_manager.get_current_model()
        return f"{model['provider']}/{model['name']}"

    def get_current_vision_model(self) -> str:
        if not self._initialized:
            raise RuntimeError("LLMManager not initialized. Call initialize() first.")

        model = self._vision_model_manager.get_current_model()
        return f"{model['provider']}/{model['name']}"


LLMManager = LLMManagerClass()
