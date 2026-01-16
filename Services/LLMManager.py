from typing import Optional, Dict
from groq import Groq
from google import genai
import os
import re
import time
import httpx
from datetime import datetime, timedelta
from Services import Logger
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class RateLimitCapturingClient(httpx.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_response_headers = {}

    def send(self, request, *args, **kwargs):
        response = super().send(request, *args, **kwargs)
        self.last_response_headers = dict(response.headers)
        return response

class _RateLimitManager:
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
    def _parse_retry_after(retry_after: Optional[str]) -> Optional[float]:
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

    def _parse_reset_time(self, reset_str: str) -> datetime:
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

    def get_wait_time(self) -> float:
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

class _ModelManager:
    MODELS = [
        {"name": "llama-3.3-70b-versatile", "provider": "groq"},      # 30 RPM, 1K RPD, 12K TPM, 100K TPD
        {"name": "llama-3.1-8b-instant", "provider": "groq"},         # 30 RPM, 14.4K RPD, 6K TPM, 500K TPD
        {"name": "gemini-3-flash", "provider": "gemini"},             # 5 RPM, 250K TPM
        {"name": "gemini-2.5-flash", "provider": "gemini"},           # 5 RPM, 250K TPM
        {"name": "gemini-2.5-flash-lite", "provider": "gemini"}       # 10 RPM, 250K TPM
    ]

    def __init__(self):
        self.current_model_index = 0
        self.model_rate_limits: Dict[str, _RateLimitManager] = {
            model["name"]: _RateLimitManager() for model in self.MODELS
        }
        self.rate_limit_cooldowns: Dict[str, datetime] = {}

    def get_current_model(self) -> Dict:
        original_index = self.current_model_index
        attempts = 0

        while attempts < len(self.MODELS):
            model = self.MODELS[self.current_model_index]

            if self._model_available(model["name"]):
                return model

            self.current_model_index = (self.current_model_index + 1) % len(self.MODELS)
            attempts += 1

            if self.current_model_index == original_index:
                return self._get_model_with_shortest_cd()

        return self.MODELS[self.current_model_index]

    def _model_available(self, model_name: str) -> bool:
        if model_name not in self.rate_limit_cooldowns:
            return True

        now = datetime.now()
        if now >= self.rate_limit_cooldowns[model_name]:
            del self.rate_limit_cooldowns[model_name]
            return True

        return False

    def _get_model_with_shortest_cd(self) -> Dict:
        if not self.rate_limit_cooldowns:
            return self.MODELS[0]

        now = datetime.now()
        min_cooldown_model_name = min(
            self.rate_limit_cooldowns.items(),
            key=lambda x: (x[1] - now).total_seconds()
        )[0]

        for model in self.MODELS:
            if model["name"] == min_cooldown_model_name:
                return model

        return self.MODELS[0]

    def update_rate_limit_info(self, model_name: str, headers: Dict[str, str]):
        if model_name in self.model_rate_limits:
            self.model_rate_limits[model_name].update_headers(headers)

    def mark_rate_limited(self, model_name: str, retry_after: Optional[str] = None):
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

        self.current_model_index = (self.current_model_index + 1) % len(self.MODELS)

    def all_models_rate_limited(self) -> bool:
        return len(self.rate_limit_cooldowns) >= len(self.MODELS)

class LLMManagerClass:
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
        return cls._instance

    def initialize(self):
        if self._initialized:
            return

        self._groq_client = Groq(api_key=GROQ_API_KEY, http_client=self._http_client)
        self._gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        self._model_manager = _ModelManager()
        self._initialized = True

        print(f"LLM MANAGER INITIALIZED.\n")

    def chat(self, prompt: str) -> str:
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
                self._log(model=model_name, provider=provider, prompt=prompt, response=assistant_response)

                return assistant_response

            except Exception as e:
                error_str = str(e).lower()

                if "rate_limit" in error_str or "429" in error_str or "too many requests" in error_str or "quota" in error_str or "resource_exhausted" in error_str:
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
                else:
                    Logger.log(f"[LLM MANAGER] - Error with {provider}/{model_name}: {str(e)}")
                    raise

        self._log_all_models_exhausted()
        return self.FALLBACK_MESSAGE

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

    def _log_success(self, model_name: str, provider: str):
            if provider == "groq":
                rate_limit_mgr = self._model_manager.model_rate_limits.get(model_name)
                if rate_limit_mgr:
                    remaining_requests = f"{rate_limit_mgr.requests_remaining}/{rate_limit_mgr.requests_limit}" if rate_limit_mgr.requests_remaining is not None else "N/A"
                    remaining_tokens = f"{rate_limit_mgr.tokens_remaining}/{rate_limit_mgr.tokens_limit}" if rate_limit_mgr.tokens_remaining is not None else "N/A"

                    now = datetime.now()
                    requests_reset = f"Resetting in {int((rate_limit_mgr.requests_reset_time - now).total_seconds())}s" if rate_limit_mgr.requests_reset_time else "N/A"
                    tokens_reset = f"Resetting in {int((rate_limit_mgr.tokens_reset_time - now).total_seconds())}s" if rate_limit_mgr.tokens_reset_time else "N/A"

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

    def _log_rate_limit_hit(self, model_name: str, provider: str, error_message: str):
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

    def _log_all_models_exhausted(self):
        best_model = None
        best_wait = None

        for model_dict in self._model_manager.MODELS:
            model_name = model_dict["name"]
            if model_name in self._model_manager.model_rate_limits:
                wait = self._model_manager.model_rate_limits[model_name].get_wait_time()
                if wait is not None:
                    if best_wait is None or wait < best_wait:
                        best_wait = wait
                        best_model = model_name

        if best_model is None and self._model_manager.rate_limit_cooldowns:
            now = datetime.now()
            min_model, min_time = min(
                self._model_manager.rate_limit_cooldowns.items(),
                key=lambda x: (x[1] - now).total_seconds()
            )
            best_model = min_model
            best_wait = (min_time - now).total_seconds()

        with open("rate_limit.log", "a", encoding="utf-8") as log_file:
            if best_model and best_wait is not None:
                wait_seconds = max(0, int(best_wait))
                log_file.write(f"--------------------ALL MODELS EXHAUSTED--------------------\n")
                log_file.write(f"{best_model} available in {wait_seconds:.1f}s\n")
                log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"------------------------------------------------------------\n\n")
            else:
                log_file.write(f"--------------------ALL MODELS EXHAUSTED--------------------\n")
                log_file.write("No reset times or cooldowns known. Fallback wait 60s\n")
                log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"------------------------------------------------------------\n\n")

    def _log(self, model: str, provider: str, prompt: str, response: str):
        with open("rag.log", "a", encoding="utf-8") as log_file:
                log_file.write(f"---\n")
                log_file.write(f"Provider: {provider}\n")
                log_file.write(f"Chat Model: {model}\n")
                log_file.write(f"RAG Output:\n{prompt}\n")
                log_file.write(f"\nResponse:\n{response}\n")
                log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"---\n\n")

    def get_current_model(self) -> str:
        if not self._initialized:
            raise RuntimeError("LLMManager not initialized. Call initialize() first.")

        model = self._model_manager.get_current_model()
        return f"{model['provider']}/{model['name']}"


LLMManager = LLMManagerClass()