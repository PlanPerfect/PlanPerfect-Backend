from typing import Optional, Dict
from groq import Groq
import os
import re
import time
import httpx
from datetime import datetime, timedelta
from Services import Logger
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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
            Logger.log(f"[GROQ MANAGER] - Error parsing rate limit headers: {str(e)}")

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
            Logger.log(f"[GROQ MANAGER] - Error parsing reset time '{reset_str}': {str(e)}")
            return datetime.now() + timedelta(seconds=60)

    def nearly_depleted(self, request_threshold: int = 5, token_threshold: int = 1000) -> bool:
        if self.requests_remaining is not None and self.requests_remaining <= request_threshold:
            return True
        if self.tokens_remaining is not None and self.tokens_remaining <= token_threshold:
            return True
        return False

    def switch_model(self) -> bool:
        if self.nearly_depleted(request_threshold=3, token_threshold=500):
            return True

        now = datetime.now()
        if self.tokens_reset_time and self.tokens_remaining:
            time_until_reset = (self.tokens_reset_time - now).total_seconds()
            if time_until_reset > 30 and self.tokens_limit:
                if self.tokens_remaining < (self.tokens_limit * 0.2):
                    return True

        return False

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
        "llama-3.3-70b-versatile",      # 30 RPM, 12K TPM
        "llama-3.1-8b-instant",         # 30 RPM, 6K TPM
        "openai/gpt-oss-120b",          # 30 RPM, 8K TPM
        "openai/gpt-oss-20b"            # 30 RPM, 8K TPM
    ]

    def __init__(self):
        self.current_model_index = 0
        self.model_rate_limits: Dict[str, _RateLimitManager] = {
            model: _RateLimitManager() for model in self.MODELS
        }
        self.rate_limit_cooldowns: Dict[str, datetime] = {}

    def _wait_seconds_until_reset_for_model(self, model: str) -> Optional[float]:
        rate_limit_mgr = self.model_rate_limits.get(model)
        if not rate_limit_mgr:
            return None
        now = datetime.now()
        candidates = []
        if rate_limit_mgr.requests_reset_time:
            dt = (rate_limit_mgr.requests_reset_time - now).total_seconds()
            if dt > 0:
                candidates.append(dt)
        if rate_limit_mgr.tokens_reset_time:
            dt = (rate_limit_mgr.tokens_reset_time - now).total_seconds()
            if dt > 0:
                candidates.append(dt)
        if candidates:
            return min(candidates)
        return None

    def get_current_model(self) -> str:
        original_index = self.current_model_index
        attempts = 0

        while attempts < len(self.MODELS):
            model = self.MODELS[self.current_model_index]

            if self._model_available(model):
                return model

            self.current_model_index = (self.current_model_index + 1) % len(self.MODELS)
            attempts += 1

            if self.current_model_index == original_index:
                return self._get_model_with_shortest_cd()

        return self.MODELS[self.current_model_index]

    def _model_available(self, model: str) -> bool:
        if model not in self.rate_limit_cooldowns:
            return True

        now = datetime.now()
        if now >= self.rate_limit_cooldowns[model]:
            del self.rate_limit_cooldowns[model]
            return True

        return False

    def _get_model_with_shortest_cd(self) -> str:
        if not self.rate_limit_cooldowns:
            return self.MODELS[0]

        now = datetime.now()
        min_cooldown_model = min(
            self.rate_limit_cooldowns.items(),
            key=lambda x: (x[1] - now).total_seconds()
        )

        return min_cooldown_model[0]

    def update_rate_limit_info(self, model: str, headers: Dict[str, str]):
        if model in self.model_rate_limits:
            self.model_rate_limits[model].update_headers(headers)

    def switch_model_proactively(self, model: str) -> bool:
        if model in self.model_rate_limits:
            return self.model_rate_limits[model].switch_model()
        return False

    def mark_rate_limited(self, model: str, retry_after: Optional[str] = None):
        now = datetime.now()

        cooldown_time = None
        if retry_after:
            secs = _RateLimitManager._parse_retry_after(retry_after)
            if secs:
                cooldown_time = now + timedelta(seconds=secs + 2)

        if not cooldown_time and model in self.model_rate_limits:
            wait_secs = self.model_rate_limits[model].get_wait_time()
            if wait_secs and wait_secs > 0:
                cooldown_time = now + timedelta(seconds=wait_secs)

        if not cooldown_time:
            cooldown_time = now + timedelta(seconds=60)

        self.rate_limit_cooldowns[model] = cooldown_time

        self.current_model_index = (self.current_model_index + 1) % len(self.MODELS)

    def get_next_model(self) -> str:
        self.current_model_index = (self.current_model_index + 1) % len(self.MODELS)
        return self.get_current_model()

class GroqManagerClass:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._http_client = RateLimitCapturingClient()
            cls._instance._initialized = False
            cls._instance._client = None
            cls._instance._model_manager = None
        return cls._instance

    def initialize(self):
        if self._initialized:
            return

        self._client = Groq(api_key=GROQ_API_KEY, http_client=self._http_client)
        self._model_manager = _ModelManager()
        self._initialized = True

        print(f"GROQ MANAGER INITIALIZED. USING \033[94m{self._model_manager.get_current_model()}\033[0m\n")

    def chat(self, prompt: str) -> str:
        if not self._initialized:
            raise RuntimeError("GroqManager not initialized. Call initialize() first.")

        temperature = 0.8
        max_tokens = 1024
        max_retries = 4
        attempts = 0

        while attempts < max_retries:
            current_model = self._model_manager.get_current_model()

            try:
                response = self._client.chat.completions.create(
                    model=current_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                headers = self._http_client.last_response_headers

                self._model_manager.update_rate_limit_info(current_model, headers)

                assistant_response = response.choices[0].message.content

                self._log_success(current_model)
                self._log(model=current_model, prompt=prompt, response=assistant_response)

                if self._model_manager.switch_model_proactively(current_model):
                    next_model = self._model_manager.get_next_model()
                    self._log_proactive_switch(current_model, next_model)

                return assistant_response

            except Exception as e:
                error_str = str(e).lower()

                if "rate_limit" in error_str or "429" in error_str or "too many requests" in error_str:
                    retry_after = None
                    if hasattr(e, 'response') and hasattr(e.response, 'headers'):
                        retry_after = e.response.headers.get('retry-after')

                    self._model_manager.mark_rate_limited(current_model, retry_after)

                    self._log_rate_limit_hit(current_model)

                    attempts += 1

                    if attempts < max_retries:
                        time.sleep(0.5)
                        continue
                    else:
                        self._log_all_models_exhausted()
                        raise RuntimeError(f"All models exhausted after {max_retries} attempts. Last error: {str(e)}")
                else:
                    Logger.log(f"[GROQ MANAGER] - Error: {str(e)}")
                    raise

        raise RuntimeError(f"Chat completion failed after {max_retries} attempts")

    def _log_success(self, model: str):
        rate_limit_mgr = self._model_manager.model_rate_limits.get(model)
        if not rate_limit_mgr:
            return

        remaining_requests = f"{rate_limit_mgr.requests_remaining}/{rate_limit_mgr.requests_limit}" if rate_limit_mgr.requests_remaining is not None else "N/A"
        remaining_tokens = f"{rate_limit_mgr.tokens_remaining}/{rate_limit_mgr.tokens_limit}" if rate_limit_mgr.tokens_remaining is not None else "N/A"

        now = datetime.now()
        requests_reset = f"Resetting in {int((rate_limit_mgr.requests_reset_time - now).total_seconds())}s" if rate_limit_mgr.requests_reset_time else "N/A"
        tokens_reset = f"Resetting in {int((rate_limit_mgr.tokens_reset_time - now).total_seconds())}s" if rate_limit_mgr.tokens_reset_time else "N/A"

        with open("rate_limit.log", "a", encoding="utf-8") as log_file:
            log_file.write(f"------------------SUCCESS------------------\n")
            log_file.write(f"Model: {model}\n")
            log_file.write(f"Remaining Requests (RPD): {remaining_requests}\n")
            log_file.write(f"Requests resetting in: {requests_reset}\n")
            log_file.write(f"Remaining Tokens (TPM): {remaining_tokens}\n")
            log_file.write(f"Tokens resetting in: {tokens_reset}\n")

            log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"-------------------------------------------\n\n")

    def _log_rate_limit_hit(self, model: str):
        rate_limit_mgr = self._model_manager.model_rate_limits.get(model)
        if not rate_limit_mgr:
            return

        with open("rate_limit.log", "a", encoding="utf-8") as log_file:
            log_file.write(f"--------------RATE LIMIT HIT--------------\n")
            log_file.write(f"Model: {model}\n")
            log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"------------------------------------------\n\n")

    def _log_all_models_exhausted(self):
        best_model = None
        best_wait = None

        for model in self.MODELS:
            wait = self._wait_seconds_until_reset_for_model(model)
            if wait is not None:
                if best_wait is None or wait < best_wait:
                    best_wait = wait
                    best_model = model

        if best_model is None and self.rate_limit_cooldowns:
            now = datetime.now()
            min_model, min_time = min(
                self.rate_limit_cooldowns.items(),
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

    def _log_proactive_switch(self, from_model: str, to_model: str):
        rate_limit_mgr = self._model_manager.model_rate_limits.get(from_model)

        reason = "Unknown"

        if rate_limit_mgr:
            if rate_limit_mgr.requests_remaining is not None and rate_limit_mgr.requests_remaining <= 3:
                reason = f"Low requests ({rate_limit_mgr.requests_remaining} remaining)"
            elif rate_limit_mgr.tokens_remaining is not None and rate_limit_mgr.tokens_remaining <= 500:
                reason = f"Low tokens ({rate_limit_mgr.tokens_remaining} remaining)"
            elif rate_limit_mgr.tokens_limit and rate_limit_mgr.tokens_remaining:
                if rate_limit_mgr.tokens_remaining < (rate_limit_mgr.tokens_limit * 0.1):
                    reason = f"Below 10% capacity ({rate_limit_mgr.tokens_remaining}/{rate_limit_mgr.tokens_limit} tokens)"

        with open("rate_limit.log", "a", encoding="utf-8") as log_file:
            log_file.write(f"--------------PROACTIVE MODEL SWITCH--------------\n")
            log_file.write(f"From: {from_model}\n")
            log_file.write(f"To: {to_model}\n")
            log_file.write(f"Reason: {reason}\n")
            log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"--------------------------------------------------\n\n")

    def _log(self, model: str, prompt: str, response: str):
        with open("rag.log", "a", encoding="utf-8") as log_file:
                log_file.write(f"---\n")
                log_file.write(f"Chat Model: {model}\n")
                log_file.write(f"RAG Output:\n{prompt}\n")
                log_file.write(f"\nResponse:\n{response}\n")
                log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"---\n\n")

    def get_current_model(self) -> str:
        if not self._initialized:
            raise RuntimeError("GroqManager not initialized. Call initialize() first.")

        return self._model_manager.get_current_model()


GroqManager = GroqManagerClass()