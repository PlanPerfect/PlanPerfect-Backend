from typing import Optional, Dict
from groq import Groq
import os
import time
import httpx
from datetime import datetime, timedelta
from dotenv import load_dotenv
from Services import Logger

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

    def _parse_reset_time(self, reset_str: str) -> datetime:
        try:
            total_seconds = 0

            if 'm' in reset_str:
                parts = reset_str.split('m')
                total_seconds += int(parts[0]) * 60
                reset_str = parts[1]

            if 's' in reset_str:
                total_seconds += float(reset_str.rstrip('s'))

            return datetime.now() + timedelta(seconds=total_seconds)
        except Exception as e:
            Logger.log(f"[GROQ MANAGER] - Error parsing reset time '{reset_str}': {str(e)}")
            return datetime.now() + timedelta(minutes=1)

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
            wait_times.append((self.tokens_reset_time - now).total_seconds())

        if self.requests_reset_time:
            wait_times.append((self.requests_reset_time - now).total_seconds())

        if wait_times:
            return max(0, min(wait_times)) + 1

        return 60

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

        if retry_after:
            try:
                cooldown_seconds = float(retry_after)
                cooldown_time = now + timedelta(seconds=cooldown_seconds + 2)
            except:
                cooldown_time = now + timedelta(seconds=60)
        elif model in self.model_rate_limits:
            wait_time = self.model_rate_limits[model].get_wait_time()
            cooldown_time = now + timedelta(seconds=wait_time)
        else:
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

        requests_reset = f"{rate_limit_mgr.requests_reset_time}"
        tokens_reset = f"{rate_limit_mgr.tokens_reset_time}"

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

        requests_reset = f"{rate_limit_mgr.requests_reset_time}"
        tokens_reset = f"{rate_limit_mgr.tokens_reset_time}"

        with open("rate_limit.log", "a", encoding="utf-8") as log_file:
            log_file.write(f"--------------RATE LIMIT HIT--------------\n")
            log_file.write(f"Model: {model}\n")
            log_file.write(f"Requests resetting in: {requests_reset}\n")
            log_file.write(f"Tokens resetting in: {tokens_reset}\n")
            log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"------------------------------------------\n\n")

    def _log_all_models_exhausted(self):
        if not self._model_manager.rate_limit_cooldowns:
            return

        now = datetime.now()

        min_cooldown_model = min(
            self._model_manager.rate_limit_cooldowns.items(),
            key=lambda x: (x[1] - now).total_seconds()
        )

        wait_seconds = (min_cooldown_model[1] - now).total_seconds()

        with open("rate_limit.log", "a", encoding="utf-8") as log_file:
            log_file.write(f"--------------------ALL MODELS EXHAUSTED--------------------\n")
            log_file.write(f"{min_cooldown_model[0]} available in {wait_seconds:.1f}s\n")
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