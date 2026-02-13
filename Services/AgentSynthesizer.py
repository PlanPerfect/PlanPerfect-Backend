from typing import Optional, Dict, List, Any
import uuid
import json
import inspect
from io import BytesIO
from datetime import datetime
import numpy as np
from PIL import Image
from Services import Logger
from Services import DatabaseManager as DM
from Services import ServiceOrchestra as SO
from Services import LLMManager
from Services import FileManager as FM

"""
AgentSynthesizer is an autonomous agentic AI system that:
1. Takes user queries and decides whether to use tools or just chat
2. Implements a ReAct-inspired loop (Think -> Act -> Observe)
3. Tracks all steps in the database for observability
4. Uses Groq/Gemini agent models with native function calling via LLMManager
"""


class AgentSynthesizerClass:
    _instance = None
    STATUS_IDLE = "idle"
    STATUS_RUNNING = "running"
    HISTORY_LIMIT = 50
    THINKING_STEP = "Thinking..."
    ANALYZING_FILES_STEP = "Analysing files..."
    DONE_STEP = "Done!"
    USER_ERROR_MESSAGE = "I'm sorry, but i'm having trouble completing your request right now. Please try again later."
    OUT_OF_SCOPE_MESSAGE = (
        "Sorry, but I'm only able to help with interior design related questions and tasks. If you need help, feel free to ask!"
    )
    TOOL_CURRENT_STEPS = {
        "generate_image": "Generating Image...",
        "classify_style": "Classifying Style...",
        "detect_furniture": "Detecting Furniture...",
        "get_recommendations": "Getting Recommendations...",
        "web_search": "Searching the web...",
        "extract_colors": "Extracting Colors...",
        "generate_floor_plan": "Generating Floor Plan...",
    }
    SYSTEM_PROMPT = (
        "You are an interior design assistant. You must only help with interior design workflows, "
        "design questions, and tasks related to rooms, furniture, floor plans, layouts, decor, "
        "styles, and color palettes. If a user asks for anything outside interior design, refuse "
        "briefly and redirect to interior-design help. Never call tools for out-of-scope tasks. "
        "When image tools require file types, use only files that match the tool requirement."
    )

    OUTPUT_BRANCHES = {
        "generate_image": "Generated Images",
        "classify_style": "Classified Style",
        "detect_furniture": "Detected Furniture",
        "get_recommendations": "Recommendations",
        "web_search": "Web Searches",
        "extract_colors": "Extracted Colors",
        "generate_floor_plan": "Generated Floor Plans",
    }

    URL_OUTPUT_TOOLS = {
        "generate_image",
        "get_recommendations",
        "generate_floor_plan",
    }

    FILE_REQUIRED_TOOLS = {
        "classify_style",
        "detect_furniture",
        "extract_colors",
        "generate_floor_plan",
    }

    FILE_REQUIREMENTS = {
        "classify_style": "an interior room photo where the design style is clearly visible",
        "detect_furniture": "a clear room photo with visible furniture",
        "extract_colors": "an image with visible colors (photo, artwork, or palette)",
        "generate_floor_plan": "a top-down floor plan or architectural layout image",
    }

    SCOPE_CLASSIFIER_INSTRUCTION = (
        "You are a strict scope classifier for an interior-design assistant. "
        "Classify the latest user query into exactly one category:\n"
        "1) interior_design: The user asks for interior design workflows/questions, or is continuing an existing interior design task.\n"
        "2) small_talk: Greeting/pleasantry/chitchat without requesting unrelated domain knowledge.\n"
        "3) out_of_scope: The user asks for non-interior-design tasks or information.\n\n"
        "Return JSON only with keys: classification, confidence, explanation.\n"
        "Valid classification values: interior_design, small_talk, out_of_scope.\n"
        "confidence must be a float from 0 to 1."
    )

    # Tool definitions in OpenAI/Groq format
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "generate_image",
                "description": "Generate an AI image based on a text prompt. Use this when the user asks to create, generate, or make an image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The detailed image generation prompt describing what to create",
                        }
                    },
                    "required": ["prompt"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "classify_style",
                "description": "Classify the interior design style of an uploaded image. Use this when user uploads an interior image and wants to know the style (modern, traditional, minimalist, etc.)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_id": {
                            "type": "string",
                            "description": "The file ID of the uploaded image to classify",
                        }
                    },
                    "required": ["file_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "detect_furniture",
                "description": "Detect and extract furniture items from an interior image. Returns cropped images of each detected furniture piece. Use when user wants to identify or extract furniture from a room photo.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_id": {
                            "type": "string",
                            "description": "The file ID of the image to analyze for furniture",
                        }
                    },
                    "required": ["file_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_recommendations",
                "description": "Get furniture recommendations based on a style and furniture type. Returns product recommendations with images. Use when user wants suggestions for specific furniture in a certain style.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "style": {
                            "type": "string",
                            "description": "The interior design style (e.g., 'modern', 'traditional', 'minimalist')",
                        },
                        "furniture_name": {
                            "type": "string",
                            "description": "The type of furniture (e.g., 'sofa', 'chair', 'table')",
                        },
                    },
                    "required": ["style", "furniture_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current interior-design information only. Use when user asks for recent interior design trends, products, materials, or style information requiring up-to-date data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "extract_colors",
                "description": "Extract the dominant color palette from an image. Returns 5 dominant colors with RGB and hex values. Use when user wants to know the color scheme of an image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_id": {
                            "type": "string",
                            "description": "The file ID of the image to extract colors from",
                        }
                    },
                    "required": ["file_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_floor_plan",
                "description": "Generate a furnished floor plan by placing furniture on an uploaded floor plan image. Use when user uploads a floor plan and wants to see furniture placement visualization.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_id": {
                            "type": "string",
                            "description": "The file ID of the floor plan image",
                        },
                        "furniture_list": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of furniture items to place on the floor plan (e.g., ['sofa', 'dining table', 'bed'])",
                        },
                    },
                    "required": ["file_id", "furniture_list"],
                },
            },
        },
    ]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._instance._file_registry = {}
        return cls._instance

    def initialize(self):
        if self._initialized:
            return

        try:
            if not LLMManager._initialized:
                raise RuntimeError("LLMManager must be initialized before AgentSynthesizer")

            self._clear_all_user_agents()
            self._initialized = True
            print("AGENT SYNTHESIZER INITIALIZED. AGENTIC SYSTEM READY.\n")
            print(f"Using agent model: {LLMManager.get_current_agent_model()}\n")
        except Exception as e:
            Logger.log(f"[AGENT SYNTHESIZER] - ERROR: Failed to initialize. Error: {e}")
            raise

    def _clear_all_user_agents(self) -> None:
        users = DM.data.get("Users")
        if not isinstance(users, dict):
            return

        cleared_count = 0
        for user_data in users.values():
            if not isinstance(user_data, dict):
                continue
            if "Agent" in user_data:
                del user_data["Agent"]
                cleared_count += 1

        if cleared_count > 0:
            DM.save()
            Logger.log(
                f"[AGENT SYNTHESIZER] - INFO: Cleared Agent state for {cleared_count} user(s) during initialization."
            )

    def register_file(
        self,
        file_id: str,
        file_obj: Any,
        user_id: Optional[str] = None,
        source: str = "runtime",
    ) -> None:
        self._file_registry[file_id] = file_obj

        if user_id:
            self._upsert_session_file(
                user_id=user_id,
                file_id=file_id,
                file_obj=file_obj,
                source=source,
            )
            self._add_step(
                user_id=user_id,
                step_type="file_uploaded",
                content={
                    "file_id": file_id,
                    "filename": getattr(file_obj, "filename", ""),
                    "content_type": getattr(file_obj, "content_type", ""),
                    "source": source,
                },
            )
            self._update_session_status(
                user_id=user_id,
                status="completed",
                current_step=self.DONE_STEP,
            )
            DM.save()

    def unregister_file(self, file_id: str, user_id: Optional[str] = None) -> None:
        self._file_registry.pop(file_id, None)

        if user_id:
            self._remove_session_file(user_id=user_id, file_id=file_id)
            self._update_session_status(
                user_id=user_id,
                status="completed",
                current_step=self.DONE_STEP,
            )
            DM.save()

    def clear_file_registry(self, user_id: Optional[str] = None) -> None:
        if user_id:
            session_files = self._get_session_files(user_id=user_id)
            for session_file in session_files:
                file_id = session_file.get("file_id")
                if isinstance(file_id, str):
                    self._file_registry.pop(file_id, None)

            agent_data = self._get_agent_record(user_id)
            agent_data["Uploaded Files"] = []
            self._clear_pending_file_action(user_id=user_id)
            self._update_session_status(
                user_id=user_id,
                status="completed",
                current_step=self.DONE_STEP,
            )
            DM.save()
            return

        self._file_registry.clear()

    async def execute(
        self,
        user_id: str,
        query: str,
        session_id: Optional[str] = None,
        max_iterations: int = 10,
        uploaded_files: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("AgentSynthesizer not initialized. Call initialize() first.")

        session_data = self._get_or_initialize_session(
            user_id=user_id,
            requested_session_id=session_id,
        )
        current_session_id = session_data["session_id"]
        history_steps = session_data.get("steps", [])[-self.HISTORY_LIMIT:]

        if uploaded_files:
            self._ingest_uploaded_files(
                user_id=user_id,
                uploaded_files=uploaded_files,
                source="request",
            )

        self._update_session_status(
            user_id=user_id,
            status="thinking",
            current_step=self.THINKING_STEP,
        )
        self._add_step(user_id=user_id, step_type="user_query", content=query)
        DM.save()

        try:
            pending_action = self._get_pending_file_action(user_id=user_id)
            forced_file_context: Optional[Dict[str, Any]] = None
            effective_query = query

            if pending_action:
                pending_resolution = self._resolve_pending_file_action(
                    user_id=user_id,
                    user_reply=query,
                    pending_action=pending_action,
                )

                pending_status = pending_resolution.get("status")
                if pending_status == "waiting_for_confirmation":
                    waiting_message = pending_resolution.get(
                        "response",
                        "Please let me know if you'd like me to continue with that file, or use a different one.",
                    )
                    self._add_step(
                        user_id=user_id,
                        step_type="response",
                        content=waiting_message,
                    )
                    self._update_session_status(
                        user_id=user_id,
                        status="awaiting_user_confirmation",
                        current_step=self.DONE_STEP,
                    )
                    DM.save()
                    return {
                        "session_id": current_session_id,
                        "status": "awaiting_user_confirmation",
                        "response": waiting_message,
                    }

                if pending_status == "needs_file_upload":
                    upload_message = pending_resolution.get(
                        "response",
                        "Sure. Please upload the file you'd like me to use instead.",
                    )
                    self._add_step(
                        user_id=user_id,
                        step_type="response",
                        content=upload_message,
                    )
                    self._update_session_status(
                        user_id=user_id,
                        status="waiting_for_file_upload",
                        current_step=self.DONE_STEP,
                    )
                    DM.save()
                    return {
                        "session_id": current_session_id,
                        "status": "waiting_for_file_upload",
                        "response": upload_message,
                    }

                if pending_status == "confirmed":
                    confirmed_action = pending_resolution.get("confirmed_action")
                    if isinstance(confirmed_action, dict):
                        return await self._execute_confirmed_file_action(
                            user_id=user_id,
                            current_session_id=current_session_id,
                            confirmed_action=confirmed_action,
                        )

                    forced_file_context = pending_resolution.get("forced_file_context")
                    confirmed_query = pending_resolution.get("effective_query")
                    if isinstance(confirmed_query, str) and confirmed_query.strip():
                        effective_query = confirmed_query

            scope_decision = None
            if not forced_file_context:
                scope_decision = self._evaluate_request_scope(
                    effective_query,
                    user_id=user_id,
                    record_step=True,
                )

            if scope_decision and not scope_decision.get("allowed", True):
                self._add_step(
                    user_id=user_id,
                    step_type="scope_blocked",
                    content={
                        "query": effective_query,
                        "reason": "out_of_scope",
                        "scope_decision": scope_decision,
                    },
                )
                self._add_step(
                    user_id=user_id,
                    step_type="response",
                    content=self.OUT_OF_SCOPE_MESSAGE,
                )
                self._update_session_status(
                    user_id=user_id,
                    status="completed",
                    current_step=self.DONE_STEP,
                )
                DM.save()
                return {
                    "session_id": current_session_id,
                    "status": "completed",
                    "response": self.OUT_OF_SCOPE_MESSAGE,
                }

            messages = self._build_messages(query=effective_query, history_steps=history_steps)

            iteration = 0
            while iteration < max_iterations:
                iteration += 1

                self._update_session_status(
                    user_id=user_id,
                    status="thinking",
                    current_step=self.THINKING_STEP,
                )
                self._add_step(
                    user_id=user_id,
                    step_type="thought",
                    content=f"Iteration {iteration}: Thinking...",
                )

                try:
                    response = LLMManager.chat_with_tools(
                        messages=messages,
                        tools=self.TOOLS,
                        temperature=0.2,
                        max_tokens=2048,
                    )
                except Exception as e:
                    error_msg = f"LLM call failed: {str(e)}"
                    Logger.log(f"[AGENT SYNTHESIZER] - ERROR: {error_msg}")
                    self._update_session_status(
                        user_id=user_id,
                        status="error",
                        current_step=self.THINKING_STEP,
                    )
                    DM.save()
                    return {
                        "session_id": current_session_id,
                        "status": "error",
                        "error": error_msg,
                        "response": self.USER_ERROR_MESSAGE,
                    }

                tool_calls = response.get("tool_calls", [])
                text_response = response.get("content")

                if not tool_calls:
                    final_response = text_response or "I've completed processing your request."

                    self._add_step(
                        user_id=user_id,
                        step_type="response",
                        content=final_response,
                    )
                    self._update_session_status(
                        user_id=user_id,
                        status="completed",
                        current_step=self.DONE_STEP,
                    )
                    DM.save()

                    return {
                        "session_id": current_session_id,
                        "status": "completed",
                        "response": final_response,
                        "iterations": iteration,
                    }

                assistant_message = {
                    "role": "assistant",
                    "content": text_response,
                    "tool_calls": tool_calls,
                }
                messages.append(assistant_message)

                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    function_args = self._safe_parse_tool_args(
                        tool_call["function"].get("arguments", "{}")
                    )
                    tool_call_id = tool_call["id"]

                    file_resolution = await self._resolve_tool_file_requirements(
                        user_id=user_id,
                        query=effective_query,
                        tool_name=function_name,
                        tool_args=function_args,
                        forced_file_context=forced_file_context,
                    )
                    resolution_status = file_resolution.get("status")

                    if file_resolution.get("applied_forced_context"):
                        forced_file_context = None

                    if resolution_status in {"ask_confirmation", "needs_file_upload"}:
                        response_message = file_resolution.get("response")
                        if not isinstance(response_message, str) or not response_message.strip():
                            response_message = (
                                "I need a file before I can continue. "
                                "Please upload one and let me know when you're ready."
                            )

                        self._add_step(
                            user_id=user_id,
                            step_type="response",
                            content=response_message,
                        )

                        status_value = (
                            "awaiting_user_confirmation"
                            if resolution_status == "ask_confirmation"
                            else "waiting_for_file_upload"
                        )
                        self._update_session_status(
                            user_id=user_id,
                            status=status_value,
                            current_step=self.DONE_STEP,
                        )
                        DM.save()

                        return {
                            "session_id": current_session_id,
                            "status": status_value,
                            "response": response_message,
                            "iterations": iteration,
                        }

                    function_args = file_resolution.get("tool_args", function_args)

                    self._update_session_status(
                        user_id=user_id,
                        status="executing",
                        current_step=self._tool_current_step(function_name),
                    )
                    self._add_step(
                        user_id=user_id,
                        step_type="tool_call",
                        content=function_args,
                        tool_name=function_name,
                    )

                    try:
                        result = await self._execute_tool(
                            function_name,
                            function_args,
                            user_id=user_id,
                        )

                        output_summary = self._store_tool_output(
                            user_id=user_id,
                            tool_name=function_name,
                            args=function_args,
                            result=result,
                        )

                        if output_summary:
                            self._add_step(
                                user_id=user_id,
                                step_type="tool_result",
                                content=output_summary,
                                tool_name=function_name,
                            )

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": json.dumps(result),
                            }
                        )
                        DM.save()

                    except Exception as e:
                        error_msg = f"Tool execution failed for {function_name}: {str(e)}"
                        Logger.log(f"[AGENT SYNTHESIZER] - ERROR: {error_msg}")

                        self._add_step(
                            user_id=user_id,
                            step_type="error",
                            content={"error": str(e)},
                            tool_name=function_name,
                        )

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": json.dumps({"error": str(e)}),
                            }
                        )
                        DM.save()

                self._update_session_status(
                    user_id=user_id,
                    status="thinking",
                    current_step=self.THINKING_STEP,
                )

            self._update_session_status(
                user_id=user_id,
                status="completed",
                current_step=self.DONE_STEP,
            )
            DM.save()

            return {
                "session_id": current_session_id,
                "status": "completed",
                "response": "I've processed your request through multiple steps. The results are available in Agent Outputs.",
                "iterations": max_iterations,
                "note": "Maximum iterations reached",
            }

        except Exception as e:
            error_msg = f"Agent execution failed: {str(e)}"
            Logger.log(f"[AGENT SYNTHESIZER] - ERROR: {error_msg}")
            self._update_session_status(
                user_id=user_id,
                status="error",
                current_step=self.THINKING_STEP,
            )
            DM.save()

            return {
                "session_id": current_session_id,
                "status": "error",
                "error": error_msg,
                "response": self.USER_ERROR_MESSAGE,
            }

    def _agent_path(self, user_id: str) -> List[str]:
        return ["Users", user_id, "Agent"]

    def _default_outputs(self) -> Dict[str, List[Any]]:
        return {
            "Generated Images": [],
            "Generated Floor Plans": [],
            "Classified Style": [],
            "Detected Furniture": [],
            "Recommendations": [],
            "Web Searches": [],
            "Extracted Colors": [],
        }

    def _default_agent_record(self, session_id: str) -> Dict[str, Any]:
        return {
            "session_id": session_id,
            "status": self.STATUS_IDLE,
            "current_step": self.THINKING_STEP,
            "steps": [],
            "Outputs": self._default_outputs(),
            "Uploaded Files": [],
            "Pending File Action": None,
        }

    def _normalize_outputs(self, outputs: Any) -> Dict[str, List[Any]]:
        normalized = self._default_outputs()
        if not isinstance(outputs, dict):
            return normalized

        for branch_name in normalized:
            branch_value = outputs.get(branch_name)
            if isinstance(branch_value, list):
                normalized[branch_name] = branch_value

        return normalized

    def _normalize_uploaded_files(self, files: Any) -> List[Dict[str, Any]]:
        if not isinstance(files, list):
            return []

        normalized: List[Dict[str, Any]] = []
        for item in files:
            if not isinstance(item, dict):
                continue

            file_id = item.get("file_id")
            if not isinstance(file_id, str) or not file_id.strip():
                continue

            normalized.append(
                {
                    "file_id": file_id.strip(),
                    "filename": str(item.get("filename") or ""),
                    "content_type": str(item.get("content_type") or ""),
                    "source": str(item.get("source") or "runtime"),
                    "uploaded_at": str(item.get("uploaded_at") or datetime.now().isoformat()),
                    "analysis": item.get("analysis") if isinstance(item.get("analysis"), dict) else {},
                }
            )

        return normalized

    def _normalize_pending_file_action(self, pending_action: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(pending_action, dict):
            return None

        tool_name = pending_action.get("tool_name")
        file_id = pending_action.get("file_id")
        if not isinstance(tool_name, str) or not isinstance(file_id, str):
            return None
        if not tool_name.strip() or not file_id.strip():
            return None

        tool_args = pending_action.get("tool_args")
        if not isinstance(tool_args, dict):
            tool_args = {}

        return {
            "tool_name": tool_name.strip(),
            "tool_args": tool_args,
            "file_id": file_id.strip(),
            "file_description": str(pending_action.get("file_description") or "uploaded file"),
            "required_file_description": str(pending_action.get("required_file_description") or ""),
            "original_query": str(pending_action.get("original_query") or ""),
            "created_at": str(pending_action.get("created_at") or datetime.now().isoformat()),
        }

    def _build_messages(self, query: str, history_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        if history_steps:
            history_payload = []
            for step in history_steps[-self.HISTORY_LIMIT :]:
                if not isinstance(step, dict):
                    continue
                history_payload.append(
                    {
                        "type": step.get("type"),
                        "tool": step.get("tool"),
                        "content": step.get("content"),
                        "timestamp": step.get("timestamp"),
                    }
                )

            if history_payload:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Session history from previous interactions (chronological, "
                            f"last {self.HISTORY_LIMIT} items): "
                            f"{json.dumps(history_payload, ensure_ascii=False)}"
                        ),
                    }
                )

        messages.append({"role": "user", "content": query})
        return messages

    def _safe_parse_tool_args(self, raw_args: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(raw_args or "{}")
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def _normalize_steps(self, steps: Any) -> List[Dict[str, Any]]:
        if not isinstance(steps, list):
            return []

        normalized = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            normalized.append(
                {
                    "type": step.get("type"),
                    "content": step.get("content"),
                    "timestamp": step.get("timestamp", datetime.now().isoformat()),
                    "tool": step.get("tool"),
                }
            )

        return normalized[-self.HISTORY_LIMIT :]

    def _select_legacy_session_id(
        self,
        sessions: Dict[str, Any],
        requested_session_id: Optional[str],
    ) -> Optional[str]:
        if requested_session_id and requested_session_id in sessions:
            return requested_session_id

        if not sessions:
            return None

        sorted_ids = sorted(
            sessions.keys(),
            key=lambda sid: str(sessions.get(sid, {}).get("updated_at", ""))
            if isinstance(sessions.get(sid), dict)
            else "",
            reverse=True,
        )
        return sorted_ids[0] if sorted_ids else None

    def _normalize_session_data(
        self,
        raw_session_data: Any,
        requested_session_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(raw_session_data, dict):
            return None

        if isinstance(raw_session_data.get("sessions"), dict):
            sessions = raw_session_data.get("sessions", {})
            selected_id = self._select_legacy_session_id(sessions, requested_session_id)
            if not selected_id:
                return None

            selected_session = sessions.get(selected_id, {})
            if not isinstance(selected_session, dict):
                selected_session = {}

            normalized = self._default_agent_record(selected_id)
            normalized_current_step = self._sanitize_current_step(selected_session.get("current_step"))
            normalized["current_step"] = normalized_current_step
            normalized["status"] = self._sanitize_status(
                selected_session.get("status", self.STATUS_IDLE),
                current_step=normalized_current_step,
            )
            normalized["steps"] = self._normalize_steps(selected_session.get("steps", []))

            existing_outputs = selected_session.get("Outputs")
            if not isinstance(existing_outputs, dict):
                existing_outputs = raw_session_data.get("Outputs")
            normalized["Outputs"] = self._normalize_outputs(existing_outputs)
            normalized["Uploaded Files"] = self._normalize_uploaded_files(
                selected_session.get("Uploaded Files", raw_session_data.get("Uploaded Files", []))
            )
            normalized["Pending File Action"] = self._normalize_pending_file_action(
                selected_session.get("Pending File Action", raw_session_data.get("Pending File Action"))
            )
            return normalized

        existing_session_id = raw_session_data.get("session_id")
        normalized_session_id = (
            existing_session_id
            if isinstance(existing_session_id, str) and existing_session_id.strip()
            else (requested_session_id or str(uuid.uuid4()))
        )

        normalized = self._default_agent_record(normalized_session_id)
        normalized_current_step = self._sanitize_current_step(raw_session_data.get("current_step"))
        normalized["current_step"] = normalized_current_step
        normalized["status"] = self._sanitize_status(
            raw_session_data.get("status", self.STATUS_IDLE),
            current_step=normalized_current_step,
        )
        normalized["steps"] = self._normalize_steps(raw_session_data.get("steps", []))
        normalized["Outputs"] = self._normalize_outputs(raw_session_data.get("Outputs"))
        normalized["Uploaded Files"] = self._normalize_uploaded_files(raw_session_data.get("Uploaded Files", []))
        normalized["Pending File Action"] = self._normalize_pending_file_action(
            raw_session_data.get("Pending File Action")
        )
        return normalized

    def _get_or_initialize_session(
        self,
        user_id: str,
        requested_session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        existing = DM.peek(self._agent_path(user_id))
        normalized = self._normalize_session_data(existing, requested_session_id)

        if not normalized:
            normalized = self._default_agent_record(requested_session_id or str(uuid.uuid4()))

        DM.data["Users"][user_id]["Agent"] = normalized
        return normalized

    def _get_agent_record(self, user_id: str) -> Dict[str, Any]:
        raw_agent = DM.peek(self._agent_path(user_id))
        normalized = self._normalize_session_data(raw_agent)

        if not normalized:
            normalized = self._default_agent_record(str(uuid.uuid4()))

        DM.data["Users"][user_id]["Agent"] = normalized
        return DM.data["Users"][user_id]["Agent"]

    def _update_session_status(self, user_id: str, status: str, current_step: str) -> None:
        agent_data = self._get_agent_record(user_id)
        normalized_current_step = self._sanitize_current_step(current_step)
        agent_data["current_step"] = normalized_current_step
        agent_data["status"] = self._sanitize_status(
            status,
            current_step=normalized_current_step,
        )
        DM.save()

    def _sanitize_current_step(self, current_step: Any) -> str:
        if isinstance(current_step, str) and current_step in {
            self.THINKING_STEP,
            self.ANALYZING_FILES_STEP,
            self.DONE_STEP,
            *self.TOOL_CURRENT_STEPS.values(),
        }:
            return current_step
        return self.THINKING_STEP

    def _sanitize_status(self, status: Any, current_step: Optional[str] = None) -> str:
        normalized_step = (
            self._sanitize_current_step(current_step)
            if current_step is not None
            else None
        )
        if normalized_step == self.DONE_STEP:
            return self.STATUS_IDLE

        raw_status = str(status).strip().lower() if isinstance(status, str) else ""
        if raw_status in {
            self.STATUS_RUNNING,
            "thinking",
            "executing",
            "analyzing_files",
        }:
            return self.STATUS_RUNNING

        if raw_status in {
            self.STATUS_IDLE,
            "completed",
            "error",
            "awaiting_user_confirmation",
            "waiting_for_file_upload",
        }:
            return self.STATUS_IDLE

        if normalized_step in {self.THINKING_STEP, self.ANALYZING_FILES_STEP, *self.TOOL_CURRENT_STEPS.values()}:
            return self.STATUS_RUNNING

        return self.STATUS_IDLE

    def _tool_current_step(self, tool_name: str) -> str:
        return self.TOOL_CURRENT_STEPS.get(tool_name, self.THINKING_STEP)

    def _add_step(
        self,
        user_id: str,
        step_type: str,
        content: Any,
        tool_name: Optional[str] = None,
    ) -> None:
        agent_data = self._get_agent_record(user_id)

        if not isinstance(agent_data.get("steps"), list):
            agent_data["steps"] = []

        step = {
            "type": step_type,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        if tool_name:
            step["tool"] = tool_name

        agent_data["steps"].append(step)
        agent_data["steps"] = self._normalize_steps(agent_data["steps"])

    def _tool_requires_file(self, tool_name: str) -> bool:
        return tool_name in self.FILE_REQUIRED_TOOLS

    def _required_file_description(self, tool_name: str) -> str:
        return self.FILE_REQUIREMENTS.get(tool_name, "a valid input file")

    def _scope_context(self, user_id: str, limit: int = 8) -> List[Dict[str, Any]]:
        agent_data = self._get_agent_record(user_id)
        steps = agent_data.get("steps", [])
        if not isinstance(steps, list):
            return []

        snippets: List[Dict[str, Any]] = []
        for step in steps[-40:]:
            if not isinstance(step, dict):
                continue

            step_type = step.get("type")
            content = step.get("content")

            if step_type == "user_query" and isinstance(content, str):
                text = content.strip()
                if text:
                    snippets.append({"type": "user_query", "text": text[:300]})
            elif step_type == "tool_call":
                tool_name = step.get("tool")
                if isinstance(tool_name, str):
                    snippets.append({"type": "tool_call", "tool": tool_name})
            elif step_type == "scope_decision" and isinstance(content, dict):
                classification = content.get("classification")
                if isinstance(classification, str):
                    snippets.append(
                        {
                            "type": "scope_decision",
                            "classification": classification,
                            "confidence": content.get("confidence"),
                        }
                    )

        return snippets[-limit:]

    def _parse_scope_classifier_response(self, raw_response: str) -> Optional[Dict[str, Any]]:
        text = str(raw_response or "").strip()
        if not text:
            return None

        candidates = [text]
        start_index = text.find("{")
        end_index = text.rfind("}")
        if start_index >= 0 and end_index > start_index:
            candidates.append(text[start_index : end_index + 1])

        parsed_payload: Optional[Dict[str, Any]] = None
        for candidate in candidates:
            try:
                loaded = json.loads(candidate)
                if isinstance(loaded, dict):
                    parsed_payload = loaded
                    break
            except Exception:
                continue

        if not parsed_payload:
            return None

        raw_classification = str(parsed_payload.get("classification", "")).strip().lower()
        classification_aliases = {
            "interior": "interior_design",
            "interior_design": "interior_design",
            "interior-design": "interior_design",
            "small_talk": "small_talk",
            "small-talk": "small_talk",
            "smalltalk": "small_talk",
            "out_of_scope": "out_of_scope",
            "out-of-scope": "out_of_scope",
            "outofscope": "out_of_scope",
        }
        classification = classification_aliases.get(raw_classification)
        if not classification:
            return None

        confidence_raw = parsed_payload.get("confidence", 0.5)
        try:
            confidence_value = float(confidence_raw)
        except Exception:
            confidence_value = 0.5
        confidence_value = max(0.0, min(1.0, confidence_value))

        explanation = parsed_payload.get("explanation", "")
        if not isinstance(explanation, str):
            explanation = str(explanation)

        return {
            "classification": classification,
            "confidence": confidence_value,
            "explanation": explanation.strip(),
        }

    def _evaluate_request_scope(
        self,
        query: str,
        user_id: Optional[str] = None,
        record_step: bool = False,
    ) -> Dict[str, Any]:
        normalized_query = str(query or "").strip()
        if not normalized_query:
            decision = {
                "classification": "small_talk",
                "confidence": 1.0,
                "explanation": "Empty/neutral input treated as small talk.",
                "allowed": True,
            }
            if record_step and user_id:
                self._add_step(user_id=user_id, step_type="scope_decision", content=decision)
            return decision

        scope_context: List[Dict[str, Any]] = []
        if user_id:
            scope_context = self._scope_context(user_id=user_id, limit=8)

        classifier_prompt = (
            f"{self.SCOPE_CLASSIFIER_INSTRUCTION}\n\n"
            f"Recent conversation context (oldest to newest):\n"
            f"{json.dumps(scope_context, ensure_ascii=False)}\n\n"
            f"Latest user query:\n{normalized_query}"
        )

        parsed = None
        try:
            raw_response = LLMManager.chat(classifier_prompt)
            parsed = self._parse_scope_classifier_response(raw_response)
        except Exception as e:
            Logger.log(f"[AGENT SYNTHESIZER] - WARNING: Scope classifier failed. Error: {e}")

        if not parsed:
            parsed = {
                "classification": "interior_design",
                "confidence": 0.35,
                "explanation": "Fallback applied because scope classifier output was invalid.",
            }

        parsed["allowed"] = parsed["classification"] in {"interior_design", "small_talk"}

        if record_step and user_id:
            self._add_step(
                user_id=user_id,
                step_type="scope_decision",
                content=parsed,
            )

        return parsed

    def _get_session_files(self, user_id: str) -> List[Dict[str, Any]]:
        agent_data = self._get_agent_record(user_id)
        normalized_files = self._normalize_uploaded_files(agent_data.get("Uploaded Files"))
        agent_data["Uploaded Files"] = normalized_files
        return normalized_files

    def _session_file_info(self, user_id: Optional[str], file_id: str) -> Dict[str, Any]:
        if user_id:
            for item in self._get_session_files(user_id=user_id):
                if item.get("file_id") == file_id:
                    return item

        file_obj = self._file_registry.get(file_id)
        return {
            "file_id": file_id,
            "filename": str(getattr(file_obj, "filename", "") or file_id),
            "content_type": str(getattr(file_obj, "content_type", "") or ""),
        }

    async def _validate_tool_file_input(
        self,
        user_id: Optional[str],
        tool_name: str,
        args: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self._tool_requires_file(tool_name):
            return None

        file_id = args.get("file_id")
        if not isinstance(file_id, str) or not file_id.strip():
            needed = self._required_file_description(tool_name)
            return {
                "error": "missing_file",
                "message": f"I need {needed} before I can continue.",
            }

        cleaned_file_id = file_id.strip()
        file_obj = self._file_registry.get(cleaned_file_id)
        if not file_obj:
            needed = self._required_file_description(tool_name)
            return {
                "error": "file_not_found",
                "message": f"I can't access that file in this session. Please upload {needed}.",
            }

        # Strict type validation for these tools:
        # - classify_style, detect_furniture: room/interior photos only
        # - generate_floor_plan: floor plan images only
        if tool_name in {"classify_style", "detect_furniture", "generate_floor_plan"}:
            file_info = self._session_file_info(user_id=user_id, file_id=cleaned_file_id)
            analysis = await self._analyze_uploaded_file_for_tool(
                tool_name=tool_name,
                file_info=file_info,
                file_obj=file_obj,
            )

            if not analysis.get("suitable"):
                filename = str(analysis.get("filename") or self._file_label(cleaned_file_id, user_id=user_id))
                needed = self._required_file_description(tool_name)

                return {
                    "error": "invalid_file_type",
                    "message": (
                        f"I can't use `{filename}` for this step. "
                        f"Please upload {needed}."
                    ),
                }

        return None

    def _upsert_session_file(
        self,
        user_id: str,
        file_id: str,
        file_obj: Optional[Any] = None,
        source: str = "runtime",
    ) -> None:
        if not isinstance(file_id, str) or not file_id.strip():
            return

        cleaned_file_id = file_id.strip()
        agent_data = self._get_agent_record(user_id)
        files = self._normalize_uploaded_files(agent_data.get("Uploaded Files"))

        filename = ""
        content_type = ""
        if file_obj is not None:
            filename = str(getattr(file_obj, "filename", "") or "")
            content_type = str(getattr(file_obj, "content_type", "") or "")

        now_iso = datetime.now().isoformat()
        matched = False
        for item in files:
            if item.get("file_id") != cleaned_file_id:
                continue
            if filename:
                item["filename"] = filename
            if content_type:
                item["content_type"] = content_type
            item["source"] = source or item.get("source", "runtime")
            item["uploaded_at"] = item.get("uploaded_at") or now_iso
            matched = True
            break

        if not matched:
            files.append(
                {
                    "file_id": cleaned_file_id,
                    "filename": filename,
                    "content_type": content_type,
                    "source": source or "runtime",
                    "uploaded_at": now_iso,
                    "analysis": {},
                }
            )

        agent_data["Uploaded Files"] = files

    def _remove_session_file(self, user_id: str, file_id: str) -> None:
        if not isinstance(file_id, str) or not file_id.strip():
            return

        cleaned_file_id = file_id.strip()
        agent_data = self._get_agent_record(user_id)
        files = self._normalize_uploaded_files(agent_data.get("Uploaded Files"))
        agent_data["Uploaded Files"] = [item for item in files if item.get("file_id") != cleaned_file_id]

        pending_action = self._normalize_pending_file_action(agent_data.get("Pending File Action"))
        if pending_action and pending_action.get("file_id") == cleaned_file_id:
            agent_data["Pending File Action"] = None

    def _set_pending_file_action(
        self,
        user_id: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        file_id: str,
        file_description: str,
        required_file_description: str,
        original_query: str,
    ) -> None:
        agent_data = self._get_agent_record(user_id)
        agent_data["Pending File Action"] = {
            "tool_name": tool_name,
            "tool_args": tool_args if isinstance(tool_args, dict) else {},
            "file_id": file_id,
            "file_description": file_description,
            "required_file_description": required_file_description,
            "original_query": original_query,
            "created_at": datetime.now().isoformat(),
        }

    def _clear_pending_file_action(self, user_id: str) -> None:
        agent_data = self._get_agent_record(user_id)
        agent_data["Pending File Action"] = None

    def _get_pending_file_action(self, user_id: str) -> Optional[Dict[str, Any]]:
        agent_data = self._get_agent_record(user_id)
        normalized = self._normalize_pending_file_action(agent_data.get("Pending File Action"))
        agent_data["Pending File Action"] = normalized
        return normalized

    def _ingest_uploaded_files(
        self,
        user_id: str,
        uploaded_files: List[Dict[str, Any]],
        source: str,
    ) -> None:
        if not isinstance(uploaded_files, list):
            return

        for item in uploaded_files:
            if not isinstance(item, dict):
                continue

            file_id = item.get("file_id")
            if not isinstance(file_id, str) or not file_id.strip():
                continue

            file_obj = item.get("file_obj") or item.get("file")
            if file_obj is not None:
                self.register_file(
                    file_id=file_id.strip(),
                    file_obj=file_obj,
                    user_id=user_id,
                    source=source,
                )
            else:
                self._upsert_session_file(
                    user_id=user_id,
                    file_id=file_id.strip(),
                    file_obj=None,
                    source=source,
                )

        DM.save()

    def _resolve_pending_file_action(
        self,
        user_id: str,
        user_reply: str,
        pending_action: Dict[str, Any],
    ) -> Dict[str, Any]:
        normalized_reply = str(user_reply or "").strip().lower()
        affirmative_keywords = (
            "yes",
            "yeah",
            "yep",
            "sure",
            "ok",
            "okay",
            "use it",
            "use that",
            "use this",
            "go with it",
            "go with this",
            "continue",
            "proceed",
            "go ahead",
            "please do",
        )
        negative_keywords = (
            "no",
            "nope",
            "don't",
            "do not",
            "dont",
            "not this",
            "another",
            "different",
            "different one",
            "different file",
            "new file",
            "upload another",
            "use another",
        )

        is_affirmative = any(keyword in normalized_reply for keyword in affirmative_keywords)
        is_negative = any(keyword in normalized_reply for keyword in negative_keywords)

        if is_affirmative and not is_negative:
            self._add_step(
                user_id=user_id,
                step_type="file_confirmation",
                content={
                    "decision": "accepted",
                    "tool": pending_action.get("tool_name"),
                    "file_id": pending_action.get("file_id"),
                },
            )
            self._clear_pending_file_action(user_id=user_id)
            DM.save()

            return {
                "status": "confirmed",
                "effective_query": pending_action.get("original_query") or user_reply,
                "confirmed_action": {
                    "tool_name": pending_action.get("tool_name"),
                    "file_id": pending_action.get("file_id"),
                    "tool_args": pending_action.get("tool_args", {}),
                    "original_query": pending_action.get("original_query") or user_reply,
                },
                "forced_file_context": {
                    "tool_name": pending_action.get("tool_name"),
                    "file_id": pending_action.get("file_id"),
                    "tool_args": pending_action.get("tool_args", {}),
                },
            }

        if is_negative and not is_affirmative:
            self._add_step(
                user_id=user_id,
                step_type="file_confirmation",
                content={
                    "decision": "rejected",
                    "tool": pending_action.get("tool_name"),
                    "file_id": pending_action.get("file_id"),
                },
            )
            self._clear_pending_file_action(user_id=user_id)
            DM.save()

            needed = pending_action.get("required_file_description") or "the required file"
            return {
                "status": "needs_file_upload",
                "response": (
                    "Sure! Please upload the file you'd like me to use instead. "
                    f"I need {needed} before I can continue."
                ),
            }

        file_desc = pending_action.get("file_description") or "that uploaded file"
        return {
            "status": "waiting_for_confirmation",
            "response": (
                f"I can continue with {file_desc}. "
                "Let me know if you'd like me to proceed with it, or if you want to use a different file."
            ),
        }

    def _file_label(self, file_id: Any, user_id: Optional[str] = None) -> str:
        if not isinstance(file_id, str) or not file_id.strip():
            return "uploaded file"

        cleaned_file_id = file_id.strip()

        if user_id:
            for item in self._get_session_files(user_id=user_id):
                if item.get("file_id") != cleaned_file_id:
                    continue
                filename = str(item.get("filename") or "").strip()
                if filename:
                    return filename

        file_obj = self._file_registry.get(cleaned_file_id)
        filename = str(getattr(file_obj, "filename", "") or "").strip()
        if filename:
            return filename

        return cleaned_file_id

    def _build_direct_tool_response(
        self,
        user_id: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        result: Any,
    ) -> str:
        if isinstance(result, dict):
            error_message = result.get("message")
            if isinstance(error_message, str) and error_message.strip() and result.get("error"):
                return error_message.strip()

        if tool_name == "classify_style":
            style = "Unknown"
            confidence_text = ""
            if isinstance(result, dict):
                style = (
                    result.get("detected_style")
                    or result.get("style")
                    or result.get("classified_style")
                    or "Unknown"
                )
                confidence = result.get("confidence")
                if isinstance(confidence, (int, float)):
                    confidence_value = float(confidence)
                    if 0.0 <= confidence_value <= 1.0:
                        confidence_value *= 100.0
                    confidence_text = f" ({confidence_value:.0f}% confidence)"

            source_name = self._file_label(tool_args.get("file_id"), user_id=user_id)
            return (
                f"I used `{source_name}` and detected the style as **{style}**{confidence_text}."
            )

        if tool_name == "detect_furniture":
            detected_furniture = self._extract_furniture_objects(result=result)
            total_items = len(detected_furniture)
            if total_items == 0 and isinstance(result, dict):
                if isinstance(result.get("total_items"), int):
                    total_items = result.get("total_items", 0)
                elif isinstance(result.get("detections"), list):
                    total_items = len(result.get("detections", []))

            source_name = self._file_label(tool_args.get("file_id"), user_id=user_id)
            if detected_furniture:
                detected_names = ", ".join(item.get("name", "") for item in detected_furniture if item.get("name"))
                if detected_names:
                    return (
                        f"I used `{source_name}` and detected {total_items} furniture item(s): {detected_names}."
                    )
            return (
                f"I used `{source_name}` and detected {total_items} furniture item(s)."
            )

        if tool_name == "extract_colors":
            colors: List[str] = []
            if isinstance(result, dict) and isinstance(result.get("colors"), list):
                for item in result.get("colors", []):
                    if isinstance(item, dict):
                        hex_code = item.get("hex")
                        if isinstance(hex_code, str) and hex_code.strip():
                            colors.append(hex_code.strip())

            source_name = self._file_label(tool_args.get("file_id"), user_id=user_id)
            if colors:
                color_text = ", ".join(colors[:5])
                return f"I extracted colors from `{source_name}`: {color_text}."
            return f"I used `{source_name}` and completed color extraction."

        if tool_name == "generate_floor_plan":
            if isinstance(result, dict):
                floor_plan_url = result.get("floor_plan_url")
                if isinstance(floor_plan_url, str) and floor_plan_url.strip():
                    furniture_items = tool_args.get("furniture_list", [])
                    furniture_text = ""
                    if isinstance(furniture_items, list) and furniture_items:
                        furniture_text = ", ".join(str(item) for item in furniture_items if str(item).strip())
                    source_name = self._file_label(tool_args.get("file_id"), user_id=user_id)
                    if furniture_text:
                        return (
                            f"I used `{source_name}` and generated an updated floor plan with {furniture_text}. "
                            f"Result: {floor_plan_url.strip()}"
                        )
                    return (
                        f"I used `{source_name}` and generated the updated floor plan. "
                        f"Result: {floor_plan_url.strip()}"
                    )

            return "I ran the floor plan generation, but I couldn't produce a valid output."

        if tool_name == "generate_image":
            if isinstance(result, dict):
                image_url = result.get("url")
                if isinstance(image_url, str) and image_url.strip():
                    return f"I generated the image. Result: {image_url.strip()}"
            return "I ran image generation, but couldn't produce an image URL."

        if tool_name == "get_recommendations":
            if isinstance(result, dict) and isinstance(result.get("recommendations"), list):
                count = len(result.get("recommendations", []))
                return f"I found {count} recommendation(s) based on your request."
            return "I ran the recommendation search, but couldn't retrieve recommendations."

        if tool_name == "web_search":
            answer = self._extract_search_result(result)
            if isinstance(answer, str) and answer.strip():
                return answer.strip()
            return "I completed the web search."

        if isinstance(result, str) and result.strip():
            return result.strip()
        if isinstance(result, dict) and isinstance(result.get("message"), str):
            message = result.get("message", "").strip()
            if message:
                return message

        return "Done."

    async def _execute_confirmed_file_action(
        self,
        user_id: str,
        current_session_id: str,
        confirmed_action: Dict[str, Any],
    ) -> Dict[str, Any]:
        tool_name = str(confirmed_action.get("tool_name") or "").strip()
        tool_args = confirmed_action.get("tool_args")
        if not isinstance(tool_args, dict):
            tool_args = {}

        file_id = confirmed_action.get("file_id")
        if isinstance(file_id, str) and file_id.strip():
            tool_args["file_id"] = file_id.strip()

        if not tool_name:
            fallback_response = "I couldn't determine which action to continue. Please resend your request."
            self._add_step(user_id=user_id, step_type="response", content=fallback_response)
            self._update_session_status(
                user_id=user_id,
                status="completed",
                current_step=self.DONE_STEP,
            )
            DM.save()
            return {
                "session_id": current_session_id,
                "status": "completed",
                "response": fallback_response,
            }

        if self._tool_requires_file(tool_name):
            file_id_value = tool_args.get("file_id")
            if not isinstance(file_id_value, str) or file_id_value not in self._file_registry:
                needed = self._required_file_description(tool_name)
                upload_message = (
                    "Sure! Please upload the file you'd like me to use instead. "
                    f"I need {needed} before I can continue."
                )
                self._add_step(user_id=user_id, step_type="response", content=upload_message)
                self._update_session_status(
                    user_id=user_id,
                    status="waiting_for_file_upload",
                    current_step=self.DONE_STEP,
                )
                DM.save()
                return {
                    "session_id": current_session_id,
                    "status": "waiting_for_file_upload",
                    "response": upload_message,
                }

        self._update_session_status(
            user_id=user_id,
            status="executing",
            current_step=self._tool_current_step(tool_name),
        )
        self._add_step(
            user_id=user_id,
            step_type="tool_call",
            content=tool_args,
            tool_name=tool_name,
        )

        try:
            result = await self._execute_tool(
                tool_name,
                tool_args,
                user_id=user_id,
            )

            output_summary = self._store_tool_output(
                user_id=user_id,
                tool_name=tool_name,
                args=tool_args,
                result=result,
            )
            if output_summary:
                self._add_step(
                    user_id=user_id,
                    step_type="tool_result",
                    content=output_summary,
                    tool_name=tool_name,
                )

            final_response = self._build_direct_tool_response(
                user_id=user_id,
                tool_name=tool_name,
                tool_args=tool_args,
                result=result,
            )
            self._add_step(
                user_id=user_id,
                step_type="response",
                content=final_response,
            )
            self._update_session_status(
                user_id=user_id,
                status="completed",
                current_step=self.DONE_STEP,
            )
            DM.save()

            return {
                "session_id": current_session_id,
                "status": "completed",
                "response": final_response,
                "iterations": 1,
            }
        except Exception as e:
            error_msg = f"Tool execution failed for {tool_name}: {str(e)}"
            Logger.log(f"[AGENT SYNTHESIZER] - ERROR: {error_msg}")
            self._add_step(
                user_id=user_id,
                step_type="error",
                content={"error": str(e)},
                tool_name=tool_name,
            )
            self._update_session_status(
                user_id=user_id,
                status="error",
                current_step=self.THINKING_STEP,
            )
            DM.save()
            return {
                "session_id": current_session_id,
                "status": "error",
                "error": error_msg,
                "response": self.USER_ERROR_MESSAGE,
            }

    async def _resolve_tool_file_requirements(
        self,
        user_id: str,
        query: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        forced_file_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self._tool_requires_file(tool_name):
            return {"status": "ready", "tool_args": tool_args}

        args = dict(tool_args or {})

        if (
            isinstance(forced_file_context, dict)
            and forced_file_context.get("tool_name") == tool_name
            and isinstance(forced_file_context.get("file_id"), str)
        ):
            merged_args = dict(forced_file_context.get("tool_args") or {})
            merged_args.update(args)
            merged_args["file_id"] = forced_file_context["file_id"]
            return {
                "status": "ready",
                "tool_args": merged_args,
                "applied_forced_context": True,
            }

        provided_file_id = args.get("file_id")
        cleaned_provided_file_id: Optional[str] = None
        if isinstance(provided_file_id, str) and provided_file_id.strip():
            cleaned_provided_file_id = provided_file_id.strip()
            if cleaned_provided_file_id not in self._file_registry:
                needed = self._required_file_description(tool_name)
                return {
                    "status": "needs_file_upload",
                    "response": (
                        "I can see a file ID in the tool input, but I can't access that file in this session. "
                        f"Please upload {needed} and let me know when you're ready."
                    ),
                }

        session_files = self._get_session_files(user_id=user_id)
        candidates_by_id: Dict[str, Dict[str, Any]] = {}

        for item in session_files:
            file_id = item.get("file_id")
            if not isinstance(file_id, str) or file_id not in self._file_registry:
                continue
            candidates_by_id[file_id] = item

        # Ensure file IDs explicitly passed by the model are also analyzed.
        if cleaned_provided_file_id and cleaned_provided_file_id in self._file_registry:
            if cleaned_provided_file_id not in candidates_by_id:
                referenced_file_obj = self._file_registry.get(cleaned_provided_file_id)
                self._upsert_session_file(
                    user_id=user_id,
                    file_id=cleaned_provided_file_id,
                    file_obj=referenced_file_obj,
                    source="tool-args",
                )
                candidates_by_id[cleaned_provided_file_id] = {
                    "file_id": cleaned_provided_file_id,
                    "filename": str(getattr(referenced_file_obj, "filename", "") or ""),
                    "content_type": str(getattr(referenced_file_obj, "content_type", "") or ""),
                    "source": "tool-args",
                    "uploaded_at": datetime.now().isoformat(),
                    "analysis": {},
                }

        candidates = list(candidates_by_id.values())

        if not candidates:
            needed = self._required_file_description(tool_name)
            return {
                "status": "needs_file_upload",
                "response": (
                    "Sure! Before we continue, I'd need you to upload "
                    f"{needed}. Let me know when you're ready."
                ),
            }

        self._update_session_status(
            user_id=user_id,
            status="analyzing_files",
            current_step=self.ANALYZING_FILES_STEP,
        )
        self._add_step(
            user_id=user_id,
            step_type="file_analysis",
            content={
                "tool": tool_name,
                "status": "started",
                "total_files": len(candidates),
            },
        )
        DM.save()

        analysis_results: List[Dict[str, Any]] = []
        for candidate in candidates:
            file_id = candidate.get("file_id")
            file_obj = self._file_registry.get(file_id)
            if not file_obj:
                continue

            analysis = await self._analyze_uploaded_file_for_tool(
                tool_name=tool_name,
                file_info=candidate,
                file_obj=file_obj,
            )
            analysis_results.append(analysis)

        summarized = [
            {
                "file_id": item.get("file_id"),
                "filename": item.get("filename"),
                "description": item.get("description"),
                "predicted_type": item.get("predicted_type"),
                "suitable": item.get("suitable"),
                "score": item.get("score"),
            }
            for item in analysis_results
        ]
        self._add_step(
            user_id=user_id,
            step_type="file_analysis",
            content={
                "tool": tool_name,
                "status": "completed",
                "results": summarized,
            },
        )
        DM.save()

        suitable_files = [
            item for item in analysis_results if item.get("suitable") and item.get("file_id")
        ]
        suitable_files.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)

        selected: Optional[Dict[str, Any]] = None
        if cleaned_provided_file_id:
            selected = next(
                (
                    item
                    for item in suitable_files
                    if item.get("file_id") == cleaned_provided_file_id
                ),
                None,
            )

        if not selected and suitable_files:
            selected = suitable_files[0]

        if not selected:
            needed = self._required_file_description(tool_name)
            return {
                "status": "needs_file_upload",
                "response": (
                    "I checked the uploaded files, but none look suitable for this task. "
                    f"Please upload {needed}, then tell me when to continue."
                ),
            }

        selected_file_id = selected.get("file_id")
        selected_filename = str(selected.get("filename") or self._file_label(selected_file_id, user_id=user_id))
        selected_description = selected.get("description") or "an uploaded image"
        selected_label = f"`{selected_filename}` ({selected_description})"
        needed = self._required_file_description(tool_name)

        self._set_pending_file_action(
            user_id=user_id,
            tool_name=tool_name,
            tool_args=args,
            file_id=selected_file_id,
            file_description=selected_label,
            required_file_description=needed,
            original_query=query,
        )
        self._add_step(
            user_id=user_id,
            step_type="file_confirmation_requested",
            content={
                "tool": tool_name,
                "file_id": selected_file_id,
                "description": selected_description,
                "filename": selected_filename,
                "from_explicit_file_id": bool(cleaned_provided_file_id),
            },
        )
        DM.save()

        response_message = (
            f"I found an uploaded image named `{selected_filename}` and it looks like {selected_description}. "
            "Would you like me to use this image, or would you prefer a different file?"
        )
        if cleaned_provided_file_id and selected_file_id == cleaned_provided_file_id:
            response_message = (
                f"I analysed the files and `{selected_filename}` (the referenced image) looks suitable "
                f"because it appears to be {selected_description}. "
                "Would you like me to use it, or do you want a different file?"
            )
        elif cleaned_provided_file_id and selected_file_id != cleaned_provided_file_id:
            response_message = (
                f"I analysed the files and found a better match than the referenced image: `{selected_filename}` "
                f"({selected_description}). Would you like me to use this one, or do you want a different file?"
            )

        return {
            "status": "ask_confirmation",
            "response": response_message,
        }

    async def _analyze_uploaded_file_for_tool(
        self,
        tool_name: str,
        file_info: Dict[str, Any],
        file_obj: Any,
    ) -> Dict[str, Any]:
        file_id = str(file_info.get("file_id") or "")
        filename = str(file_info.get("filename") or getattr(file_obj, "filename", "") or "")
        content_type = str(file_info.get("content_type") or getattr(file_obj, "content_type", "") or "")

        lower_filename = filename.lower()
        lower_content_type = content_type.lower()
        if not lower_content_type and "." in lower_filename:
            lower_content_type = f"image/{lower_filename.rsplit('.', 1)[-1]}"

        if not lower_content_type.startswith("image/"):
            return {
                "file_id": file_id,
                "filename": filename or file_id,
                "description": f"`{filename or file_id}` is not an image file",
                "suitable": False,
                "score": 0.0,
            }

        try:
            file_bytes = await self._read_file_bytes(file_obj)
            if not file_bytes:
                raise ValueError("Empty file content")

            with Image.open(BytesIO(file_bytes)) as img:
                rgb = img.convert("RGB")
                width, height = rgb.size
                sample_size = (min(width, 256), min(height, 256))
                sample = rgb.resize(sample_size)

                rgb_array = np.asarray(sample, dtype=np.float32)
                gray = np.asarray(sample.convert("L"), dtype=np.float32)
                hsv = np.asarray(sample.convert("HSV"), dtype=np.float32)

                saturation_mean = float(hsv[:, :, 1].mean())
                bright_ratio = float((gray > 215).mean())
                dark_ratio = float((gray < 70).mean())

                grad_x = np.abs(np.diff(gray, axis=1))
                grad_y = np.abs(np.diff(gray, axis=0))
                edge_ratio_x = float((grad_x > 35).mean()) if grad_x.size else 0.0
                edge_ratio_y = float((grad_y > 35).mean()) if grad_y.size else 0.0
                edge_ratio = (edge_ratio_x + edge_ratio_y) / 2.0

                sampled_pixels = rgb_array.reshape(-1, 3)[::8]
                unique_colors = int(np.unique(sampled_pixels.astype(np.uint8), axis=0).shape[0])

            floor_name_hint = any(
                keyword in lower_filename
                for keyword in ("floor", "plan", "layout", "blueprint", "architect", "unit")
            )
            palette_name_hint = any(
                keyword in lower_filename for keyword in ("palette", "swatch", "color", "colour")
            )

            likely_floor_plan = (
                (
                    saturation_mean < 45
                    and bright_ratio > 0.45
                    and edge_ratio > 0.08
                    and dark_ratio < 0.30
                )
                or (
                    floor_name_hint
                    and saturation_mean < 65
                    and bright_ratio > 0.38
                    and edge_ratio > 0.06
                )
            )
            likely_palette = (
                palette_name_hint
                or (unique_colors < 70 and edge_ratio < 0.12 and saturation_mean > 18)
            )
            likely_photo = (
                not likely_floor_plan
                and not likely_palette
                and (saturation_mean > 40 or unique_colors > 130)
                and edge_ratio < 0.48
            )

            predicted_type = "unclear"
            description = "an image with unclear structure"
            if likely_floor_plan:
                predicted_type = "floor_plan"
                description = "an image that looks like a floor plan or architectural layout"
            elif likely_photo:
                predicted_type = "room_photo"
                description = "an image that looks like a room/interior photo"
            elif likely_palette:
                predicted_type = "palette_or_graphic"
                description = "an image that looks like a color palette or graphic"

            if tool_name == "generate_floor_plan":
                suitable = predicted_type == "floor_plan"
                score = 0.93 if suitable else 0.10
            elif tool_name in {"classify_style", "detect_furniture"}:
                suitable = predicted_type == "room_photo"
                score = 0.90 if suitable else 0.10
            elif tool_name == "extract_colors":
                suitable = True
                score = 0.95 if predicted_type == "palette_or_graphic" else 0.75
            else:
                suitable = False
                score = 0.0

            return {
                "file_id": file_id,
                "filename": filename or file_id,
                "description": description,
                "predicted_type": predicted_type,
                "suitable": suitable,
                "score": score,
            }
        except Exception:
            return {
                "file_id": file_id,
                "filename": filename or file_id,
                "description": f"`{filename or file_id}` couldn't be analyzed reliably",
                "predicted_type": "unknown",
                "suitable": False,
                "score": 0.0,
            }

    async def _read_file_bytes(self, file_obj: Any) -> bytes:
        if hasattr(file_obj, "_content") and isinstance(getattr(file_obj, "_content"), (bytes, bytearray)):
            return bytes(getattr(file_obj, "_content"))

        seek_obj = getattr(file_obj, "seek", None)
        tell_obj = getattr(file_obj, "tell", None)
        read_obj = getattr(file_obj, "read", None)

        start_pos = None
        if callable(tell_obj):
            try:
                start_pos = await self._call_maybe_async(tell_obj)
            except Exception:
                start_pos = None
        elif hasattr(file_obj, "file") and callable(getattr(file_obj.file, "tell", None)):
            try:
                start_pos = file_obj.file.tell()
            except Exception:
                start_pos = None

        data: bytes = b""
        if callable(read_obj):
            raw = await self._call_maybe_async(read_obj)
            if isinstance(raw, (bytes, bytearray)):
                data = bytes(raw)
        elif hasattr(file_obj, "file") and callable(getattr(file_obj.file, "read", None)):
            raw = file_obj.file.read()
            if isinstance(raw, (bytes, bytearray)):
                data = bytes(raw)

        if start_pos is not None:
            if callable(seek_obj):
                try:
                    await self._call_maybe_async(seek_obj, start_pos)
                except Exception:
                    pass
            elif hasattr(file_obj, "file") and callable(getattr(file_obj.file, "seek", None)):
                try:
                    file_obj.file.seek(start_pos)
                except Exception:
                    pass

        return data

    async def _call_maybe_async(self, fn: Any, *args: Any) -> Any:
        if not callable(fn):
            return None
        result = fn(*args)
        if inspect.isawaitable(result):
            return await result
        return result

    def _resolve_image_url(self, file_id: Optional[str], result: Any) -> Optional[str]:
        if isinstance(result, dict):
            for key in ("url", "image_url", "image", "floor_plan_url"):
                value = result.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        if isinstance(file_id, str) and file_id.strip():
            cleaned = file_id.strip()
            if cleaned.startswith("http://") or cleaned.startswith("https://"):
                return cleaned

            try:
                return FM.get_optimized_url(cleaned)
            except Exception:
                return None

        return None

    def _extract_urls_from_result(self, tool_name: str, result: Any) -> List[str]:
        urls: List[str] = []

        def add_url(candidate: Any) -> None:
            if not isinstance(candidate, str):
                return
            value = candidate.strip()
            if value:
                urls.append(value)

        if tool_name in {"generate_image", "generate_floor_plan"}:
            if isinstance(result, dict):
                add_url(result.get("url"))
                add_url(result.get("floor_plan_url"))

        elif tool_name == "detect_furniture":
            detections = result.get("detections", []) if isinstance(result, dict) else []
            if isinstance(detections, list):
                for item in detections:
                    if isinstance(item, dict):
                        add_url(item.get("url"))

        elif tool_name in {"get_recommendations"}:
            recommendations = result.get("recommendations", []) if isinstance(result, dict) else []
            if isinstance(recommendations, list):
                for item in recommendations:
                    if isinstance(item, dict):
                        add_url(item.get("url"))
                        add_url(item.get("image"))

        unique_urls = list(dict.fromkeys(urls))
        return unique_urls

    def _extract_furniture_objects(self, result: Any) -> List[Dict[str, str]]:
        if not isinstance(result, dict):
            return []

        detections = result.get("detections", [])
        if not isinstance(detections, list):
            return []

        furniture_objects: List[Dict[str, str]] = []
        seen_pairs = set()

        for item in detections:
            if not isinstance(item, dict):
                continue

            name_candidate = item.get("class") or item.get("name") or item.get("label")
            name = str(name_candidate).strip() if name_candidate is not None else ""
            if not name:
                name = "unknown"

            url_candidate = item.get("url") or item.get("image_url") or item.get("image")
            if not isinstance(url_candidate, str):
                continue

            url = url_candidate.strip()
            if not url:
                continue

            dedupe_key = (name, url)
            if dedupe_key in seen_pairs:
                continue

            seen_pairs.add(dedupe_key)
            furniture_objects.append(
                {
                    "name": name,
                    "url": url,
                }
            )

        return furniture_objects

    def _extract_search_result(self, result: Any) -> str:
        if isinstance(result, str):
            return result

        if isinstance(result, dict):
            for key in ("answer", "result", "response", "message", "error"):
                value = result.get(key)
                if isinstance(value, str):
                    return value
            return json.dumps(result, ensure_ascii=False)

        try:
            return json.dumps(result, ensure_ascii=False)
        except Exception:
            return str(result)

    def _store_tool_output(
        self,
        user_id: str,
        tool_name: str,
        args: Dict[str, Any],
        result: Any,
    ) -> Optional[Dict[str, Any]]:
        branch_name = self.OUTPUT_BRANCHES.get(tool_name)
        if not branch_name:
            return None

        agent_data = self._get_agent_record(user_id)
        outputs = self._normalize_outputs(agent_data.get("Outputs"))

        if not isinstance(outputs.get(branch_name), list):
            outputs[branch_name] = []

        target_list = outputs[branch_name]
        items_added = 0
        tool_result_metadata: Dict[str, Any] = {}

        if tool_name in self.URL_OUTPUT_TOOLS:
            url_list = self._extract_urls_from_result(tool_name=tool_name, result=result)
            for url in url_list:
                target_list.append(url)
                items_added += 1

        elif tool_name == "detect_furniture":
            source_file_id = args.get("file_id")
            source_file_ref = source_file_id.strip() if isinstance(source_file_id, str) and source_file_id.strip() else None
            image_url = None
            if isinstance(result, dict):
                for key in ("image_url", "url", "image"):
                    value = result.get(key)
                    if isinstance(value, str) and value.strip():
                        image_url = value.strip()
                        break
            if not image_url and isinstance(source_file_ref, str) and source_file_ref.startswith(("http://", "https://")):
                image_url = source_file_ref
            furniture_objects = self._extract_furniture_objects(result=result)

            target_list.append(
                {
                    "image_url": image_url,
                    "source_file_id": source_file_ref,
                    "furniture": furniture_objects,
                }
            )
            items_added = 1
            tool_result_metadata["source_file_id"] = source_file_ref
            tool_result_metadata["furniture"] = furniture_objects

        elif tool_name == "classify_style":
            image_url = self._resolve_image_url(args.get("file_id"), result)
            style = "Unknown"
            if isinstance(result, dict):
                style = (
                    result.get("detected_style")
                    or result.get("style")
                    or result.get("classified_style")
                    or "Unknown"
                )

            target_list.append(
                {
                    "image_url": image_url,
                    "style": style,
                }
            )
            items_added = 1

        elif tool_name == "web_search":
            target_list.append(
                {
                    "query": args.get("query", ""),
                    "result": self._extract_search_result(result),
                }
            )
            items_added = 1

        elif tool_name == "extract_colors":
            image_url = self._resolve_image_url(args.get("file_id"), result)
            colors = []
            if isinstance(result, dict) and isinstance(result.get("colors"), list):
                colors = result.get("colors", [])

            target_list.append(
                {
                    "image_url": image_url,
                    "colors": colors,
                }
            )
            items_added = 1

        else:
            return None

        agent_data["Outputs"] = outputs
        summary = {
            "output_branch": f"Outputs/{branch_name}",
            "items_added": items_added,
        }
        summary.update(tool_result_metadata)
        return summary

    async def _execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Any:
        validation_error = await self._validate_tool_file_input(
            user_id=user_id,
            tool_name=tool_name,
            args=args,
        )
        if validation_error:
            return validation_error

        if tool_name == "generate_image":
            return SO.generate_image(prompt=args["prompt"])

        if tool_name == "classify_style":
            file_obj = self._file_registry.get(args["file_id"])
            if not file_obj:
                return {"error": "File not found in registry"}
            return await SO.classify_style(file=file_obj)

        if tool_name == "detect_furniture":
            file_obj = self._file_registry.get(args["file_id"])
            if not file_obj:
                return {"error": "File not found in registry"}
            return await SO.detect_furniture(file=file_obj)

        if tool_name in {"get_recommendations"}:
            return await SO.get_recommendations(
                style=args["style"],
                furniture_name=args["furniture_name"],
            )

        if tool_name == "web_search":
            query = str(args.get("query") or "")
            scope_decision = self._evaluate_request_scope(
                query,
                user_id=user_id,
                record_step=False,
            )
            if scope_decision.get("classification") != "interior_design":
                return {
                    "error": "out_of_scope",
                    "message": self.OUT_OF_SCOPE_MESSAGE,
                }
            return await SO.web_search(query=args["query"])

        if tool_name == "extract_colors":
            file_obj = self._file_registry.get(args["file_id"])
            if not file_obj:
                return {"error": "File not found in registry"}
            return await SO.extract_colors(file=file_obj)

        if tool_name == "generate_floor_plan":
            file_obj = self._file_registry.get(args["file_id"])
            if not file_obj:
                return {"error": "File not found in registry"}
            return await SO.generate_floor_plan(
                file=file_obj,
                furniture_list=args["furniture_list"],
            )

        raise ValueError(f"Unknown tool: {tool_name}")

    def get_session(self, user_id: str, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        raw = DM.peek(self._agent_path(user_id))
        normalized = self._normalize_session_data(raw, requested_session_id=session_id)
        if not normalized:
            return None

        if session_id and normalized.get("session_id") != session_id:
            return None

        return normalized

    def list_sessions(self, user_id: str) -> Optional[Dict[str, Any]]:
        session = self.get_session(user_id=user_id)
        if not session:
            return {}

        active_session_id = session.get("session_id")
        return {active_session_id: session} if active_session_id else {}

    def clear_session(self, user_id: str) -> bool:
        user_data = DM.data["Users"].get(user_id)
        if not isinstance(user_data, dict):
            return True

        if "Agent" in user_data:
            del user_data["Agent"]
            return DM.save()

        return True


AgentSynthesizer = AgentSynthesizerClass()
