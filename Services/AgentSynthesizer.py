from typing import Optional, Dict, List, Any
import uuid
import json
import inspect
import re
import random
from datetime import datetime
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
    HISTORY_LIMIT = 100
    THINKING_STEP = "Thinking..."
    ANALYZING_FILES_STEP = "Analysing files..."
    SUMMARIZING_STEP = "Summarising..."
    DONE_STEP = "Done!"
    USER_ERROR_MESSAGE = "I ran into an issue while working on that request. Please try again in a moment."
    RECOMMENDATION_STYLE_PROMPT = "Sure! I can help you with that. What style would you prefer?"
    OUT_OF_SCOPE_MESSAGE = (
        "I can only help with interior design tasks and questions, but I'm happy to help with anything in that space."
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
        "When image tools require file types, use only files that match the tool requirement. "
        "If files were uploaded in this request/session, do not ask the user to upload again; "
        "use the appropriate tool with the available uploaded file. "
        "When a user asks for furniture recommendations by style and furniture type, call the "
        "`get_recommendations` tool instead of answering directly from general knowledge. "
        "Never include URLs/links or confidence percentages/scores in user-facing responses. "
        "Use a warm, natural, conversational tone instead of robotic phrasing. Keep responses concise, "
        "clear, and helpful."
    )

    RECOMMENDATION_INTENT_KEYWORDS = (
        "recommend",
        "recommendation",
        "suggest",
        "suggestion",
        "options",
        "ideas",
        "find me",
        "show me",
        "looking for",
    )

    RECOMMENDATION_STYLE_REFERENCE_KEYWORDS = (
        "my room style",
        "my room s style",
        "room style",
        "room s style",
        "based on my room style",
        "based on my room s style",
        "based on my style",
        "based on the style",
        "based on that style",
        "based on this style",
        "same style",
        "that style",
        "this style",
    )

    STYLE_ALIASES = {
        "boho": "bohemian",
        "bohemian": "bohemian",
        "modern": "modern",
        "minimalist": "minimalist",
        "scandinavian": "scandinavian",
        "industrial": "industrial",
        "rustic": "rustic",
        "traditional": "traditional",
        "contemporary": "contemporary",
        "coastal": "coastal",
        "farmhouse": "farmhouse",
        "mid century modern": "mid century modern",
        "mid-century modern": "mid century modern",
        "japandi": "japandi",
        "wabi sabi": "wabi sabi",
    }

    FURNITURE_ALIASES = {
        "table": "table",
        "tables": "table",
        "dining table": "dining table",
        "dining tables": "dining table",
        "coffee table": "coffee table",
        "coffee tables": "coffee table",
        "side table": "side table",
        "side tables": "side table",
        "end table": "end table",
        "end tables": "end table",
        "console table": "console table",
        "console tables": "console table",
        "sofa": "sofa",
        "sofas": "sofa",
        "sectional": "sectional",
        "sectionals": "sectional",
        "chair": "chair",
        "chairs": "chair",
        "armchair": "armchair",
        "armchairs": "armchair",
        "bed": "bed",
        "beds": "bed",
        "desk": "desk",
        "desks": "desk",
        "bookshelf": "bookshelf",
        "bookshelves": "bookshelf",
        "cabinet": "cabinet",
        "cabinets": "cabinet",
        "dresser": "dresser",
        "dressers": "dresser",
        "nightstand": "nightstand",
        "nightstands": "nightstand",
        "ottoman": "ottoman",
        "ottomans": "ottoman",
        "bench": "bench",
        "benches": "bench",
        "stool": "stool",
        "stools": "stool",
        "bar stool": "bar stool",
        "bar stools": "bar stool",
        "tv stand": "tv stand",
        "tv stands": "tv stand",
    }

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
        "extract_colors": "a clear interior room photo with visible colors",
        "generate_floor_plan": "a top-down floor plan or architectural layout image",
    }

    SCOPE_CLASSIFIER_INSTRUCTION = (
        "You are a scope classifier for an interior-design assistant. "
        "Classify the latest user query into exactly one category:\n"
        "1) interior_design: The user asks about interior design, rooms, furniture, floor plans, "
        "layouts, decor, styles, colors, design trends, or is continuing an interior design workflow. "
        "Also use this for requests to search for design-related information.\n"
        "2) small_talk: Greetings, pleasantries, chitchat, questions about the ongoing conversation "
        "itself, questions the assistant can answer from conversation context "
        "(e.g. 'what is my name?', 'what did I just ask?', 'can you summarise what we discussed?'), "
        "or simple follow-up clarifications.\n"
        "3) out_of_scope: The user explicitly requests help with topics entirely unrelated to "
        "interior design AND unrelated to the current conversation "
        "(e.g. writing code, medical advice, sports results, cooking recipes unrelated to design).\n\n"
        "Important: When in doubt, prefer interior_design or small_talk over out_of_scope. "
        "Only classify as out_of_scope when the request is clearly and completely unrelated.\n\n"
        "Return JSON only with keys: classification, confidence, explanation.\n"
        "Valid classification values: interior_design, small_talk, out_of_scope.\n"
        "confidence must be a float from 0 to 1."
    )

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
                "description": "Extract the dominant color palette from an interior room image. Returns 5 dominant colors with RGB and hex values.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_id": {
                            "type": "string",
                            "description": "The file ID of the room image to extract colors from",
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
                        "furniture_counts": {
                            "type": "object",
                            "description": "Optional quantities per furniture item. Keys are furniture names and values are counts (default count is 1 when omitted). Example: {'chair': 4, 'sofa': 1}",
                            "additionalProperties": {
                                "type": "integer",
                                "minimum": 1,
                            },
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

            self._initialized = True
            print(f"AGENT SYNTHESIZER INITIALIZED. Model: \033[94m{LLMManager.get_current_agent_model()}\033[0m\n")
        except Exception as e:
            Logger.log(f"[AGENT SYNTHESIZER] - ERROR: Failed to initialize. Error: {e}")
            raise

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
        current_request_file_ids: List[str] = []

        if uploaded_files:
            self._ingest_uploaded_files(
                user_id=user_id,
                uploaded_files=uploaded_files,
                source="request",
            )
            for item in uploaded_files:
                if not isinstance(item, dict):
                    continue
                file_id = item.get("file_id")
                if isinstance(file_id, str) and file_id.strip():
                    current_request_file_ids.append(file_id.strip())
            if current_request_file_ids:
                current_request_file_ids = list(dict.fromkeys(current_request_file_ids))
                refreshed_agent_data = self._get_agent_record(user_id)
                history_steps = refreshed_agent_data.get("steps", [])[-self.HISTORY_LIMIT:]

        self._update_session_status(
            user_id=user_id,
            status="thinking",
            current_step=self.THINKING_STEP,
        )
        self._add_step(user_id=user_id, step_type="user_query", content=query)
        DM.save()

        try:
            effective_query = query

            scope_decision = self._evaluate_request_scope(
                effective_query,
                user_id=user_id,
                record_step=True,
            )

            if scope_decision and not scope_decision.get("allowed", True):
                out_of_scope_response = self._sanitize_user_response(self.OUT_OF_SCOPE_MESSAGE)
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
                    content=out_of_scope_response,
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
                    "response": out_of_scope_response,
                }

            messages = self._build_messages(query=effective_query, history_steps=history_steps)
            completed_tool_runs: List[Dict[str, Any]] = []
            tool_result_cache: Dict[str, Any] = {}

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
                        "response": self._sanitize_user_response(self.USER_ERROR_MESSAGE),
                    }

                tool_calls = response.get("tool_calls", [])
                text_response = response.get("content")

                if not tool_calls and not completed_tool_runs:
                    parsed_recommendation = self._extract_recommendation_args_from_query(
                        query=effective_query,
                        user_id=user_id,
                    )
                    if parsed_recommendation and parsed_recommendation.get("furniture_name"):
                        if not parsed_recommendation.get("style"):
                            style_prompt = self._sanitize_user_response(self.RECOMMENDATION_STYLE_PROMPT)
                            self._add_step(
                                user_id=user_id,
                                step_type="response",
                                content=style_prompt,
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
                                "response": style_prompt,
                                "iterations": iteration,
                            }

                        forced_tool_call = self._build_forced_recommendation_tool_call(
                            user_id=user_id,
                            query=effective_query,
                            iteration=iteration,
                        )
                        if forced_tool_call:
                            tool_calls = [forced_tool_call]
                            text_response = None

                if not tool_calls:
                    if completed_tool_runs:
                        if len(completed_tool_runs) > 1:
                            self._update_session_status(
                                user_id=user_id,
                                status="thinking",
                                current_step=self.SUMMARIZING_STEP,
                            )
                            self._add_step(
                                user_id=user_id,
                                step_type="thought",
                                content=f"Iteration {iteration}: Summarising tool outputs...",
                            )
                            self._add_step(
                                user_id=user_id,
                                step_type="summary",
                                content={
                                    "iteration": iteration,
                                    "tool_runs": len(completed_tool_runs),
                                },
                            )

                        final_response = self._build_tool_run_response(
                            user_id=user_id,
                            completed_tool_runs=completed_tool_runs,
                        )
                    else:
                        final_response = self._sanitize_user_response(
                            text_response or "I've completed processing your request."
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

                    if function_name == "get_recommendations":
                        parsed_recommendation = self._extract_recommendation_args_from_query(
                            query=effective_query,
                            user_id=user_id,
                        ) or {}
                        parsed_furniture = str(parsed_recommendation.get("furniture_name") or "").strip()
                        parsed_style = self._normalize_style_name(parsed_recommendation.get("style"))
                        if parsed_furniture and not parsed_style:
                            style_prompt = self._sanitize_user_response(self.RECOMMENDATION_STYLE_PROMPT)
                            self._add_step(
                                user_id=user_id,
                                step_type="response",
                                content=style_prompt,
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
                                "response": style_prompt,
                                "iterations": iteration,
                            }

                        function_args = self._resolve_recommendation_args(
                            user_id=user_id,
                            query=effective_query,
                            existing_args=function_args,
                        )

                        style_value = self._normalize_style_name(function_args.get("style"))
                        if not style_value:
                            style_prompt = self._sanitize_user_response(self.RECOMMENDATION_STYLE_PROMPT)
                            self._add_step(
                                user_id=user_id,
                                step_type="response",
                                content=style_prompt,
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
                                "response": style_prompt,
                                "iterations": iteration,
                            }
                        function_args["style"] = style_value

                    file_resolution = await self._resolve_tool_file_requirements(
                        user_id=user_id,
                        tool_name=function_name,
                        tool_args=function_args,
                        current_request_file_ids=current_request_file_ids,
                    )
                    resolution_status = file_resolution.get("status")

                    if resolution_status in {"needs_file_upload", "unsuitable_file"}:
                        response_message = file_resolution.get("response")
                        if not isinstance(response_message, str) or not response_message.strip():
                            response_message = self._image_requirement_message(
                                tool_name=function_name,
                                reason="required",
                            )
                        response_message = self._sanitize_user_response(response_message)

                        self._add_step(
                            user_id=user_id,
                            step_type="response",
                            content=response_message,
                        )

                        if resolution_status == "unsuitable_file":
                            status_value = "completed"
                        else:
                            status_value = "waiting_for_file_upload"
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
                        call_cache_key = self._tool_call_cache_key(function_name, function_args)
                        reused_cached_result = call_cache_key in tool_result_cache
                        if reused_cached_result:
                            result = tool_result_cache[call_cache_key]
                            self._add_step(
                                user_id=user_id,
                                step_type="thought",
                                content=f"Iteration {iteration}: Reused cached result for `{function_name}`.",
                            )
                        else:
                            result = await self._execute_tool(
                                function_name,
                                function_args,
                                user_id=user_id,
                                allowed_file_ids=current_request_file_ids,
                            )
                            tool_result_cache[call_cache_key] = result

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

                            completed_tool_runs.append(
                                {
                                    "tool_name": function_name,
                                    "tool_args": function_args,
                                    "result": result,
                                }
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
                        completed_tool_runs.append(
                            {
                                "tool_name": function_name,
                                "tool_args": function_args,
                                "result": {
                                    "error": str(e),
                                    "message": (
                                        f"I ran into an issue while running `{function_name}`."
                                    ),
                                },
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
                "response": self._sanitize_user_response(
                    "I've processed your request through multiple steps. The results are available in Agent Outputs."
                ),
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
                "response": self._sanitize_user_response(self.USER_ERROR_MESSAGE),
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

    def _normalize_for_cache(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, dict):
            return {
                str(key): self._normalize_for_cache(val)
                for key, val in sorted(value.items(), key=lambda item: str(item[0]))
            }

        if isinstance(value, (list, tuple, set)):
            return [self._normalize_for_cache(item) for item in value]

        return str(value)

    def _tool_call_cache_key(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        normalized_args = self._normalize_for_cache(tool_args if isinstance(tool_args, dict) else {})
        try:
            args_text = json.dumps(
                normalized_args,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except Exception:
            args_text = str(normalized_args)
        return f"{str(tool_name or '').strip()}::{args_text}"

    def _normalize_recommendation_query_text(self, query: str) -> str:
        normalized_query = re.sub(r"[^\w\s-]", " ", str(query or "").lower())
        normalized_query = re.sub(r"\s{2,}", " ", normalized_query).strip()
        return normalized_query

    def _query_references_prior_style(self, normalized_query: str) -> bool:
        if not isinstance(normalized_query, str) or not normalized_query.strip():
            return False
        return any(
            keyword in normalized_query
            for keyword in self.RECOMMENDATION_STYLE_REFERENCE_KEYWORDS
        )

    def _normalize_style_name(self, style: Any) -> Optional[str]:
        normalized_style = re.sub(r"\s{2,}", " ", str(style or "").strip().lower())
        if not normalized_style or normalized_style in {"unknown", "n/a", "none", "null"}:
            return None
        return self.STYLE_ALIASES.get(normalized_style, normalized_style)

    def _latest_classified_style(self, user_id: Optional[str]) -> Optional[str]:
        if not isinstance(user_id, str) or not user_id.strip():
            return None

        agent_data = self._get_agent_record(user_id)
        outputs = self._normalize_outputs(agent_data.get("Outputs"))
        style_entries = outputs.get("Classified Style", [])
        if not isinstance(style_entries, list):
            return None

        for entry in reversed(style_entries):
            style_candidate = None
            if isinstance(entry, dict):
                style_candidate = (
                    entry.get("style")
                    or entry.get("detected_style")
                    or entry.get("classified_style")
                )
            elif isinstance(entry, str):
                style_candidate = entry

            normalized_style = self._normalize_style_name(style_candidate)
            if normalized_style:
                return normalized_style

        return None

    def _extract_recommendation_args_from_query(
        self,
        query: str,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, str]]:
        normalized_query = self._normalize_recommendation_query_text(query)
        if not normalized_query:
            return None

        has_intent = any(keyword in normalized_query for keyword in self.RECOMMENDATION_INTENT_KEYWORDS)
        if not has_intent:
            return None

        style = None
        for alias in sorted(self.STYLE_ALIASES.keys(), key=len, reverse=True):
            if re.search(rf"\b{re.escape(alias)}\b", normalized_query):
                style = self.STYLE_ALIASES[alias]
                break
        if not style and self._query_references_prior_style(normalized_query):
            style = self._latest_classified_style(user_id=user_id)

        furniture_name = None
        for alias in sorted(self.FURNITURE_ALIASES.keys(), key=len, reverse=True):
            if re.search(rf"\b{re.escape(alias)}\b", normalized_query):
                furniture_name = self.FURNITURE_ALIASES[alias]
                break

        if not furniture_name:
            return None

        parsed_args: Dict[str, str] = {"furniture_name": furniture_name}
        normalized_style = self._normalize_style_name(style)
        if normalized_style:
            parsed_args["style"] = normalized_style

        return parsed_args

    def _resolve_recommendation_args(
        self,
        user_id: Optional[str],
        query: str,
        existing_args: Dict[str, Any],
    ) -> Dict[str, Any]:
        resolved_args = dict(existing_args or {})
        parsed_args = self._extract_recommendation_args_from_query(
            query=query,
            user_id=user_id,
        ) or {}

        if not isinstance(resolved_args.get("furniture_name"), str) or not resolved_args.get("furniture_name", "").strip():
            furniture_name = parsed_args.get("furniture_name")
            if isinstance(furniture_name, str) and furniture_name.strip():
                resolved_args["furniture_name"] = furniture_name.strip()

        normalized_style = self._normalize_style_name(resolved_args.get("style"))
        if normalized_style:
            resolved_args["style"] = normalized_style
        else:
            style_from_context = self._normalize_style_name(parsed_args.get("style"))
            if style_from_context:
                resolved_args["style"] = style_from_context

        return resolved_args

    def _build_forced_recommendation_tool_call(
        self,
        user_id: str,
        query: str,
        iteration: int,
    ) -> Optional[Dict[str, Any]]:
        args = self._extract_recommendation_args_from_query(query=query, user_id=user_id)
        if not args or not isinstance(args.get("style"), str) or not args.get("style", "").strip():
            return None

        return {
            "id": f"forced_get_recommendations_{iteration}",
            "type": "function",
            "function": {
                "name": "get_recommendations",
                "arguments": json.dumps(args),
            },
        }

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
            self.SUMMARIZING_STEP,
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

        if normalized_step in {
            self.THINKING_STEP,
            self.ANALYZING_FILES_STEP,
            self.SUMMARIZING_STEP,
            *self.TOOL_CURRENT_STEPS.values(),
        }:
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

    def _image_requirement_message(self, tool_name: Optional[str], reason: str = "required") -> str:
        normalized_tool_name = str(tool_name).strip() if isinstance(tool_name, str) else ""
        if reason == "unsuitable":
            if normalized_tool_name == "generate_floor_plan":
                return self._sanitize_user_response(
                    "I'm sorry, but the image you provided does not seem to be an image of a floor-plan."
                )
            return self._sanitize_user_response(
                "I'm sorry, but the image you provided does not seem to be an image of a room."
            )

        if normalized_tool_name == "generate_floor_plan":
            return self._sanitize_user_response(
                "Sure! I can help you with that. I'd need you to upload a clear image of your floor plan."
            )
        return self._sanitize_user_response(
            "Sure! I can help you with that. I'd need you to upload a clear image of your room."
        )

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
        allowed_file_ids: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self._tool_requires_file(tool_name):
            return None

        file_id = args.get("file_id")
        if not isinstance(file_id, str) or not file_id.strip():
            return {
                "error": "missing_file",
                "message": self._image_requirement_message(tool_name=tool_name, reason="required"),
            }

        cleaned_file_id = file_id.strip()
        allowed_file_id_set: Optional[set] = None
        if isinstance(allowed_file_ids, list):
            allowed_file_id_set = {
                item.strip()
                for item in allowed_file_ids
                if isinstance(item, str) and item.strip()
            }
            if not allowed_file_id_set:
                return {
                    "error": "missing_file",
                    "message": self._image_requirement_message(tool_name=tool_name, reason="required"),
                }
            if cleaned_file_id not in allowed_file_id_set:
                return {
                    "error": "file_not_found",
                    "message": self._image_requirement_message(tool_name=tool_name, reason="required"),
                }

        file_obj = self._file_registry.get(cleaned_file_id)
        if not file_obj:
            return {
                "error": "file_not_found",
                "message": self._image_requirement_message(tool_name=tool_name, reason="not_found"),
            }

        if tool_name in {"classify_style", "detect_furniture", "generate_floor_plan", "extract_colors"}:
            file_info = self._session_file_info(user_id=user_id, file_id=cleaned_file_id)
            analysis = await self._analyze_uploaded_file_for_tool(
                tool_name=tool_name,
                file_info=file_info,
                file_obj=file_obj,
            )
            if not analysis.get("suitable"):
                return {
                    "error": "invalid_file_type",
                    "message": self._image_requirement_message(tool_name=tool_name, reason="unsuitable"),
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

    def _strip_links_from_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        cleaned = re.sub(
            r"\[([^\]]+)\]\((?:https?://|www\.)[^)]+\)",
            r"\1",
            text,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"(?:https?://|www\.)\S+", "", cleaned, flags=re.IGNORECASE)
        return cleaned

    def _strip_tool_markup_from_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        cleaned = text
        patterns = [
            r"<\s*function\s*=\s*[a-zA-Z_][\w\-]*\s*>\s*\{.*?\}\s*</\s*function\s*>",
            r"<\s*[a-zA-Z_][\w\-]*\s*>\s*\{.*?\}\s*</\s*function\s*>",
            r"<\s*function_call\s+name\s*=\s*[\"']?[a-zA-Z_][\w\-]*[\"']?\s*>\s*\{.*?\}\s*</\s*function_call\s*>",
            r"^\s*[a-zA-Z_][\w\-]*\s*\(\s*\{.*\}\s*\)\s*$",
        ]
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        return cleaned

    def _strip_confidence_from_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        cleaned = text
        confidence_patterns = [
            r"\(\s*(?:with\s+)?confidence(?:\s*(?:score|level))?\s*(?:is|was|of|:|=)?\s*\d+(?:\.\d+)?\s*%?\s*\)",
            r"\b(?:with\s+)?confidence(?:\s*(?:score|level))?\s*(?:is|was|of|:|=)?\s*\d+(?:\.\d+)?\s*%?\b",
            r"\b\d+(?:\.\d+)?\s*%\s*(?:confidence|certainty)\b",
            r"\bconfidence\s*(?:score|level)?\s*(?:is|was|of|:|=)\s*\d+(?:\.\d+)?\s*%?\b",
            r"\bconfidence\s*(?:score|level)?\s*(?:is|was|of|:|=)\s*(?:very\s+)?(?:high|medium|low)\b",
            r"\bconfidence\s*(?:score|level)\b",
        ]
        for pattern in confidence_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        return cleaned

    def _strip_dangling_link_phrases(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        cleaned = text
        cleaned = re.sub(
            r"(?:^|\s)(?:you can|you may|please)?\s*(?:view|see|find|access)\s+"
            r"(?:it|them|this|that)?\s*(?:here|at)\s*:?\s*$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\b(?:here|link|url)\s*:\s*$", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        cleaned = re.sub(r"[:;]\s*$", "", cleaned).strip()
        return cleaned

    def _sanitize_user_response(self, text: Any) -> str:
        value = str(text or "")
        without_markup = self._strip_tool_markup_from_text(value)
        without_links = self._strip_links_from_text(without_markup)
        without_confidence = self._strip_confidence_from_text(without_links)
        compact = self._strip_dangling_link_phrases(without_confidence)
        compact = re.sub(r"\s{2,}", " ", compact).strip()
        return compact or "Done."

    def _pick_response_template(self, templates: List[str], fallback: str) -> str:
        options = [str(item).strip() for item in templates if isinstance(item, str) and item.strip()]
        if not options:
            return fallback
        return random.choice(options)

    def _natural_join(self, values: List[str]) -> str:
        cleaned_values: List[str] = []
        seen = set()
        for value in values:
            entry = str(value or "").strip()
            if not entry:
                continue
            key = entry.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned_values.append(entry)

        if not cleaned_values:
            return ""
        if len(cleaned_values) == 1:
            return cleaned_values[0]
        if len(cleaned_values) == 2:
            return f"{cleaned_values[0]} and {cleaned_values[1]}"
        return f"{', '.join(cleaned_values[:-1])}, and {cleaned_values[-1]}"

    def _build_tool_run_response(
        self,
        user_id: str,
        completed_tool_runs: List[Dict[str, Any]],
    ) -> str:
        if not completed_tool_runs:
            return self._sanitize_user_response("Done.")

        if len(completed_tool_runs) == 1:
            item = completed_tool_runs[0]
            return self._build_direct_tool_response(
                user_id=user_id,
                tool_name=str(item.get("tool_name") or ""),
                tool_args=item.get("tool_args") if isinstance(item.get("tool_args"), dict) else {},
                result=item.get("result"),
            )

        return self._build_multi_tool_response(
            user_id=user_id,
            completed_tool_runs=completed_tool_runs,
        )

    def _build_multi_tool_response(
        self,
        user_id: str,
        completed_tool_runs: List[Dict[str, Any]],
    ) -> str:
        snippets: List[str] = []
        seen = set()
        seen_tool_calls = set()

        for item in completed_tool_runs:
            tool_name = str(item.get("tool_name") or "").strip()
            if not tool_name:
                continue
            tool_args = item.get("tool_args") if isinstance(item.get("tool_args"), dict) else {}
            tool_call_key = self._tool_call_cache_key(tool_name, tool_args)
            if tool_call_key in seen_tool_calls:
                continue
            seen_tool_calls.add(tool_call_key)
            snippet = self._build_direct_tool_response(
                user_id=user_id,
                tool_name=tool_name,
                tool_args=tool_args,
                result=item.get("result"),
            )
            cleaned_snippet = self._sanitize_user_response(snippet)
            if not cleaned_snippet:
                continue
            dedupe_key = cleaned_snippet.lower()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            snippets.append(cleaned_snippet)

        if not snippets:
            return self._sanitize_user_response("Done.")

        if len(snippets) == 1:
            return self._sanitize_user_response(snippets[0])

        opener = self._pick_response_template(
            [
                "I've completed the requested steps.",
                "All requested steps are done.",
                "Everything you asked for has been completed.",
            ],
            "I've completed the requested steps.",
        )
        closer = self._pick_response_template(
            [
                "Let me know if you want any refinements.",
                "If you'd like changes, I can adjust it.",
                "I can fine-tune any part of this if you want.",
            ],
            "Let me know if you want any refinements.",
        )

        if len(snippets) > 3:
            snippet_text = " ".join(snippets[:3] + ["I also completed the remaining step(s)."])
        else:
            snippet_text = " ".join(snippets)

        return self._sanitize_user_response(f"{opener} {snippet_text} {closer}")

    def _build_direct_tool_response(
        self,
        user_id: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        result: Any,
    ) -> str:
        response_text = "Done."

        def _has_generated_asset(payload: Any) -> bool:
            if not isinstance(payload, dict):
                return False
            for key in ("url", "floor_plan_url", "file_id", "image_url"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return True
            return False

        if isinstance(result, dict):
            error_message = result.get("message")
            if isinstance(error_message, str) and error_message.strip() and result.get("error"):
                response_text = error_message.strip()
                return self._sanitize_user_response(response_text)

        if tool_name == "classify_style":
            style = "unknown"
            if isinstance(result, dict):
                style = (
                    result.get("detected_style")
                    or result.get("style")
                    or result.get("classified_style")
                    or "unknown"
                )
            style = str(style).strip() or "unknown"
            style_lower = style.lower()
            if style_lower == "unknown":
                response_text = self._pick_response_template(
                    [
                        "I couldn't determine a clear room style from the image yet. Let me know if you'd like further assistance!",
                        "I wasn't able to identify a clear style this time. Let me know if you'd like further assistance!",
                        "The style wasn't clear enough for me to classify yet. Let me know if you'd like further assistance!",
                    ],
                    "I couldn't determine a clear room style from the image yet. Let me know if you'd like further assistance!",
                )
                return self._sanitize_user_response(response_text)

            response_text = self._pick_response_template(
                [
                    f"Your room style looks like **{style}**. Let me know if you'd like further assistance!",
                    f"It seems your room is **{style}** in style. Let me know if you'd need help with your room!",
                    f"I'd classify your room as **{style}**. I hope that helps! Let me know if you want to do anything else with your room.",
                ],
                f"Your room style looks like **{style}**. Let me know if you'd like further assistance!",
            )
            return self._sanitize_user_response(response_text)

        if tool_name == "detect_furniture":
            counts_by_name: Dict[str, int] = {}
            detections = result.get("detections", []) if isinstance(result, dict) else []
            if isinstance(detections, list):
                for item in detections:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("class") or item.get("name") or item.get("label") or "").strip()
                    if not name:
                        continue
                    counts_by_name[name] = counts_by_name.get(name, 0) + 1

            if not counts_by_name:
                for item in self._extract_furniture_objects(result=result):
                    name = str(item.get("name") or "").strip()
                    if not name:
                        continue
                    counts_by_name[name] = counts_by_name.get(name, 0) + 1

            total_items = sum(counts_by_name.values())
            if total_items == 0 and isinstance(result, dict):
                if isinstance(result.get("total_items"), int):
                    total_items = max(0, int(result.get("total_items", 0)))
                elif isinstance(result.get("detections"), list):
                    total_items = len(result.get("detections", []))

            def _counted_name(name: str, count: int) -> str:
                clean_name = str(name or "").strip()
                if not clean_name:
                    return ""

                if count <= 1:
                    lowered = clean_name.lower()
                    article = "an" if lowered[:1] in {"a", "e", "i", "o", "u"} else "a"
                    return f"{article} {clean_name}"

                lowered = clean_name.lower()
                if lowered.endswith(("s", "x", "z", "ch", "sh")):
                    plural = f"{clean_name}es"
                elif lowered.endswith("y") and len(clean_name) > 1 and lowered[-2] not in {"a", "e", "i", "o", "u"}:
                    plural = f"{clean_name[:-1]}ies"
                else:
                    plural = f"{clean_name}s"
                return f"{count} {plural}"

            furniture_parts: List[str] = []
            for name, count in counts_by_name.items():
                counted = _counted_name(name=name, count=count)
                if counted:
                    furniture_parts.append(counted)
            furniture_text = self._natural_join(furniture_parts)
            if furniture_text:
                response_text = self._pick_response_template(
                    [
                        f"It seems like your room has {furniture_text}. Let me know if you'd like further assistance!",
                        f"From what I can see, your room includes {furniture_text}. Let me know if you'd need help with your room!",
                        f"I can see {furniture_text} in your room. I hope that helps! Let me know if you want to do anything else with your room.",
                    ],
                    f"It seems like your room has {furniture_text}. Let me know if you'd like further assistance!",
                )
                return self._sanitize_user_response(response_text)

            response_text = self._pick_response_template(
                [
                    f"It looks like I detected {total_items} furniture item(s) in your room. Let me know if you'd like further assistance!",
                    f"I found {total_items} furniture item(s) in your room. Let me know if you'd need help with your room!",
                    f"I detected {total_items} furniture item(s) in your room. I hope that helps! Let me know if you want to do anything else with your room.",
                ],
                f"It looks like I detected {total_items} furniture item(s) in your room. Let me know if you'd like further assistance!",
            )
            return self._sanitize_user_response(response_text)

        if tool_name == "extract_colors":
            colors: List[str] = []
            if isinstance(result, dict) and isinstance(result.get("colors"), list):
                for item in result.get("colors", []):
                    if isinstance(item, dict):
                        hex_code = item.get("hex")
                        if isinstance(hex_code, str) and hex_code.strip():
                            colors.append(hex_code.strip())
                    elif isinstance(item, str) and item.strip():
                        colors.append(item.strip())

            if colors:
                color_text = ", ".join(colors[:5])
                response_text = self._pick_response_template(
                    [
                        f"The dominant colors I found are {color_text}. Let me know if you'd like further assistance!",
                        f"Your palette looks like {color_text}. Let me know if you'd need help with your room!",
                        f"I picked out these key colors: {color_text}. I hope that helps! Let me know if you want to do anything else with your room.",
                    ],
                    f"The dominant colors I found are {color_text}. Let me know if you'd like further assistance!",
                )
                return self._sanitize_user_response(response_text)
            response_text = self._pick_response_template(
                [
                    "I completed color extraction, but no clear palette was returned. Let me know if you'd like further assistance!",
                    "I wasn't able to pull a clear color palette from this image. Let me know if you'd like further assistance!",
                    "I finished the color analysis, but the palette wasn't clear enough to report. Let me know if you'd like further assistance!",
                ],
                "I completed color extraction, but no clear palette was returned. Let me know if you'd like further assistance!",
            )
            return self._sanitize_user_response(response_text)

        if tool_name == "generate_floor_plan":
            if _has_generated_asset(result):
                response_text = self._pick_response_template(
                    [
                        "I've successfully generated your updated floor plan and saved it to the **Agent Ensemble**. Let me know if you'd like further assistance!",
                        "Your new floor plan is ready and stored in the **Agent Ensemble**. Let me know if you'd like further assistance!",
                        "Floor plan generation is complete, and the result is saved in the **Agent Ensemble**. Let me know if you'd like further assistance!",
                    ],
                    "I've successfully generated your updated floor plan and saved it to the **Agent Ensemble**. Let me know if you'd like further assistance!",
                )
                return self._sanitize_user_response(response_text)

            response_text = self._pick_response_template(
                [
                    "I tried generating the floor plan, but couldn't produce a valid output. Let me know if you'd like further assistance!",
                    "I wasn't able to generate a valid floor plan this time. Let me know if you'd like further assistance!",
                    "Floor plan generation started, but no valid output was produced. Let me know if you'd like further assistance!",
                ],
                "I tried generating the floor plan, but couldn't produce a valid output. Let me know if you'd like further assistance!",
            )
            return self._sanitize_user_response(response_text)

        if tool_name == "generate_image":
            if _has_generated_asset(result):
                response_text = self._pick_response_template(
                    [
                        "Your generated image is ready and saved to the **Agent Ensemble**.",
                        "Done. I generated the image and saved it to the **Agent Ensemble**.",
                        "Image generation is complete, and it's now saved to the **Agent Ensemble**.",
                    ],
                    "Your generated image is ready and saved to the **Agent Ensemble**.",
                )
                return self._sanitize_user_response(response_text)

            response_text = self._pick_response_template(
                [
                    "I tried generating the image, but couldn't produce a valid output.",
                    "Image generation didn't return a valid output this time.",
                    "I couldn't generate a valid image from that request.",
                ],
                "I tried generating the image, but couldn't produce a valid output.",
            )
            return self._sanitize_user_response(response_text)

        if tool_name == "get_recommendations":
            if isinstance(result, dict) and isinstance(result.get("recommendations"), list):
                count = len(result.get("recommendations", []))
                recommendation_names: List[str] = []
                for item in result.get("recommendations", []):
                    if not isinstance(item, dict):
                        continue
                    name = (
                        item.get("name")
                        or item.get("title")
                        or item.get("product_name")
                        or item.get("furniture_name")
                    )
                    if isinstance(name, str) and name.strip():
                        recommendation_names.append(name.strip())

                condensed_names = self._natural_join(recommendation_names[:3])
                if condensed_names:
                    response_text = self._pick_response_template(
                        [
                            f"I found {count} recommendation(s) and saved them to the **Agent Ensemble**.",
                            f"I pulled {count} recommendation(s) for you and saved them to the **Agent Ensemble**.",
                            f"Your recommendations are ready and saved to the **Agent Ensemble**.",
                        ],
                        f"I found {count} recommendation(s) and saved them to the **Agent Ensemble**.",
                    )
                else:
                    response_text = self._pick_response_template(
                        [
                            f"I pulled {count} recommendation(s) based on your request.",
                            f"I found {count} recommendation(s) for your request.",
                            f"Your recommendations are ready, with {count} result(s) returned.",
                        ],
                        f"I pulled {count} recommendation(s) based on your request.",
                    )
                return self._sanitize_user_response(response_text)
            response_text = self._pick_response_template(
                [
                    "I tried pulling recommendations, but couldn't retrieve results.",
                    "I wasn't able to retrieve recommendations this time.",
                    "Recommendation retrieval didn't return results for that request.",
                ],
                "I tried pulling recommendations, but couldn't retrieve results.",
            )
            return self._sanitize_user_response(response_text)

        if tool_name == "web_search":
            answer = self._extract_search_result(result)
            if isinstance(answer, str) and answer.strip():
                response_text = self._pick_response_template(
                    [
                        f"Here's what I found: {answer.strip()}",
                        f"I looked that up for you: {answer.strip()}",
                        f"Quick summary from the search: {answer.strip()}",
                    ],
                    answer.strip(),
                )
                return self._sanitize_user_response(response_text)
            response_text = self._pick_response_template(
                [
                    "I finished the web search.",
                    "I completed the web search, but no summary text came back.",
                    "Web search is done.",
                ],
                "I finished the web search.",
            )
            return self._sanitize_user_response(response_text)

        if isinstance(result, str) and result.strip():
            response_text = result.strip()
            return self._sanitize_user_response(response_text)
        if isinstance(result, dict) and isinstance(result.get("message"), str):
            message = result.get("message", "").strip()
            if message:
                response_text = message
                return self._sanitize_user_response(response_text)

        return self._sanitize_user_response(response_text)

    async def _resolve_tool_file_requirements(
        self,
        user_id: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        current_request_file_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if not self._tool_requires_file(tool_name):
            return {"status": "ready", "tool_args": tool_args}

        args = dict(tool_args or {})
        current_request_file_id_set = {
            file_id.strip()
            for file_id in (current_request_file_ids or [])
            if isinstance(file_id, str) and file_id.strip()
        }

        if not current_request_file_id_set:
            return {
                "status": "needs_file_upload",
                "response": self._image_requirement_message(
                    tool_name=tool_name,
                    reason="required",
                ),
            }

        provided_file_id = args.get("file_id")
        cleaned_provided_file_id: Optional[str] = None
        if isinstance(provided_file_id, str) and provided_file_id.strip():
            cleaned_provided_file_id = provided_file_id.strip()
            if cleaned_provided_file_id not in self._file_registry:
                cleaned_provided_file_id = None
            elif cleaned_provided_file_id not in current_request_file_id_set:
                cleaned_provided_file_id = None

        candidates: List[Dict[str, Any]] = []
        for fid in current_request_file_id_set:
            if fid not in self._file_registry:
                continue
            file_info = self._session_file_info(user_id=user_id, file_id=fid)
            candidates.append(file_info)

        if not candidates:
            return {
                "status": "needs_file_upload",
                "response": self._image_requirement_message(
                    tool_name=tool_name,
                    reason="not_found",
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

        suitable_files = [item for item in analysis_results if item.get("suitable") and item.get("file_id")]
        suitable_files.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)

        selected: Optional[Dict[str, Any]] = None
        selection_reason = "auto-selected suitable file from current prompt uploads"

        if cleaned_provided_file_id:
            selected = next(
                (
                    item
                    for item in suitable_files
                    if str(item.get("file_id") or "").strip() == cleaned_provided_file_id
                ),
                None,
            )
            if selected:
                selection_reason = "tool-specified uploaded file from current prompt"

        if not selected and suitable_files:
            selected = suitable_files[0]
            selection_reason = "auto-selected best suitable file from current prompt uploads"

        if not selected:
            return {
                "status": "unsuitable_file",
                "response": self._image_requirement_message(
                    tool_name=tool_name,
                    reason="unsuitable",
                ),
            }

        selected_file_id = str(selected.get("file_id") or "").strip()
        if selected_file_id:
            merged_args = dict(args)
            merged_args["file_id"] = selected_file_id
            self._add_step(
                user_id=user_id,
                step_type="file_selected",
                content={
                    "tool": tool_name,
                    "file_id": selected_file_id,
                    "reason": selection_reason,
                    "filename": str(selected.get("filename") or ""),
                    "description": str(selected.get("description") or ""),
                    "suitable": bool(selected.get("suitable")),
                    "score": selected.get("score"),
                },
            )
            DM.save()
            return {
                "status": "ready",
                "tool_args": merged_args,
            }

        return {
            "status": "needs_file_upload",
            "response": self._image_requirement_message(
                tool_name=tool_name,
                reason="selection_error",
            ),
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

        mime_type = self._infer_image_mime_type(filename=filename, content_type=content_type)

        if not mime_type.startswith("image/"):
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

            required_description = self._required_file_description(tool_name)
            expects_floor_plan = tool_name == "generate_floor_plan"
            expected_scene = (
                "a top-down view of a floor plan"
                if expects_floor_plan
                else "an interior room"
            )
            analysis_prompt = (
                "You are validating an uploaded image for an interior-design assistant.\n"
                "Decide if the image matches the required scene.\n\n"
                f"Tool name: {tool_name}\n"
                f"Required scene: {expected_scene}\n\n"
                "Return JSON only with exactly these keys:\n"
                "{\n"
                '  "predicted_type": "floor_plan|room_photo|other|unclear",\n'
                '  "suitable": true,\n'
                '  "score": 0.0,\n'
                '  "description": "short plain-English description"\n'
                "}\n\n"
                "Rules:\n"
                "- If tool is generate_floor_plan: suitable=true only when the image clearly shows a top-down floor-plan layout.\n"
                "- For all other image tools: suitable=true only when the image clearly shows an interior room scene.\n"
                "- If uncertain, set suitable=false.\n"
                "- score must be between 0.0 and 1.0.\n"
                "- description should mention what is visible.\n"
                "- no links."
            )
            raw_analysis = LLMManager.chat_with_vision(
                prompt=analysis_prompt,
                image_bytes=file_bytes,
                mime_type=mime_type,
                temperature=0.0,
                max_tokens=300,
            )
            parsed_analysis = self._parse_json_object(raw_analysis)
            fallback_text = str(raw_analysis or "").strip().lower()
            if not parsed_analysis:
                yes_no_match = re.search(r"\b(yes|no)\b", fallback_text)
                if yes_no_match:
                    parsed_analysis = {
                        "predicted_type": "floor_plan" if (expects_floor_plan and yes_no_match.group(1) == "yes") else (
                            "room_photo" if (not expects_floor_plan and yes_no_match.group(1) == "yes") else "unclear"
                        ),
                        "suitable": yes_no_match.group(1) == "yes",
                        "score": 0.75 if yes_no_match.group(1) == "yes" else 0.2,
                        "description": str(raw_analysis or "").strip() or "an image with unclear structure",
                    }
                else:
                    raise ValueError("Vision model did not return valid JSON.")

            predicted_type = str(parsed_analysis.get("predicted_type") or "unclear").strip().lower()
            if predicted_type not in {"floor_plan", "room_photo", "other", "unclear"}:
                predicted_type = "unclear"
            if predicted_type == "other":
                predicted_type = "unclear"

            suitable_raw = parsed_analysis.get("suitable")
            if isinstance(suitable_raw, bool):
                suitable = suitable_raw
            else:
                suitable = str(suitable_raw).strip().lower() in {"true", "1", "yes"}

            score_raw = parsed_analysis.get("score", 0.0)
            try:
                score = float(score_raw)
            except Exception:
                score = 0.0
            score = max(0.0, min(1.0, score))

            description_raw = parsed_analysis.get("description")
            if isinstance(description_raw, str) and description_raw.strip():
                description = description_raw.strip()
            else:
                if suitable:
                    description = (
                        "a top-down floor-plan layout with walls and room boundaries"
                        if expects_floor_plan
                        else "an interior room scene with visible layout and furnishings"
                    )
                else:
                    description = (
                        "the image does not appear to show a top-down floor-plan layout"
                        if expects_floor_plan
                        else "the image does not appear to show an interior room"
                    )

            if suitable and expects_floor_plan and predicted_type == "room_photo":
                suitable = False
            if suitable and not expects_floor_plan and predicted_type == "floor_plan":
                suitable = False

            if not suitable and score > 0.49:
                score = 0.49
            if suitable and score < 0.5:
                score = 0.5

            return {
                "file_id": file_id,
                "filename": filename or file_id,
                "description": self._normalize_analysis_description(
                    tool_name=tool_name,
                    description=description,
                    predicted_type=predicted_type,
                    required_description=required_description,
                ),
                "predicted_type": predicted_type,
                "suitable": suitable,
                "score": score,
            }
        except Exception as e:
            Logger.log(
                f"[AGENT SYNTHESIZER] - WARNING: Vision file analysis failed for {filename or file_id}. Error: {str(e)}"
            )
            return {
                "file_id": file_id,
                "filename": filename or file_id,
                "description": f"`{filename or file_id}` couldn't be analyzed reliably",
                "predicted_type": "unknown",
                "suitable": False,
                "score": 0.0,
            }

    def _infer_image_mime_type(self, filename: str, content_type: str) -> str:
        lower_content_type = str(content_type or "").strip().lower()
        if lower_content_type.startswith("image/"):
            return lower_content_type

        lower_filename = str(filename or "").strip().lower()
        extension_to_mime = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp",
            "gif": "image/gif",
            "bmp": "image/bmp",
            "tif": "image/tiff",
            "tiff": "image/tiff",
            "avif": "image/avif",
            "heic": "image/heic",
            "heif": "image/heif",
        }
        if "." in lower_filename:
            extension = lower_filename.rsplit(".", 1)[-1]
            if extension in extension_to_mime:
                return extension_to_mime[extension]
            return f"image/{extension}"

        return ""

    def _parse_json_object(self, raw_value: Any) -> Optional[Dict[str, Any]]:
        text = str(raw_value or "").strip()
        if not text:
            return None

        candidates = [text]
        start_index = text.find("{")
        end_index = text.rfind("}")
        if start_index >= 0 and end_index > start_index:
            candidates.append(text[start_index : end_index + 1])

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue

        return None

    def _normalize_analysis_description(
        self,
        tool_name: str,
        description: str,
        predicted_type: str,
        required_description: str,
    ) -> str:
        cleaned_description = self._strip_links_from_text(str(description or "")).strip()
        if not cleaned_description:
            cleaned_description = "an image with unclear structure"

        lower_description = cleaned_description.lower()
        fallback_by_tool = {
            "classify_style": "a room scene with visible decor, furnishings, and style cues",
            "detect_furniture": "a room scene with visible furniture pieces and layout",
            "generate_floor_plan": "a top-down plan showing walls, room boundaries, and layout lines",
            "extract_colors": "a room scene with visible colors across walls, furniture, and decor",
        }
        fallback_by_type = {
            "room_photo": "a room scene with visible furnishings and decor",
            "floor_plan": "a top-down layout drawing with room and wall structure",
            "palette_or_graphic": "a graphic-style image with clear color blocks and tones",
            "unclear": "an image with unclear structure",
        }
        fallback_description = fallback_by_tool.get(
            tool_name,
            fallback_by_type.get(predicted_type, "an image with unclear structure"),
        )

        required_tokens = set(re.findall(r"[a-z0-9]+", str(required_description or "").lower()))
        description_tokens = set(re.findall(r"[a-z0-9]+", lower_description))
        token_overlap = 0.0
        if required_tokens:
            token_overlap = len(required_tokens & description_tokens) / float(len(required_tokens))

        banned_starts = (
            "clear interior room photo",
            "an interior room photo",
            "a clear room photo",
            "a top down floor plan",
            "a top-down floor plan",
            "an image with visible colors",
            "a valid floor plan",
            "clear interior room",
        )

        looks_like_requirement_echo = (
            lower_description in {
                str(required_description or "").strip().lower(),
                f"a {str(required_description or '').strip().lower()}",
                f"an {str(required_description or '').strip().lower()}",
            }
            or any(lower_description.startswith(prefix) for prefix in banned_starts)
            or (
                token_overlap >= 0.65
                and len(description_tokens) <= max(8, len(required_tokens) + 3)
            )
        )

        if looks_like_requirement_echo:
            return self._lowercase_leading_text(fallback_description)

        return self._lowercase_leading_text(cleaned_description)

    def _lowercase_leading_text(self, text: str) -> str:
        value = str(text or "").strip()
        if not value:
            return value

        return value[:1].lower() + value[1:]

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
        if result is None:
            return ""

        if isinstance(result, str):
            return result

        if isinstance(result, dict):
            for key in ("answer", "result", "response", "message", "error"):
                value = result.get(key)
                if isinstance(value, str):
                    return value
            if not result:
                return ""
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
        allowed_file_ids: Optional[List[str]] = None,
    ) -> Any:
        validation_error = await self._validate_tool_file_input(
            user_id=user_id,
            tool_name=tool_name,
            args=args,
            allowed_file_ids=allowed_file_ids,
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
            style = self._normalize_style_name(args.get("style"))
            if not style:
                return {
                    "error": "missing_style",
                    "message": self.RECOMMENDATION_STYLE_PROMPT,
                }
            furniture_name = str(args.get("furniture_name") or "").strip()
            if not furniture_name:
                return {
                    "error": "missing_furniture_name",
                    "message": "Sure! I can help you with that. Which furniture would you like recommendations for?",
                }
            return await SO.get_recommendations(
                style=style,
                furniture_name=furniture_name,
            )

        if tool_name == "web_search":
            query = str(args.get("query") or "")
            scope_decision = self._evaluate_request_scope(
                query,
                user_id=user_id,
                record_step=False,
            )
            if scope_decision.get("classification") == "out_of_scope":
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
                furniture_counts=args.get("furniture_counts"),
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
