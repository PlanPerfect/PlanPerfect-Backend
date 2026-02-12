from typing import Optional, Dict, List, Any
import uuid
import json
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
    HISTORY_LIMIT = 20
    THINKING_STEP = "Thinking..."
    DONE_STEP = "Done!"
    USER_ERROR_MESSAGE = "I'm sorry, but i'm having trouble completing your request right now. Please try again later."
    TOOL_CURRENT_STEPS = {
        "generate_image": "Generating Image...",
        "classify_style": "Classifying Style...",
        "detect_furniture": "Detecting Furniture...",
        "get_recommendations": "Getting Reccomendations...",
        "web_search": "Searching the web...",
        "extract_colors": "Extracting Colors...",
        "generate_floor_plan": "Generating Floor Plan...",
    }
    SYSTEM_PROMPT = (
        "You are an AI design assistant with access to tools for image generation, "
        "style analysis, furniture detection, recommendations, and more. Use tools "
        "when appropriate to help users with their interior design needs. Be "
        "helpful, creative, and detailed in your responses."
    )

    OUTPUT_BRANCHES = {
        "generate_image": "Generated Images",
        "classify_style": "Classified Style",
        "detect_furniture": "Detected Furniture",
        "get_recommendations": "Reccomendations",
        "web_search": "Web Searches",
        "extract_colors": "Extracted Colors",
        "generate_floor_plan": "Generated Floor Plans",
    }

    URL_OUTPUT_TOOLS = {
        "generate_image",
        "get_recommendations",
        "generate_floor_plan",
    }

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
                "description": "Search the web for current information. Use when user asks about recent events, current data, or information that requires up-to-date knowledge beyond your training.",
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

            self._initialized = True
            print("AGENT SYNTHESIZER INITIALIZED. AGENTIC SYSTEM READY.\n")
            print(f"Using agent model: {LLMManager.get_current_agent_model()}\n")
        except Exception as e:
            Logger.log(f"[AGENT SYNTHESIZER] - ERROR: Failed to initialize. Error: {e}")
            raise

    def register_file(self, file_id: str, file_obj: Any) -> None:
        self._file_registry[file_id] = file_obj

    def clear_file_registry(self) -> None:
        self._file_registry.clear()

    async def execute(
        self,
        user_id: str,
        query: str,
        session_id: Optional[str] = None,
        max_iterations: int = 10,
    ) -> Dict[str, Any]:
        """
        Execute an agentic query with tool calling and state tracking.

        Args:
            user_id: The user ID for database tracking
            query: The user's query/request
            session_id: Optional preferred ID (used only when creating a new user session)
            max_iterations: Maximum number of think-act cycles

        Returns:
            Dict containing the final response and session metadata
        """
        if not self._initialized:
            raise RuntimeError("AgentSynthesizer not initialized. Call initialize() first.")

        session_data = self._get_or_initialize_session(
            user_id=user_id,
            requested_session_id=session_id,
        )
        current_session_id = session_data["session_id"]
        history_steps = session_data.get("steps", [])[-self.HISTORY_LIMIT:]

        self._update_session_status(
            user_id=user_id,
            status="thinking",
            current_step=self.THINKING_STEP,
        )
        self._add_step(user_id=user_id, step_type="user_query", content=query)
        DM.save()

        try:
            messages = self._build_messages(query=query, history_steps=history_steps)

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
                        result = await self._execute_tool(function_name, function_args)

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
            "Reccomendations": [],
            "Web Searches": [],
            "Extracted Colors": [],
        }

    def _default_agent_record(self, session_id: str) -> Dict[str, Any]:
        return {
            "session_id": session_id,
            "status": "idle",
            "current_step": self.THINKING_STEP,
            "steps": [],
            "Outputs": self._default_outputs(),
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
            normalized["status"] = selected_session.get("status", "idle")
            normalized["current_step"] = self._sanitize_current_step(selected_session.get("current_step"))
            normalized["steps"] = self._normalize_steps(selected_session.get("steps", []))

            existing_outputs = selected_session.get("Outputs")
            if not isinstance(existing_outputs, dict):
                existing_outputs = raw_session_data.get("Outputs")
            normalized["Outputs"] = self._normalize_outputs(existing_outputs)
            return normalized

        existing_session_id = raw_session_data.get("session_id")
        normalized_session_id = (
            existing_session_id
            if isinstance(existing_session_id, str) and existing_session_id.strip()
            else (requested_session_id or str(uuid.uuid4()))
        )

        normalized = self._default_agent_record(normalized_session_id)
        normalized["status"] = raw_session_data.get("status", "idle")
        normalized["current_step"] = self._sanitize_current_step(raw_session_data.get("current_step"))
        normalized["steps"] = self._normalize_steps(raw_session_data.get("steps", []))
        normalized["Outputs"] = self._normalize_outputs(raw_session_data.get("Outputs"))
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
        agent_data["status"] = status
        agent_data["current_step"] = self._sanitize_current_step(current_step)
        DM.save()

    def _sanitize_current_step(self, current_step: Any) -> str:
        if isinstance(current_step, str) and current_step in {self.THINKING_STEP, self.DONE_STEP, *self.TOOL_CURRENT_STEPS.values()}:
            return current_step
        return self.THINKING_STEP

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

        if tool_name in self.URL_OUTPUT_TOOLS:
            url_list = self._extract_urls_from_result(tool_name=tool_name, result=result)
            for url in url_list:
                target_list.append(url)
                items_added += 1

        elif tool_name == "detect_furniture":
            image_url = self._resolve_image_url(args.get("file_id"), result)
            furniture_urls = self._extract_urls_from_result(tool_name=tool_name, result=result)
            target_list.append(
                {
                    "image_url": image_url,
                    "furniture": furniture_urls,
                }
            )
            items_added = 1

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
        return {
            "output_branch": f"Outputs/{branch_name}",
            "items_added": items_added,
        }

    async def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Execute a tool from ServiceOrchestra"""

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
