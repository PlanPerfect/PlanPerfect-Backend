from typing import Optional, Dict, List, Any
import uuid
import json
from datetime import datetime
from Services import Logger
from Services import DatabaseManager as DM
from Services import ServiceOrchestra as SO
from Services import LLMManager

"""
AgentSynthesizer is an autonomous agentic AI system that:
1. Takes user queries and decides whether to use tools or just chat
2. Implements a ReAct-inspired loop (Think → Act → Observe)
3. Tracks all steps in the database for observability
4. Uses Groq (Llama 3.3 70B) with native function calling via LLMManager
"""

class AgentSynthesizerClass:
    _instance = None

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
                            "description": "The detailed image generation prompt describing what to create"
                        }
                    },
                    "required": ["prompt"]
                }
            }
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
                            "description": "The file ID of the uploaded image to classify"
                        }
                    },
                    "required": ["file_id"]
                }
            }
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
                            "description": "The file ID of the image to analyze for furniture"
                        }
                    },
                    "required": ["file_id"]
                }
            }
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
                            "description": "The interior design style (e.g., 'modern', 'traditional', 'minimalist')"
                        },
                        "furniture_name": {
                            "type": "string",
                            "description": "The type of furniture (e.g., 'sofa', 'chair', 'table')"
                        }
                    },
                    "required": ["style", "furniture_name"]
                }
            }
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
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }
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
                            "description": "The file ID of the image to extract colors from"
                        }
                    },
                    "required": ["file_id"]
                }
            }
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
                            "description": "The file ID of the floor plan image"
                        },
                        "furniture_list": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of furniture items to place on the floor plan (e.g., ['sofa', 'dining table', 'bed'])"
                        }
                    },
                    "required": ["file_id", "furniture_list"]
                }
            }
        }
    ]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._instance._file_registry = {}  # Maps file_id to file objects
        return cls._instance

    def initialize(self):
        """Initialize the agent synthesizer"""
        if self._initialized:
            return

        try:
            # LLMManager should already be initialized by the app
            # We just verify it's ready
            if not LLMManager._initialized:
                raise RuntimeError("LLMManager must be initialized before AgentSynthesizer")

            self._initialized = True
            print("AGENT SYNTHESIZER INITIALIZED. AGENTIC SYSTEM READY.\n")
            print(f"Using agent model: {LLMManager.get_current_agent_model()}\n")
        except Exception as e:
            Logger.log(f"[AGENT SYNTHESIZER] - ERROR: Failed to initialize. Error: {e}")
            raise

    def register_file(self, file_id: str, file_obj: Any) -> None:
        """Register an uploaded file for use in tool calls"""
        self._file_registry[file_id] = file_obj

    def clear_file_registry(self) -> None:
        """Clear all registered files"""
        self._file_registry.clear()

    async def execute(
        self,
        user_id: str,
        query: str,
        session_id: Optional[str] = None,
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Execute an agentic query with tool calling and state tracking.

        Args:
            user_id: The user ID for database tracking
            query: The user's query/request
            session_id: Optional session ID (generates new if not provided)
            max_iterations: Maximum number of think-act cycles

        Returns:
            Dict containing the final response and session metadata
        """
        if not self._initialized:
            raise RuntimeError("AgentSynthesizer not initialized. Call initialize() first.")

        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        # Initialize session in database
        session_path = ["Users", user_id, "Agent", "sessions", session_id]
        DM.set_value(session_path, {
            "query": query,
            "status": "initializing",
            "current_step": "Starting agent execution",
            "steps": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        })
        DM.save()

        try:
            # Build conversation history (OpenAI format for Groq)
            messages = [
                {
                    "role": "system",
                    "content": "You are an AI design assistant with access to tools for image generation, style analysis, furniture detection, recommendations, and more. Use tools when appropriate to help users with their interior design needs. Be helpful, creative, and detailed in your responses."
                },
                {
                    "role": "user",
                    "content": query
                }
            ]

            iteration = 0
            while iteration < max_iterations:
                iteration += 1

                # Update status: thinking
                self._update_session_status(
                    session_path,
                    "thinking",
                    f"Analyzing query and deciding next action (iteration {iteration})"
                )
                self._add_step(session_path, "thought", f"Iteration {iteration}: Thinking...")

                # Call LLM with tools via LLMManager
                try:
                    response = LLMManager.chat_with_tools(
                        messages=messages,
                        tools=self.TOOLS,
                        temperature=0.2,
                        max_tokens=2048
                    )
                except Exception as e:
                    error_msg = f"LLM call failed: {str(e)}"
                    Logger.log(f"[AGENT SYNTHESIZER] - ERROR: {error_msg}")
                    self._update_session_status(session_path, "error", error_msg)
                    DM.save()
                    return {
                        "session_id": session_id,
                        "status": "error",
                        "error": error_msg,
                        "response": "I encountered an error while processing your request."
                    }

                # Check if there are function calls
                tool_calls = response.get("tool_calls", [])
                text_response = response.get("content")

                # If no function calls, we have a final response
                if not tool_calls:
                    final_response = text_response or "I've completed processing your request."

                    self._add_step(session_path, "response", final_response)
                    self._update_session_status(session_path, "completed", "Query completed successfully")
                    DM.save()

                    return {
                        "session_id": session_id,
                        "status": "completed",
                        "response": final_response,
                        "iterations": iteration
                    }

                # Add assistant's response to conversation
                assistant_message = {
                    "role": "assistant",
                    "content": text_response,
                    "tool_calls": tool_calls
                }
                messages.append(assistant_message)

                # Execute function calls
                for tool_call in tool_calls:
                    function_name = tool_call["function"]["name"]
                    function_args = json.loads(tool_call["function"]["arguments"])
                    tool_call_id = tool_call["id"]

                    # Update status: executing tool
                    self._update_session_status(
                        session_path,
                        "executing",
                        f"Executing tool: {function_name}"
                    )
                    self._add_step(session_path, "tool_call", {
                        "tool": function_name,
                        "args": function_args
                    })

                    # Execute the tool
                    try:
                        result = await self._execute_tool(function_name, function_args)

                        self._add_step(session_path, "tool_result", {
                            "tool": function_name,
                            "result": result
                        })

                        # Add tool result to conversation
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": json.dumps(result)
                        })

                    except Exception as e:
                        error_msg = f"Tool execution failed for {function_name}: {str(e)}"
                        Logger.log(f"[AGENT SYNTHESIZER] - ERROR: {error_msg}")

                        self._add_step(session_path, "error", {
                            "tool": function_name,
                            "error": str(e)
                        })

                        # Add error to conversation
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": json.dumps({"error": str(e)})
                        })

                # Update status: observing results
                self._update_session_status(
                    session_path,
                    "observing",
                    "Processing tool results and planning next step"
                )

                # Continue loop to let LLM process results

            # Max iterations reached
            self._update_session_status(
                session_path,
                "completed",
                "Maximum iterations reached"
            )
            DM.save()

            return {
                "session_id": session_id,
                "status": "completed",
                "response": "I've processed your request through multiple steps. The results are available in the session history.",
                "iterations": max_iterations,
                "note": "Maximum iterations reached"
            }

        except Exception as e:
            error_msg = f"Agent execution failed: {str(e)}"
            Logger.log(f"[AGENT SYNTHESIZER] - ERROR: {error_msg}")
            self._update_session_status(session_path, "error", error_msg)
            DM.save()

            return {
                "session_id": session_id,
                "status": "error",
                "error": error_msg,
                "response": "I encountered an error while processing your request."
            }

    async def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Execute a tool from ServiceOrchestra"""

        if tool_name == "generate_image":
            result = SO.generate_image(prompt=args["prompt"])
            return result

        elif tool_name == "classify_style":
            file_obj = self._file_registry.get(args["file_id"])
            if not file_obj:
                return {"error": "File not found in registry"}
            result = await SO.classify_style(file=file_obj)
            return result

        elif tool_name == "detect_furniture":
            file_obj = self._file_registry.get(args["file_id"])
            if not file_obj:
                return {"error": "File not found in registry"}
            result = await SO.detect_furniture(file=file_obj)
            return result

        elif tool_name == "get_recommendations":
            result = await SO.get_recommendations(
                style=args["style"],
                furniture_name=args["furniture_name"]
            )
            return result

        elif tool_name == "web_search":
            result = await SO.web_search(query=args["query"])
            return result

        elif tool_name == "extract_colors":
            file_obj = self._file_registry.get(args["file_id"])
            if not file_obj:
                return {"error": "File not found in registry"}
            result = await SO.extract_colors(file=file_obj)
            return result

        elif tool_name == "generate_floor_plan":
            file_obj = self._file_registry.get(args["file_id"])
            if not file_obj:
                return {"error": "File not found in registry"}
            result = await SO.generate_floor_plan(
                file=file_obj,
                furniture_list=args["furniture_list"]
            )
            return result

        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def _update_session_status(self, session_path: List[str], status: str, current_step: str):
        """Update the session status and current step in database"""
        session_data = DM.peek(session_path)
        if session_data:
            session_data["status"] = status
            session_data["current_step"] = current_step
            session_data["updated_at"] = datetime.now().isoformat()
            DM.set_value(session_path, session_data)

    def _add_step(self, session_path: List[str], step_type: str, content: Any):
        """Add a step to the session history"""
        session_data = DM.peek(session_path)
        if session_data:
            if "steps" not in session_data:
                session_data["steps"] = []

            session_data["steps"].append({
                "type": step_type,
                "content": content,
                "timestamp": datetime.now().isoformat()
            })

            DM.set_value(session_path, session_data)

    def get_session(self, user_id: str, session_id: str) -> Optional[Dict]:
        """Retrieve a session from the database"""
        session_path = ["Users", user_id, "Agent", "sessions", session_id]
        return DM.peek(session_path)

    def list_sessions(self, user_id: str) -> Optional[Dict]:
        """List all sessions for a user"""
        sessions_path = ["Users", user_id, "Agent", "sessions"]
        return DM.peek(sessions_path)


AgentSynthesizer = AgentSynthesizerClass()
