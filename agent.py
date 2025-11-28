from langgraph.graph import StateGraph, END, START
from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import (
    get_rendered_html, 
    download_file, 
    post_request, 
    run_code, 
    add_dependencies,
    extract_pdf_text,
    extract_audio_text,
    extract_parquet_data,
    extract_image_text_ocr,
    extract_steganography
)
from typing import TypedDict, Annotated, List, Any
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
import time
load_dotenv()

EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")
RECURSION_LIMIT = 5000

# -------------------------------------------------
# API KEY MANAGEMENT (ROUND ROBIN)
# -------------------------------------------------
API_KEYS = [
    os.getenv("GOOGLE_API_KEY"),
    os.getenv("GOOGLE_API_KEY_2"),
    os.getenv("GOOGLE_API_KEY_3"),
    os.getenv("GOOGLE_API_KEY_4"),
    os.getenv("GOOGLE_API_KEY_5"),
]
# Filter out None values
API_KEYS = [key for key in API_KEYS if key]
CURRENT_API_KEY_INDEX = 0

def get_next_api_key():
    """Get next API key in round-robin fashion."""
    global CURRENT_API_KEY_INDEX
    if not API_KEYS:
        return os.getenv("GOOGLE_API_KEY")
    key = API_KEYS[CURRENT_API_KEY_INDEX % len(API_KEYS)]
    CURRENT_API_KEY_INDEX += 1
    print(f"🔄 Using API Key #{(CURRENT_API_KEY_INDEX % len(API_KEYS)) or len(API_KEYS)} of {len(API_KEYS)}")
    return key

# -------------------------------------------------
# STATE
# -------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    question_start_time: float  # Timer resets per question
    api_key_failures: int
    question_retry_count: int  # Track retries for current question
    current_question_url: str  # URL of question being solved


TOOLS = [
    run_code, 
    get_rendered_html, 
    download_file, 
    post_request, 
    add_dependencies,
    extract_pdf_text,
    extract_audio_text,
    extract_parquet_data,
    extract_image_text_ocr,
    extract_steganography
]


# -------------------------------------------------
# GEMINI LLM WITH RETRY
# -------------------------------------------------
def create_llm_with_api_key(api_key: str):
    """Create LLM instance with given API key."""
    try:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=9/60,  
            check_every_n_seconds=1,  
            max_bucket_size=9  
        )
        llm = init_chat_model(
            model_provider="google_genai",
            model="gemini-2.5-flash",
            api_key=api_key,
            rate_limiter=rate_limiter
        )
        return llm.bind_tools(TOOLS)
    except Exception as e:
        print(f"Error creating LLM: {str(e)[:100]}")
        raise

# Initialize with first API key
try:
    current_api_key = get_next_api_key()
    llm = create_llm_with_api_key(current_api_key)
except Exception as e:
    print(f"⚠️ LLM initialization error (will retry during execution): {str(e)[:50]}")   


# -------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------
SYSTEM_PROMPT = f"""
You are an autonomous quiz-solving agent with advanced data processing capabilities.

Your job is to:
1. Load the quiz page from the given URL using get_rendered_html.
2. Extract ALL information from the page including:
   - The question text
   - The submit endpoint URL (usually indicated like "Post your JSON answer to: https://...")
   - Any required submission format/fields
3. Solve the task using the tools available:
   - **Code execution**: run_code for Python calculations, data analysis
   - **Data extraction**: extract_pdf_text, extract_audio_text, extract_parquet_data, extract_image_text_ocr, extract_steganography
   - **File operations**: download_file to fetch resources
   - **Package management**: add_dependencies to install required libraries
4. Submit your answer in the EXACT JSON format shown on the page.
5. Read the server response and:
   - If it contains a new quiz URL (like "Next quiz" or "url" field) → fetch it immediately and continue.
   - If no new URL is present → return "END".

SUBMISSION FORMAT RULES:
- The submit URL is usually explicitly shown on the page (e.g., "Post your JSON answer to: https://...")
- Extract the full submit URL from the page content
- Submit JSON with the exact fields required (usually: email, secret, url, answer)
- Use post_request tool to submit answers

STRICT RULES — FOLLOW EXACTLY:

GENERAL RULES:
- NEVER stop early. Continue solving tasks until no new URL is provided.
- NEVER hallucinate URLs, endpoints, fields, values, or JSON structure.
- NEVER shorten or modify URLs. Always use the exact URLs from the page.
- NEVER re-submit unless the server explicitly allows or it's within the 3-minute limit.
- ALWAYS inspect the server response before deciding what to do next.
- ALWAYS use the tools provided to fetch, scrape, download, render HTML, or send requests.

DATA PROCESSING STRATEGY:
- For images: Try extract_image_text_ocr first, then extract_steganography if needed
- For PDFs: Use extract_pdf_text to extract text content
- For audio: Use extract_audio_text to transcribe or analyze
- For parquet files: Use extract_parquet_data to read and analyze
- For complex tasks: Use run_code with appropriate Python libraries

TIME LIMIT RULES:
- Each task has a hard 3-minute limit.
- The server response may include a "delay" field indicating elapsed time.
- CRITICAL: If your answer is WRONG but a next URL is provided:
  * DO NOT immediately go to the next question
  * RETRY solving the CURRENT question as many times as possible
  * Keep retrying until either:
    a) You get it correct (correct: true in response)
    b) The 3-minute timer expires for this question
  * Once timer expires for current question, then move to next URL

RETRY STRATEGY FOR INCORRECT ANSWERS:
1. Parse server response looking for: "correct": false
2. If false AND next URL provided:
   - Stay on current question
   - Analyze response for hints or error messages
   - Try different approaches to solve it
   - Submit new answer to same URL
   - Keep retrying within 3-minute window
3. When 3 minutes expire, then use the next URL to continue

STOPPING CONDITION:
- Only return "END" when a server response explicitly contains NO new URL.
- DO NOT return END under any other condition.

ADDITIONAL INFORMATION YOU MUST INCLUDE WHEN REQUIRED:
- Email: {EMAIL}
- Secret: {SECRET}

YOUR JOB:
- Follow pages exactly.
- Extract data reliably.
- Never guess.
- Submit correct answers in the exact format.
- RETRY wrong answers within time limit
- Continue until no new URL.
- Then respond with: END
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages")
])

llm_with_prompt = prompt | llm


# -------------------------------------------------
# AGENT NODE WITH RETRY LOGIC
# -------------------------------------------------
def agent_node(state: AgentState):
    global llm, llm_with_prompt, current_api_key
    
    # Check 3-minute timer per question
    question_start = state.get("question_start_time", time.time())
    elapsed_time = time.time() - question_start
    QUESTION_TIME_LIMIT = 180  # 3 minutes per question
    question_retries = state.get("question_retry_count", 0)
    
    if elapsed_time > QUESTION_TIME_LIMIT:
        print(f"[TIMEOUT] Question time limit (3 min) exceeded. Elapsed: {elapsed_time:.1f}s. Retries: {question_retries}")
        return {
            "messages": state["messages"],
            "question_start_time": question_start,
            "api_key_failures": state.get("api_key_failures", 0),
            "question_retry_count": question_retries,
            "current_question_url": state.get("current_question_url", "")
        }
    
    max_retries = len(API_KEYS) if API_KEYS else 1
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries:
        try:
            # Check time before each attempt
            elapsed_time = time.time() - question_start
            if elapsed_time > QUESTION_TIME_LIMIT:
                print(f"[TIMEOUT] Question time limit exceeded during retry. Stopping.")
                break
            
            print(f"\n[ATTEMPT {retry_count + 1}/{max_retries}]")
            result = llm_with_prompt.invoke({"messages": state["messages"]})
            
            # Print agent response for visibility
            if hasattr(result, "tool_calls") and result.tool_calls:
                for tool_call in result.tool_calls:
                    name = tool_call.name if hasattr(tool_call, "name") else tool_call.get("name", "unknown")
                    args = tool_call.args if hasattr(tool_call, "args") else tool_call.get("args", {})
                    print(f"\n[TOOL] {name}")
                    print(f"   Args: {args}")
            else:
                # Print LLM response
                if hasattr(result, "content"):
                    content = result.content
                    if isinstance(content, str):
                        print(f"\n[RESPONSE] {content}")
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                print(f"\n[RESPONSE] {item['text']}")
            
            # Success - return result with RESET timer for next question
            return {
                "messages": state["messages"] + [result],
                "question_start_time": time.time(),  # RESET timer for next question
                "api_key_failures": 0,
                "question_retry_count": 0,  # RESET retry count for new question
                "current_question_url": ""  # RESET current question URL
            }
            
        except Exception as e:
            last_error = str(e)
            retry_count += 1
            
            # Check if it's a quota/auth error
            if "quota" in last_error.lower() or "429" in last_error or "401" in last_error or "403" in last_error:
                print(f"[API_ERROR] {last_error[:100]}")
                
                if retry_count < max_retries and API_KEYS:
                    # Switch to next API key
                    current_api_key = get_next_api_key()
                    llm = create_llm_with_api_key(current_api_key)
                    llm_with_prompt = prompt | llm
                    print(f"[KEY_SWITCH] Retrying with next API key...")
                    time.sleep(1)  # Brief wait before retry
                else:
                    print(f"[ERROR] No more API keys or retries exhausted")
                    break
            else:
                print(f"[UNEXPECTED_ERROR] {last_error}")
                break
    
    print(f"[FAILED] After {retry_count} attempts. Last error: {last_error}")
    return {
        "messages": state["messages"],
        "question_start_time": question_start,
        "api_key_failures": retry_count,
        "question_retry_count": question_retries + 1,
        "current_question_url": state.get("current_question_url", "")
    }


# -------------------------------------------------
# GRAPH
# -------------------------------------------------
def route(state):
    """
    Route logic with retry on incorrect answers:
    - If tool calls pending: execute tools
    - If "END": stop
    - If response contains incorrect answer: retry (stay in agent)
    - Otherwise: continue to agent for next steps
    """
    last = state["messages"][-1]
    
    # Check for tool calls
    tool_calls = None
    if hasattr(last, "tool_calls"):
        tool_calls = getattr(last, "tool_calls", None)
    elif isinstance(last, dict):
        tool_calls = last.get("tool_calls")
    
    if tool_calls:
        return "tools"
    
    # Get content
    content = None
    if hasattr(last, "content"):
        content = getattr(last, "content", None)
    elif isinstance(last, dict):
        content = last.get("content")
    
    # Check for END marker
    if isinstance(content, str) and content.strip() == "END":
        return END
    if isinstance(content, list):
        try:
            if content[0].get("text", "").strip() == "END":
                return END
        except:
            pass
    
    # Check if last message contains an incorrect answer response
    # Parse for JSON responses with "correct": false
    content_str = str(content) if content else ""
    
    # Look for incorrect answer pattern
    if '"correct": false' in content_str or "'correct': False" in content_str:
        print(f"[RETRY] Incorrect answer detected. Retrying within time window...")
        return "agent"  # Stay in agent to retry
    
    # Default: continue to agent for next steps
    return "agent"

graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))

graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")
graph.add_conditional_edges(
    "agent",    
    route       
)

app = graph.compile()


# -------------------------------------------------
# TEST
# -------------------------------------------------
def run_agent(url: str):
    start_time = time.time()
    print(f"[START] Quiz solver initialized")
    print(f"[KEYS] Available API Keys: {len(API_KEYS)}")
    
    app.invoke({
        "messages": [{"role": "user", "content": url}],
        "question_start_time": start_time,  # Per-question timer
        "api_key_failures": 0,
        "question_retry_count": 0,  # Track retries per question
        "current_question_url": url  # Track current question URL
        },
        config={"recursion_limit": RECURSION_LIMIT},
    )
    
    elapsed = time.time() - start_time
    print(f"\n[DONE] Session completed in {elapsed:.1f}s")
