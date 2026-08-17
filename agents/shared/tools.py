from __future__ import annotations

import contextlib
import datetime
import io
import json
import math
import subprocess

import requests
from dotenv import load_dotenv, find_dotenv
from langchain_core.tools import tool
from tavily import TavilyClient
from agents.shared.logging import log_tool_call

# =============================================================================
# Configuration
# =============================================================================

load_dotenv(find_dotenv())

tavily_client = TavilyClient()


# =============================================================================
# Utility Tools
# =============================================================================

@tool
def current_datetime() -> str:
    """
    Get the current date and time.
    Useful for answering time-sensitive questions.
    """
    log_tool_call("current_datetime")

    return datetime.datetime.now().isoformat()


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.

    Examples:
        10 + 20
        sqrt(16)
        sin(0.5)
        log(100)
    """

    log_tool_call(
        "calculator",
        expression=expression,
    )

    allowed_names = {
        k: getattr(math, k)
        for k in dir(math)
        if not k.startswith("_")
    }

    try:
        result = eval(
            expression,
            {"__builtins__": {}},
            allowed_names,
        )

        return str(result)

    except Exception as e:
        return f"Calculation error: {e}"


# =============================================================================
# Tavily Search
# =============================================================================

@tool
def web_search(query: str) -> str:
    """
    Search the web using Tavily.

    Best for:
    - Current events
    - Facts
    - Research
    - Documentation lookup
    """

    log_tool_call(
        "web_search",
        query=query,
    )

    try:
        result = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True,
            include_raw_content=False,
        )

        formatted = {
            "answer": result.get("answer"),
            "results": [
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "content": item.get("content"),
                }
                for item in result.get("results", [])
            ],
        }

        return json.dumps(
            formatted,
            indent=2,
            ensure_ascii=False,
        )

    except Exception as e:
        return f"Search error: {e}"


# =============================================================================
# Tavily Extract
# =============================================================================

@tool
def extract_webpage(url: str) -> str:
    """
    Extract content from a webpage.
    Useful after search when the agent wants details.
    """

    log_tool_call(
        "extract_webpage",
        url=url,
    )

    try:
        result = tavily_client.extract(
            urls=[url],
        )

        pages = []

        for page in result.get("results", []):
            pages.append(
                {
                    "url": page.get("url"),
                    "title": page.get("title"),
                    "content": page.get("raw_content"),
                }
            )

        return json.dumps(
            pages,
            indent=2,
            ensure_ascii=False,
        )

    except Exception as e:
        return f"Extraction error: {e}"


# =============================================================================
# Weather
# =============================================================================

@tool
def weather(city: str) -> str:
    """
    Get current weather for a city.
    Uses Open-Meteo APIs.
    """

    log_tool_call(
        "weather",
        city=city,
    )

    try:
        geo_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": city,
                "count": 1,
            },
            timeout=10,
        )

        geo_data = geo_response.json()

        if not geo_data.get("results"):
            return f"City not found: {city}"

        location = geo_data["results"][0]

        latitude = location["latitude"]
        longitude = location["longitude"]

        weather_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current_weather": True,
            },
            timeout=10,
        )

        weather_data = weather_response.json()

        return json.dumps(
            {
                "city": city,
                "latitude": latitude,
                "longitude": longitude,
                "temperature": weather_data["current_weather"]["temperature"],
                "windspeed": weather_data["current_weather"]["windspeed"],
                "weathercode": weather_data["current_weather"]["weathercode"],
            },
            indent=2,
        )

    except Exception as e:
        return f"Weather error: {e}"


# =============================================================================
# Python REPL
# =============================================================================

@tool
def python_repl(code: str) -> str:
    """
    Execute Python code and return stdout.

    WARNING:
    Use only in trusted environments.
    """

    log_tool_call(
        "python_repl",
        code=code,
    )

    stdout_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_buffer):
            exec(code, {})

        output = stdout_buffer.getvalue()

        return output if output else "Execution completed."

    except Exception as e:
        return f"Python execution error: {e}"


# =============================================================================
# File Read
# =============================================================================

@tool
def read_file(path: str) -> str:
    """
    Read a text file.
    """

    log_tool_call(
        "read_file",
        path=path,
    )

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            return file.read()

    except Exception as e:
        return f"Read error: {e}"


# =============================================================================
# File Write
# =============================================================================

@tool
def write_file(data: str) -> str:
    """
    Write content to a file.

    Input JSON:

    {
      "path": "output.txt",
      "content": "hello world"
    }
    """

    log_tool_call(
        "write_file",
        data=data,
    )

    try:
        payload = json.loads(data)

        path = payload["path"]
        content = payload["content"]

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:
            file.write(content)

        return f"Successfully wrote file: {path}"

    except Exception as e:
        return f"Write error: {e}"


# =============================================================================
# HTTP GET
# =============================================================================

@tool
def http_get(url: str) -> str:
    """
    Make a GET request.
    Useful for APIs.
    """

    log_tool_call(
        "http_get",
        url=url,
    )

    try:
        response = requests.get(
            url,
            timeout=20,
        )

        return response.text[:10000]

    except Exception as e:
        return f"HTTP error: {e}"


# =============================================================================
# Shell Tool
# =============================================================================

@tool
def run_shell(command: str) -> str:
    """
    Execute shell commands.

    WARNING:
    Only expose this in trusted environments.
    """

    log_tool_call(
        "run_shell",
        command=command,
    )

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = (
            result.stdout
            if result.stdout
            else result.stderr
        )

        return output

    except Exception as e:
        return f"Shell error: {e}"


# =============================================================================
# Vector Search Placeholder
# =============================================================================

@tool
def document_search(query: str) -> str:
    """
    Search internal knowledge base.

    Replace with:
    - Chroma
    - Qdrant
    - Pinecone
    - PgVector
    """

    log_tool_call(
        "document_search",
        query=query,
    )

    return (
        f"Document search not implemented. "
        f"Query: {query}"
    )


# =============================================================================
# Tool Groups
# =============================================================================

personal_tools = [
    current_datetime,
    calculator,
    weather,
    web_search,
]

research_tools = [
    web_search,
    extract_webpage,
    document_search,
    current_datetime,
]

coding_tools = [
    python_repl,
    read_file,
    write_file,
]

admin_tools = [
    run_shell,
    http_get,
]

all_tools = [
    current_datetime,
    calculator,
    weather,
    web_search,
    extract_webpage,
    document_search,
    python_repl,
    read_file,
    write_file,
    http_get,
    run_shell,
]