import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
load_dotenv()
from composio import Composio

c = Composio(api_key=os.getenv("COMPOSIO_API_KEY"))
GROQ_TOOL = "COMPOSIO_SEARCH_GROQ_CHAT"
GROQ_VERSION = "20260903_00"
GROQ_MODEL = "openai/gpt-oss-120b"

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RESEARCH_DIR = DATA_DIR / "research"

APPS_FILE = DATA_DIR / "apps.csv"
RESEARCH_FILE = RESEARCH_DIR / "_results.csv"
VERIFICATION_FILE = DATA_DIR / "verification.csv"
FAILURES_FILE = DATA_DIR / "failures.csv"

TOOL_VERSION = "20260903_00"

SEARCH_TOOL = "COMPOSIO_SEARCH_DUCK_DUCK_GO"
FETCH_TOOL = "COMPOSIO_SEARCH_FETCH_URL_CONTENT"
GROQ_TOOL = "COMPOSIO_SEARCH_GROQ_CHAT"

LLM_MODEL = "openai/gpt-oss-120b"

MAX_RETRIES = 3
RETRY_DELAY = 3
REQUEST_DELAY = 1

MAX_DOCUMENTS = 4
MAX_DOCUMENT_CHARS = 9000

MAX_VERIFICATION_DOCUMENTS = 3
MAX_VERIFICATION_CHARS = 7000


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(BASE_DIR / ".env")

COMPOSIO_API_KEY = os.getenv("COMPOSIO_API_KEY")

if not COMPOSIO_API_KEY:
    raise RuntimeError(
        "COMPOSIO_API_KEY not found in .env"
    )

composio = Composio(api_key=COMPOSIO_API_KEY)


# ============================================================
# CSV SCHEMAS
# ============================================================

RESEARCH_FIELDS = [
    "id",
    "app_name",
    "category",
    "description",
    "auth_methods",
    "self_serve_or_gated",
    "credential_access_details",
    "api_surface",
    "api_breadth",
    "mcp_available",
    "mcp_details",
    "buildability_verdict",
    "main_blocker",
    "evidence",
]

VERIFICATION_FIELDS = [
    "id",
    "app_name",
    "field",
    "verified_value",
    "verification_status",
    "verification_source",
    "notes",
]

FAILURE_FIELDS = [
    "id",
    "app_name",
    "category",
    "stage",
    "error",
]


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_text(value):
    if value is None:
        return ""

    return str(value).strip()


def clean_url(url):
    """
    Normalize a URL-like value into a clean URL.

    Handles:
    - plain URLs
    - Markdown links
    - surrounding punctuation
    - escaped characters
    """

    if not url:
        return None

    url = str(url).strip()

    # Markdown URL:
    # [Title](https://example.com)
    markdown_match = re.search(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
        url
    )

    if markdown_match:
        url = markdown_match.group(2)

    # Extract a plain URL embedded in text.
    plain_match = re.search(
        r"https?://[^\s<>\[\]\(\)\"']+",
        url
    )

    if plain_match:
        url = plain_match.group(0)

    # Normalize common escaped characters.
    url = url.replace("\\/", "/")
    url = url.replace("\\_", "_")
    url = url.replace("\\.", ".")
    url = url.replace("\\-", "-")

    url = url.strip("'\"")

    # Remove trailing punctuation.
    url = url.rstrip(")]},.;:'\"")

    if not url.startswith(("http://", "https://")):
        return None

    if "..." in url:
        return None

    try:
        parsed = urlparse(url)

        if not parsed.netloc:
            return None

    except Exception:
        return None

    return url


def normalize_urls(values):
    if values is None:
        return []

    if isinstance(values, str):
        values = re.split(r"[\n,]+", values)

    output = []

    for value in values:
        url = clean_url(value)

        if url and url not in output:
            output.append(url)

    return output


# ============================================================
# RESPONSE TEXT EXTRACTION
# ============================================================

def response_to_text(response):
    """
    Extract useful text from Composio responses.

    Fetch responses commonly contain:

        data
          -> results
              -> content

    Those fields are prioritized over execution metadata.
    """

    if response is None:
        return ""

    if isinstance(response, str):
        return response.strip()

    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(response, dict):

        # First inspect Fetch-style data/results.
        data = response.get("data")

        if isinstance(data, dict):

            results = data.get("results")

            if isinstance(results, list):

                pieces = []

                for item in results:

                    if not isinstance(item, dict):
                        continue

                    content = (
                        item.get("content")
                        or item.get("text")
                        or item.get("markdown")
                        or item.get("body")
                    )

                    if content:
                        pieces.append(
                            str(content).strip()
                        )

                if pieces:
                    return "\n\n".join(pieces).strip()

        # Direct content keys.
        preferred_keys = [
            "content",
            "text",
            "response",
            "output",
            "result",
        ]

        for key in preferred_keys:

            if key not in response:
                continue

            text = response_to_text(
                response[key]
            )

            if text:
                return text

        # Inspect data.
        if "data" in response:

            text = response_to_text(
                response["data"]
            )

            if text:
                return text

        # Recursive fallback.
        ignored_keys = {
            "requestId",
            "successful",
            "error",
            "statuses",
            "composio_execution_message",
        }

        for key, value in response.items():

            if key in ignored_keys:
                continue

            text = response_to_text(value)

            if text:
                return text

        return ""

    # --------------------------------------------------------
    # Lists
    # --------------------------------------------------------

    if isinstance(response, list):

        pieces = []

        for item in response:

            text = response_to_text(item)

            if text:
                pieces.append(text)

        return "\n".join(pieces).strip()

    # --------------------------------------------------------
    # Objects
    # --------------------------------------------------------

    for attribute in [
        "content",
        "text",
        "message",
        "response",
        "output",
        "result",
        "data",
    ]:

        try:

            value = getattr(
                response,
                attribute,
                None
            )

            if value is not None:

                text = response_to_text(value)

                if text:
                    return text

        except Exception:
            pass

    try:
        return str(response).strip()

    except Exception:
        return ""


# ============================================================
# ROBUST JSON PARSER
# ============================================================

def parse_json_response(text):
    """
    Robustly parse JSON returned by the Groq LLM.

    Handles:
    - plain JSON
    - Markdown code fences
    - explanatory text surrounding JSON
    - embedded JSON objects
    - embedded JSON arrays
    - common Markdown-link formatting
    """

    if text is None:
        return None

    if not isinstance(text, str):
        text = response_to_text(text)

    if not text:
        return None

    text = text.strip()

    # --------------------------------------------------------
    # Remove code fences
    # --------------------------------------------------------

    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```\s*$",
        "",
        text
    )

    text = text.strip()

    # --------------------------------------------------------
    # Direct JSON
    # --------------------------------------------------------

    try:
        return json.loads(text)

    except (
        json.JSONDecodeError,
        TypeError,
        ValueError
    ):
        pass

    # --------------------------------------------------------
    # Balanced JSON extraction
    # --------------------------------------------------------

    def extract_balanced_json(source):

        for start, opening in enumerate(source):

            if opening not in "{[":
                continue

            closing = "}" if opening == "{" else "]"

            depth = 0
            in_string = False
            escape = False

            for index in range(
                start,
                len(source)
            ):

                char = source[index]

                if escape:
                    escape = False
                    continue

                if char == "\\" and in_string:
                    escape = True
                    continue

                if char == '"':
                    in_string = not in_string
                    continue

                if in_string:
                    continue

                if char == opening:
                    depth += 1

                elif char == closing:

                    depth -= 1

                    if depth == 0:

                        candidate = source[
                            start:index + 1
                        ]

                        try:
                            return json.loads(
                                candidate
                            )

                        except (
                            json.JSONDecodeError,
                            TypeError,
                            ValueError
                        ):
                            break

        return None

    parsed = extract_balanced_json(text)

    if parsed is not None:
        return parsed

    # --------------------------------------------------------
    # Markdown-link cleanup
    # --------------------------------------------------------

    repaired = re.sub(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
        r"\2",
        text
    )

    repaired = repaired.replace(
        "```json",
        ""
    )

    repaired = repaired.replace(
        "```",
        ""
    )

    parsed = extract_balanced_json(repaired)

    if parsed is not None:
        return parsed

    # --------------------------------------------------------
    # Final object fallback
    # --------------------------------------------------------

    start = repaired.find("{")
    end = repaired.rfind("}")

    if start >= 0 and end > start:

        candidate = repaired[
            start:end + 1
        ].strip()

        try:
            return json.loads(candidate)

        except Exception:
            pass

    return None


# ============================================================
# GROQ
# ============================================================

def groq_chat(messages, max_tokens=1500, temperature=0):
    """
    Call Groq through Composio Search and return the model's
    generated message content.
    """
    response = composio.tools.execute(
        GROQ_TOOL,
        {
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        version=GROQ_VERSION,
    )

    # Groq chat completion response
    if isinstance(response, dict):
        data = response.get("data", {})

        if isinstance(data, dict):
            choices = data.get("choices", [])

            if choices:
                message = choices[0].get("message", {})

                if isinstance(message, dict):
                    content = message.get("content")

                    if content:
                        return str(content).strip()

    # Fallback for other Composio response formats
    return response_to_text(response)
# ============================================================
# LOAD APPS
# ============================================================

def load_apps():

    if not APPS_FILE.exists():
        raise FileNotFoundError(
            f"Missing dataset: {APPS_FILE}"
        )

    with open(
        APPS_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        rows = []

        for row in reader:

            rows.append({
                "id": clean_text(
                    row.get("id")
                ),
                "app_name": clean_text(
                    row.get("app_name")
                ),
                "category": clean_text(
                    row.get("category")
                ),
                "website_or_docs_hint":
                    clean_text(
                        row.get(
                            "website_or_docs_hint"
                        )
                    ),
            })

        return rows


# ============================================================
# LOAD RESEARCH
# ============================================================

def load_existing_research():

    if not RESEARCH_FILE.exists():
        return []

    with open(
        RESEARCH_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        return [
            {
                field: row.get(
                    field,
                    ""
                )
                for field in RESEARCH_FIELDS
            }
            for row in reader
        ]


# ============================================================
# LOAD FAILURES
# ============================================================

def load_existing_failures():

    if not FAILURES_FILE.exists():
        return []

    with open(
        FAILURES_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        return [
            {
                field: row.get(
                    field,
                    ""
                )
                for field in FAILURE_FIELDS
            }
            for row in reader
        ]


# ============================================================
# SAVE RESEARCH
# ============================================================

def save_research_results(rows):

    RESEARCH_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    deduplicated = {}

    for row in rows:

        app_id = str(
            row.get("id", "")
        )

        if app_id:
            deduplicated[app_id] = row

    ordered_rows = sorted(
        deduplicated.values(),
        key=lambda x: int(
            x.get("id", 0)
        )
    )

    with open(
        RESEARCH_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=RESEARCH_FIELDS
        )

        writer.writeheader()

        for row in ordered_rows:

            output = {}

            for field in RESEARCH_FIELDS:

                value = row.get(
                    field,
                    ""
                )

                if field in {
                    "auth_methods",
                    "api_surface",
                    "evidence",
                }:

                    if isinstance(value, list):

                        value = json.dumps(
                            value,
                            ensure_ascii=False
                        )

                output[field] = value

            writer.writerow(output)


# ============================================================
# SAVE VERIFICATION
# ============================================================

def save_verification_results(rows):

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        VERIFICATION_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=VERIFICATION_FIELDS
        )

        writer.writeheader()

        for row in rows:

            writer.writerow({
                field: row.get(
                    field,
                    ""
                )
                for field in VERIFICATION_FIELDS
            })


# ============================================================
# SAVE FAILURES
# ============================================================

def save_failures(rows):

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        FAILURES_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FAILURE_FIELDS
        )

        writer.writeheader()

        for row in rows:

            writer.writerow({
                field: row.get(
                    field,
                    ""
                )
                for field in FAILURE_FIELDS
            })


# ============================================================
# SEARCH
# ============================================================

def web_search(query):

    return composio.tools.execute(
        SEARCH_TOOL,
        {
            "query": query,
            "start": 0,
        },
        version=TOOL_VERSION,
    )


# ============================================================
# SEARCH URL FILTERING
# ============================================================

def is_bad_search_url(url):

    if not url:
        return True

    lower = url.lower()

    blocked_domains = [
        "external-content.duckduckgo.com",
        "duckduckgo.com",
        "bing.com",
        "google.com",
    ]

    try:

        domain = urlparse(
            url
        ).netloc.lower()

    except Exception:
        return True

    for blocked in blocked_domains:

        if (
            domain == blocked
            or domain.endswith(
                "." + blocked
            )
        ):
            return True

    blocked_patterns = [
        "/ip3/",
        "/favicon",
        ".ico",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".webp",
        "image?",
        "thumbnail",
        "tracking",
        "utm_",
    ]

    for pattern in blocked_patterns:

        if pattern in lower:
            return True

    return False


# ============================================================
# EXTRACT SEARCH URLS
# ============================================================

def extract_search_urls(search_response):

    text = response_to_text(
        search_response
    )

    raw = str(
        search_response
    )

    combined = (
        text +
        "\n" +
        raw
    )

    markdown_urls = re.findall(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
        combined
    )

    plain_urls = re.findall(
        r"https?://[^\s<>\[\]\(\)\"']+",
        combined
    )

    result = []

    # Markdown URLs
    for _, url in markdown_urls:

        url = clean_url(url)

        if not url:
            continue

        if is_bad_search_url(url):
            continue

        if url not in result:
            result.append(url)

    # Plain URLs
    for url in plain_urls:

        url = clean_url(url)

        if not url:
            continue

        if is_bad_search_url(url):
            continue

        if url not in result:
            result.append(url)

    return result


# ============================================================
# DOMAIN HELPERS
# ============================================================

def extract_domains_from_hint(hint):

    if not hint:
        return []

    domains = []

    urls = re.findall(
        r"https?://[^\s,;]+",
        hint
    )

    for raw_url in urls:

        url = clean_url(raw_url)

        if not url:
            continue

        try:

            domain = urlparse(
                url
            ).netloc.lower()

            if domain.startswith("www."):
                domain = domain[4:]

            if domain and domain not in domains:
                domains.append(domain)

        except Exception:
            pass

    if not domains:

        for token in re.split(
            r"[\s,;]+",
            hint
        ):

            token = token.strip()

            if "." not in token:
                continue

            token = token.replace(
                "https://",
                ""
            )

            token = token.replace(
                "http://",
                ""
            )

            token = token.split("/")[0]

            if token.startswith("www."):
                token = token[4:]

            token = token.lower()

            if token not in domains:
                domains.append(token)

    return domains


def base_domain(domain):

    domain = domain.lower().strip()

    if domain.startswith("www."):
        domain = domain[4:]

    parts = domain.split(".")

    if len(parts) >= 2:
        return ".".join(parts[-2:])

    return domain


# ============================================================
# SAFE ALIAS FALLBACK
# ============================================================

def get_alias_domains(app_name):

    aliases = {

        "freshdesk": [
            "freshworks.com",
            "freshdesk.com",
        ],

        "zoho cliq": [
            "zoho.com",
        ],

        "lark (larksuite)": [
            "larksuite.com",
            "larkoffice.com",
        ],

        "salesforce commerce cloud": [
            "salesforce.com",
        ],

        "datadog": [
            "datadoghq.com",
        ],

        "jira": [
            "atlassian.com",
        ],

        "devin": [
            "devin.ai",
            "cognition.ai",
        ],

        "otter ai": [
            "otter.ai",
        ],

        "mailchimp": [
            "mailchimp.com",
        ],

        "shopify": [
            "shopify.com",
            "shopify.dev",
        ],

        "hubspot": [
            "hubspot.com",
        ],

        "zendesk": [
            "zendesk.com",
        ],

        "bigcommerce": [
            "bigcommerce.com",
        ],

        "supabase": [
            "supabase.com",
        ],

        "snowflake": [
            "snowflake.com",
        ],

        "smartsheet": [
            "smartsheet.com",
        ],

        "asana": [
            "asana.com",
        ],

        "stripe": [
            "stripe.com",
        ],

        "plaid": [
            "plaid.com",
        ],

        "binance": [
            "binance.com",
        ],

        "xero": [
            "xero.com",
        ],

        "ramp": [
            "ramp.com",
        ],

        "plain": [
            "plain.com",
        ],

    }

    return aliases.get(
        app_name.lower().strip(),
        []
    )


# ============================================================
# DOMAIN MATCHING
# ============================================================

def app_domain_matches(url, app):

    try:

        domain = urlparse(
            url
        ).netloc.lower()

        if domain.startswith("www."):
            domain = domain[4:]

    except Exception:
        return False

    # 1. Dataset hint domain.
    hint_domains = extract_domains_from_hint(
        app.get(
            "website_or_docs_hint",
            ""
        )
    )

    for hint_domain in hint_domains:

        if (
            domain == hint_domain
            or domain.endswith(
                "." + hint_domain
            )
            or base_domain(domain)
            == base_domain(hint_domain)
        ):
            return True

    # 2. App name in domain.
    name = clean_text(
        app.get(
            "app_name",
            ""
        )
    ).lower()

    normalized_name = re.sub(
        r"[^a-z0-9]",
        "",
        name
    )

    normalized_domain = re.sub(
        r"[^a-z0-9]",
        "",
        domain
    )

    if (
        normalized_name
        and normalized_name in normalized_domain
    ):
        return True

    # 3. Explicit alias.
    for alias in get_alias_domains(name):

        alias = alias.lower()

        if (
            domain == alias
            or domain.endswith(
                "." + alias
            )
        ):
            return True

    return False


# ============================================================
# DOCUMENTATION URL SELECTION
# ============================================================

def select_documentation_urls(
    app,
    search_response
):

    urls = extract_search_urls(
        search_response
    )

    if not urls:
        return []

    official = [
        url
        for url in urls
        if app_domain_matches(
            url,
            app
        )
    ]

    # Prefer vendor documentation.
    candidates = (
        official
        if official
        else urls
    )

    final = []

    for url in candidates:

        if is_bad_search_url(url):
            continue

        if url not in final:
            final.append(url)

    return final[:MAX_DOCUMENTS]


# ============================================================
# FETCH DOCUMENTS
# ============================================================

def extract_fetched_content(response):

    """
    Extract actual page content from Fetch.
    """

    if isinstance(response, dict):

        data = response.get(
            "data",
            response
        )

        if isinstance(data, dict):

            results = data.get(
                "results",
                []
            )

            if isinstance(results, list):

                pieces = []

                for item in results:

                    if not isinstance(item, dict):
                        continue

                    content = (
                        item.get("content")
                        or item.get("text")
                        or item.get("markdown")
                        or item.get("body")
                    )

                    if content:
                        pieces.append(
                            str(content).strip()
                        )

                if pieces:
                    return "\n\n".join(
                        pieces
                    ).strip()

    return response_to_text(
        response
    )


def fetch_documents(
    urls,
    max_characters=MAX_DOCUMENT_CHARS
):

    if not urls:
        return ""

    clean_urls = []

    for url in urls:

        cleaned = clean_url(url)

        if not cleaned:
            continue

        if is_bad_search_url(cleaned):
            continue

        if cleaned not in clean_urls:
            clean_urls.append(cleaned)

    if not clean_urls:
        return ""

    response = composio.tools.execute(
        FETCH_TOOL,
        {
            "urls": clean_urls,
            "max_characters": max_characters,
        },
        version=TOOL_VERSION,
    )

    return extract_fetched_content(
        response
    )


# ============================================================
# RESEARCH EXTRACTION
# ============================================================

def extract_research(
    app,
    documentation,
    evidence_urls
):

    prompt = f"""
You are a research analyst conducting product/API
research for an AI Product Operations team.

APP:
{app["app_name"]}

CATEGORY:
{app["category"]}

Use ONLY the supplied documentation.

DOCUMENTATION:
{documentation[:8000]}

Return ONLY valid JSON.

Required format:

{{
  "app_name": "{app["app_name"]}",
  "description": "",
  "auth_methods": [],
  "self_serve_or_gated": "",
  "credential_access_details": "",
  "api_surface": [],
  "api_breadth": "",
  "mcp_available": false,
  "mcp_details": "",
  "buildability_verdict": "",
  "main_blocker": "",
  "evidence": []
}}

Rules:

1. Return JSON only.
2. No Markdown.
3. No code fences.
4. Do not invent facts.
5. Do not invent numbers.
6. auth_methods means API/programmatic authentication.
7. Do not classify normal website login as API authentication.
8. self_serve_or_gated concerns obtaining API/MCP
   credentials or developer access.
9. api_breadth must be exactly:
   Narrow, Moderate, Broad, or Extensive.
10. buildability_verdict must be exactly:
    Buildable,
    Buildable with constraints,
    Not currently buildable.
11. mcp_available must be true only when documented
    evidence supports MCP availability specifically
    for this app.
12. Distinguish official MCP from community MCP.
13. Generic MCP documentation does not prove
    app-specific MCP support.
14. evidence must contain URLs from supplied documentation.
15. Do not make unsupported quantitative claims.
16. Be conservative when evidence is insufficient.
"""

    messages = [
        {
            "role": "system",
            "content": (
                "Return compact valid JSON only. "
                "Never add commentary."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            response = groq_chat(
                messages,
                max_tokens=2200,
                temperature=0
            )

            parsed = parse_json_response(
                response
            )

            if isinstance(parsed, dict):
                return parsed

            print(
                f"Research JSON invalid "
                f"on attempt "
                f"{attempt}/{MAX_RETRIES}."
            )

            messages = [
                {
                    "role": "system",
                    "content": (
                        "Return valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": f"""
Repair the following malformed output.

Required JSON keys:

app_name
description
auth_methods
self_serve_or_gated
credential_access_details
api_surface
api_breadth
mcp_available
mcp_details
buildability_verdict
main_blocker
evidence

Return ONLY valid JSON.

Malformed output:
{response}
""",
                },
            ]

        except Exception as e:

            print(
                f"Research extraction "
                f"attempt {attempt} failed: "
                f"{e}"
            )

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)

    # --------------------------------------------------------
    # Field-by-field fallback
    # --------------------------------------------------------

    print(
        "Using field-by-field extraction fallback..."
    )

    result = {
        "app_name": app["app_name"],
        "description": "",
        "auth_methods": [],
        "self_serve_or_gated": "",
        "credential_access_details": "",
        "api_surface": [],
        "api_breadth": "",
        "mcp_available": False,
        "mcp_details": "",
        "buildability_verdict": "",
        "main_blocker": "",
        "evidence": evidence_urls,
    }

    fields = [

        (
            "description",
            "Give a one-sentence description."
        ),

        (
            "auth_methods",
            "List documented API/programmatic "
            "authentication methods only."
        ),

        (
            "self_serve_or_gated",
            "State whether API/MCP credentials "
            "are self-serve, gated, or mixed."
        ),

        (
            "credential_access_details",
            "Explain how a developer obtains "
            "API credentials or developer access."
        ),

        (
            "api_surface",
            "List documented API surfaces such as "
            "REST, GraphQL, SOAP, SDKs, webhooks, MCP."
        ),

        (
            "api_breadth",
            "Classify as Narrow, Moderate, Broad, "
            "or Extensive."
        ),

        (
            "mcp_available",
            "Answer only true or false based on "
            "app-specific documented MCP evidence."
        ),

        (
            "mcp_details",
            "Describe documented MCP support and "
            "distinguish official/community."
        ),

        (
            "buildability_verdict",
            "Choose Buildable, Buildable with constraints, "
            "or Not currently buildable."
        ),

        (
            "main_blocker",
            "State the main documented blocker, "
            "or empty if none."
        ),
    ]

    for field, instruction in fields:

        small_prompt = f"""
APP:
{app["app_name"]}

DOCUMENTATION:
{documentation[:5000]}

TASK:
{instruction}

Rules:
Use only documented information.
Do not guess.
Return ONLY the answer.
"""

        try:

            answer = groq_chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "Answer only from "
                            "supplied evidence."
                        ),
                    },
                    {
                        "role": "user",
                        "content": small_prompt,
                    },
                ],
                max_tokens=300,
                temperature=0
            )

            answer = clean_text(
                answer
            ).strip("\"'")

            if field in {
                "auth_methods",
                "api_surface",
            }:

                result[field] = [
                    x.strip()
                    for x in re.split(
                        r",|;",
                        answer
                    )
                    if x.strip()
                ]

            elif field == "mcp_available":

                result[field] = (
                    answer.lower() == "true"
                )

            else:

                result[field] = answer

        except Exception as e:

            print(
                f"Field fallback failed "
                f"for {field}: {e}"
            )

    return result


# ============================================================
# RESEARCH ONE APP
# ============================================================

def research_one_app(app):

    print(
        f"\nSearching: "
        f"{app['app_name']}"
    )

    query = (
        f"{app['app_name']} official developer "
        f"API documentation authentication "
        f"OAuth API key MCP credentials"
    )

    search_response = web_search(
        query
    )

    print(
        "Selecting documentation sources..."
    )

    selected_urls = select_documentation_urls(
        app,
        search_response
    )

    if not selected_urls:
        raise RuntimeError(
            "No usable documentation URLs found."
        )

    print(
        f"Documentation URLs: "
        f"{len(selected_urls)}"
    )

    for url in selected_urls:
        print(f"- {url}")

    print(
        "Fetching documentation..."
    )

    documentation = fetch_documents(
        selected_urls
    )

    if not documentation:
        raise RuntimeError(
            "Documentation fetch returned "
            "empty content."
        )

    print(
        "Extracting structured research..."
    )

    parsed = extract_research(
        app,
        documentation,
        selected_urls
    )

    if not isinstance(parsed, dict):
        raise RuntimeError(
            "Research extraction did not "
            "return a JSON object."
        )

    result = {
        "id": app["id"],
        "app_name": app["app_name"],
        "category": app["category"],

        "description": clean_text(
            parsed.get(
                "description",
                ""
            )
        ),

        "auth_methods": parsed.get(
            "auth_methods",
            []
        ),

        "self_serve_or_gated": clean_text(
            parsed.get(
                "self_serve_or_gated",
                ""
            )
        ),

        "credential_access_details": clean_text(
            parsed.get(
                "credential_access_details",
                ""
            )
        ),

        "api_surface": parsed.get(
            "api_surface",
            []
        ),

        "api_breadth": clean_text(
            parsed.get(
                "api_breadth",
                ""
            )
        ),

        "mcp_available": bool(
            parsed.get(
                "mcp_available",
                False
            )
        ),

        "mcp_details": clean_text(
            parsed.get(
                "mcp_details",
                ""
            )
        ),

        "buildability_verdict": clean_text(
            parsed.get(
                "buildability_verdict",
                ""
            )
        ),

        "main_blocker": clean_text(
            parsed.get(
                "main_blocker",
                ""
            )
        ),

        "evidence": normalize_urls(
            parsed.get(
                "evidence",
                []
            )
        ),
    }

    if not result["evidence"]:
        result["evidence"] = selected_urls

    return result


# ============================================================
# VERIFICATION QUERY
# ============================================================

def get_verification_query(
    app,
    research,
    field
):

    app_name = research["app_name"]

    if field == "auth_methods":

        return (
            f"{app_name} official developer "
            f"API authentication documentation "
            f"OAuth API key bearer token credentials"
        )

    if field == "self_serve_or_gated":

        return (
            f"{app_name} official developer "
            f"API credentials signup pricing "
            f"free trial developer access "
            f"approval admin contact sales"
        )

    if field == "mcp_available":

        return (
            f'"{app_name}" MCP server '
            f'"{app_name}" Model Context Protocol '
            f'"{app_name}" MCP integration '
            f'"{app_name}" official MCP'
        )

    return (
        f"{app_name} official developer documentation"
    )


# ============================================================
# VERIFICATION URL SELECTION
# ============================================================

def select_verification_urls(
    app,
    search_response
):

    urls = extract_search_urls(
        search_response
    )

    if not urls:
        return []

    official = [
        url
        for url in urls
        if app_domain_matches(
            url,
            app
        )
    ]

    candidates = (
        official
        if official
        else urls
    )

    final = []

    for url in candidates:

        if is_bad_search_url(url):
            continue

        if url not in final:
            final.append(url)

    return final[
        :MAX_VERIFICATION_DOCUMENTS
    ]


# ============================================================
# MCP VALIDATION
# ============================================================

def is_generic_mcp_source(url):

    if not url:
        return True

    lower = url.lower()

    generic_patterns = [
        "github.com/modelcontextprotocol/",
        "github.com/model-contextprotocol/",
        "github.com/model-context-protocol/",
        "modelcontextprotocol.io",
        "modelcontextprotocol.info",
        "mcp-docs",
        "model-context-protocol",
    ]

    return any(
        pattern in lower
        for pattern in generic_patterns
    )


def mcp_source_is_app_specific(
    app,
    url,
    documentation
):

    if not url:
        return False

    if is_generic_mcp_source(url):
        return False

    # Official vendor source.
    if app_domain_matches(
        url,
        app
    ):
        return True

    # Community source:
    # require the app name in the fetched content.
    app_name = clean_text(
        app.get(
            "app_name",
            ""
        )
    ).lower()

    normalized_app = re.sub(
        r"[^a-z0-9]",
        "",
        app_name
    )

    normalized_doc = re.sub(
        r"[^a-z0-9]",
        "",
        documentation.lower()
    )

    if (
        normalized_app
        and normalized_app in normalized_doc
    ):
        return True

    tokens = [
        token
        for token in re.findall(
            r"[a-z0-9]+",
            app_name
        )
        if len(token) >= 4
    ]

    if not tokens:
        return False

    matched = sum(
        1
        for token in tokens
        if token in documentation.lower()
    )

    required = max(
        1,
        min(2, len(tokens))
    )

    return matched >= required


# ============================================================
# VERIFICATION CLAIM
# ============================================================

def build_verification_claim(
    research,
    field
):

    if field == "auth_methods":

        return f"""
The first-pass research claims these
API/programmatic authentication methods:

{research.get("auth_methods", "")}

Verify these API authentication methods.

Do NOT treat normal website login,
email/password login, or general SSO as API
authentication unless evidence explicitly shows
that it is used for API access.
"""

    if field == "self_serve_or_gated":

        return f"""
The first-pass research says:

Self-serve/gated:
{research.get("self_serve_or_gated", "")}

Credential access details:
{research.get("credential_access_details", "")}

Verify whether a developer can actually obtain
the relevant API/MCP credentials or developer
access as described.

Do NOT confuse ordinary product signup with
API access.
"""

    if field == "mcp_available":

        return f"""
The first-pass research says:

MCP available:
{research.get("mcp_available", "")}

MCP details:
{research.get("mcp_details", "")}

Verify whether an MCP integration/server exists
SPECIFICALLY FOR {research.get("app_name", "")}.

Distinguish:

- official vendor MCP
- official partner MCP
- community MCP
- no MCP found

A generic MCP specification, generic MCP documentation,
or generic MCP server repository is NOT evidence that
{research.get("app_name", "")} has an MCP integration.

Do not call a community implementation official.
"""

    return ""


# ============================================================
# VERIFY FIELD
# ============================================================

def verify_field(
    app,
    research,
    field
):

    app_name = research["app_name"]

    print(
        f"Verifying "
        f"{app_name} → {field}"
    )

    query = get_verification_query(
        app,
        research,
        field
    )

    search_response = web_search(
        query
    )

    search_text = response_to_text(
        search_response
    )

    verification_urls = select_verification_urls(
        app,
        search_response
    )

    # --------------------------------------------------------
    # MCP filtering
    # --------------------------------------------------------

    if field == "mcp_available":

        verification_urls = [
            url
            for url in verification_urls
            if not is_generic_mcp_source(url)
        ]

    documentation = ""

    if verification_urls:

        try:

            documentation = fetch_documents(
                verification_urls,
                max_characters=MAX_VERIFICATION_CHARS
            )

        except Exception as e:

            print(
                f"Verification fetch warning "
                f"for {app_name} → {field}: "
                f"{e}"
            )

    # --------------------------------------------------------
    # Search fallback
    # --------------------------------------------------------

    if not documentation:

        documentation = (
            "No page content was successfully fetched.\n\n"
            "Search-result text:\n"
            +
            search_text[:8000]
        )

    # --------------------------------------------------------
    # MCP app-specific evidence
    # --------------------------------------------------------

    if field == "mcp_available":

        app_specific_urls = []

        for url in verification_urls:

            if mcp_source_is_app_specific(
                app,
                url,
                documentation
            ):
                app_specific_urls.append(url)

        verification_urls = app_specific_urls

        if not verification_urls:

            documentation = (
                "No app-specific MCP evidence was found.\n\n"
                "Generic MCP sources were intentionally excluded.\n\n"
                +
                documentation[:6000]
            )

    claim = build_verification_claim(
        research,
        field
    )

    source_list = "\n".join(
        verification_urls
    )

    prompt = f"""
You are an independent verification analyst.

Your task is to verify an EXISTING research claim
using fresh evidence.

APP:
{app_name}

FIELD:
{field}

FIRST-PASS CLAIM:
{claim}

FRESH DOCUMENTATION:
{documentation[:12000]}

VERIFICATION URLS:
{source_list}

Return ONLY valid JSON:

{{
  "verified_value": "",
  "verification_status": "",
  "verification_source": "",
  "notes": ""
}}

Allowed verification_status values:

verified
partially_verified
contradicted
not_verified

Definitions:

verified:
Fresh evidence supports the first-pass claim.

partially_verified:
Fresh evidence supports some of the claim
but not all details, or an important qualification
exists.

contradicted:
Fresh evidence directly conflicts with the claim.

not_verified:
Available evidence is insufficient to determine
whether the claim is correct.

IMPORTANT:

1. Verify the actual research claim, not the CSV
   field name.

2. auth_methods means API/programmatic authentication.

3. self_serve_or_gated means API/MCP credential
   or developer access.

4. mcp_available means MCP specifically for the
   named app.

5. Generic MCP repositories/specifications do NOT
   prove app-specific MCP support.

6. Prefer official vendor documentation.

7. Never use favicon/image URLs as evidence.

8. Do not invent facts.

9. Do not infer contradiction merely because evidence
   is missing.

10. Missing evidence should normally produce
    not_verified.

11. verification_source must be a real documentation URL.

12. Use one of the verification URLs whenever possible.

13. Keep notes concise and evidence-based.

14. Return JSON only.
"""

    try:

        response = groq_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Return valid JSON only. "
                        "No commentary."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=1000,
            temperature=0
        )

        parsed = parse_json_response(
            response
        )

    except Exception as e:

        return {
            "id": research["id"],
            "app_name": app_name,
            "field": field,
            "verified_value": "",
            "verification_status": "not_verified",
            "verification_source": "",
            "notes": f"Verification failed: {e}",
        }

    # --------------------------------------------------------
    # Parsing failure
    # --------------------------------------------------------

    if not isinstance(parsed, dict):

        return {
            "id": research["id"],
            "app_name": app_name,
            "field": field,
            "verified_value": "",
            "verification_status": "not_verified",
            "verification_source": (
                verification_urls[0]
                if verification_urls
                else ""
            ),
            "notes":
                "Verification response could not be parsed.",
        }

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    status = clean_text(
        parsed.get(
            "verification_status",
            "not_verified"
        )
    ).lower()

    allowed_statuses = {
        "verified",
        "partially_verified",
        "contradicted",
        "not_verified",
    }

    if status not in allowed_statuses:
        status = "not_verified"

    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    source = clean_url(
        parsed.get(
            "verification_source",
            ""
        )
    )

    if source and is_bad_search_url(source):
        source = None

    # --------------------------------------------------------
    # Source must be one of verified URLs whenever possible.
    # --------------------------------------------------------

    if source and verification_urls:

        if source not in verification_urls:

            # Do not blindly trust an arbitrary LLM-generated URL.
            source = verification_urls[0]

    if not source and verification_urls:
        source = verification_urls[0]

    # --------------------------------------------------------
    # MCP safety
    # --------------------------------------------------------

    if field == "mcp_available":

        if not verification_urls:

            status = "not_verified"

            verified_value = ""

            notes = (
                "No app-specific MCP evidence was found. "
                "Generic MCP sources were excluded."
            )

            source = ""

        else:

            verified_value = clean_text(
                parsed.get(
                    "verified_value",
                    ""
                )
            )

            notes = clean_text(
                parsed.get(
                    "notes",
                    ""
                )
            )

    else:

        verified_value = clean_text(
            parsed.get(
                "verified_value",
                ""
            )
        )

        notes = clean_text(
            parsed.get(
                "notes",
                ""
            )
        )

    return {
        "id": research["id"],
        "app_name": app_name,
        "field": field,
        "verified_value": verified_value,
        "verification_status": status,
        "verification_source": source or "",
        "notes": notes,
    }


# ============================================================
# VERIFY ONE APP
# ============================================================

def verify_one_app(
    app,
    research
):

    fields = [
        "auth_methods",
        "self_serve_or_gated",
        "mcp_available",
    ]

    results = []

    for field in fields:

        try:

            result = verify_field(
                app,
                research,
                field
            )

            results.append(result)

        except Exception as e:

            print(
                f"Verification error "
                f"{research['app_name']} "
                f"→ {field}: {e}"
            )

            results.append({
                "id":
                    research["id"],

                "app_name":
                    research["app_name"],

                "field":
                    field,

                "verified_value":
                    "",

                "verification_status":
                    "not_verified",

                "verification_source":
                    "",

                "notes":
                    str(e),
            })

        time.sleep(
            REQUEST_DELAY
        )

    return results


# ============================================================
# VERIFICATION SUMMARY
# ============================================================

def print_verification_summary(
    verification_rows
):

    status_counts = {
        "verified": 0,
        "partially_verified": 0,
        "contradicted": 0,
        "not_verified": 0,
    }

    field_counts = {}

    for row in verification_rows:

        status = row.get(
            "verification_status",
            "not_verified"
        )

        if status not in status_counts:
            status = "not_verified"

        status_counts[status] += 1

        field = row.get(
            "field",
            ""
        )

        if field not in field_counts:

            field_counts[field] = {
                "verified": 0,
                "partially_verified": 0,
                "contradicted": 0,
                "not_verified": 0,
            }

        field_counts[field][status] += 1

    print(
        "\nVerification status:"
    )

    for status in [
        "verified",
        "partially_verified",
        "contradicted",
        "not_verified",
    ]:

        print(
            f"{status}: "
            f"{status_counts[status]}"
        )

    print(
        "\nVerification by field:"
    )

    for field, counts in field_counts.items():

        print(
            f"\n{field}:"
        )

        for status in [
            "verified",
            "partially_verified",
            "contradicted",
            "not_verified",
        ]:

            print(
                f"  {status}: "
                f"{counts[status]}"
            )


# ============================================================
# VERIFICATION-ONLY MODE
# ============================================================

def run_verification_only():

    apps = load_apps()

    research_rows = load_existing_research()

    total_apps = len(apps)

    research_count = len(
        research_rows
    )

    print("=" * 70)

    print(
        "VERIFICATION-ONLY MODE"
    )

    print("=" * 70)

    print(
        f"Applications in apps.csv: "
        f"{total_apps}"
    )

    print(
        f"Research records found: "
        f"{research_count}"
    )

    # --------------------------------------------------------
    # Validate 100/100 research coverage.
    # --------------------------------------------------------

    if research_count != total_apps:

        research_ids = {
            str(row.get("id"))
            for row in research_rows
        }

        expected_ids = {
            str(app.get("id"))
            for app in apps
        }

        missing_ids = sorted(
            expected_ids - research_ids,
            key=lambda x: int(x)
        )

        print("\nSTOP:")
        print(
            "Research dataset is incomplete."
        )

        print(
            "Missing IDs:"
        )

        print(
            ", ".join(missing_ids)
        )

        print(
            "\nNo verification was performed."
        )

        return

    app_by_id = {
        str(app["id"]): app
        for app in apps
    }

    research_by_id = {
        str(row["id"]): row
        for row in research_rows
    }

    ordered_ids = sorted(
        research_by_id.keys(),
        key=lambda x: int(x)
    )

    # Always create a fresh verification pass.
    verification_rows = []

    print(
        "\nFresh independent verification pass."
    )

    print(
        "Existing verification.csv will NOT be reused."
    )

    for position, app_id in enumerate(
        ordered_ids,
        start=1
    ):

        research = research_by_id[
            app_id
        ]

        app = app_by_id[
            app_id
        ]

        print("\n")
        print("=" * 70)

        print(
            f"VERIFY {position}/{total_apps}: "
            f"{app['app_name']}"
        )

        print("=" * 70)

        results = verify_one_app(
            app,
            research
        )

        verification_rows.extend(
            results
        )

        # Incremental save.
        save_verification_results(
            verification_rows
        )

    verification_count = len(
        verification_rows
    )

    print("\n")
    print("=" * 70)

    print(
        "VERIFICATION COMPLETED"
    )

    print("=" * 70)

    print(
        f"Research records: "
        f"{research_count}/{total_apps}"
    )

    print(
        f"Verification records: "
        f"{verification_count}"
    )

    print_verification_summary(
        verification_rows
    )

    print(
        "\nOutput:"
    )

    print(
        f"Verification: "
        f"{VERIFICATION_FILE}"
    )

    print(
        "\nVerification-only run finished."
    )


# ============================================================
# RESEARCH-ONLY MODE
# ============================================================

def run_research_only():

    apps = load_apps()

    research_rows = (
        load_existing_research()
    )

    failures = (
        load_existing_failures()
    )

    total = len(apps)

    print("=" * 70)

    print(
        "RESEARCH-ONLY MODE"
    )

    print("=" * 70)

    print(
        f"Loaded {total} applications"
    )

    print(
        f"Existing research rows: "
        f"{len(research_rows)}"
    )

    researched_ids = {
        str(row.get("id"))
        for row in research_rows
    }

    for index, app in enumerate(
        apps,
        start=1
    ):

        app_id = str(
            app["id"]
        )

        print("\n")
        print("=" * 70)

        print(
            f"APP {index}/{total}: "
            f"{app['app_name']}"
        )

        print("=" * 70)

        if app_id in researched_ids:

            print(
                f"Skipping "
                f"{app['app_name']} "
                f"(already researched)."
            )

            continue

        try:

            research = research_one_app(
                app
            )

            research_rows = [
                row
                for row in research_rows
                if str(
                    row.get("id")
                ) != app_id
            ]

            research_rows.append(
                research
            )

            researched_ids.add(
                app_id
            )

            save_research_results(
                research_rows
            )

            failures = [
                row
                for row in failures
                if str(
                    row.get("id")
                ) != app_id
            ]

            save_failures(
                failures
            )

            print(
                f"Research completed: "
                f"{app['app_name']}"
            )

        except Exception as e:

            print(
                f"Research FAILED for "
                f"{app['app_name']}: "
                f"{e}"
            )

            failures = [
                row
                for row in failures
                if str(
                    row.get("id")
                ) != app_id
            ]

            failures.append({
                "id":
                    app["id"],

                "app_name":
                    app["app_name"],

                "category":
                    app["category"],

                "stage":
                    "research",

                "error":
                    str(e),
            })

            save_failures(
                failures
            )

        time.sleep(
            REQUEST_DELAY
        )

    print("\n")
    print("=" * 70)

    print(
        "RESEARCH-ONLY COMPLETED"
    )

    print("=" * 70)

    print(
        f"Research records: "
        f"{len(research_rows)}/{total}"
    )

    print(
        f"Failures: "
        f"{len(failures)}"
    )


# ============================================================
# FULL PIPELINE
# ============================================================

def run_full_pipeline():

    print("=" * 70)

    print(
        "FULL PIPELINE MODE"
    )

    print("=" * 70)

    # Research resumes from existing results.
    run_research_only()

    apps = load_apps()

    research_rows = (
        load_existing_research()
    )

    if len(research_rows) != len(apps):

        print(
            "\nResearch incomplete."
        )

        print(
            "Verification will NOT start."
        )

        return

    run_verification_only()


# ============================================================
# ENTRY POINT
# ============================================================

def main():

    mode = (
        sys.argv[1].lower()
        if len(sys.argv) > 1
        else "full"
    )

    if mode == "verify":

        run_verification_only()

    elif mode == "research":

        run_research_only()

    elif mode == "full":

        run_full_pipeline()

    else:

        print(
            "Invalid mode."
        )

        print("\nUse:")

        print(
            "  python research_agent.py verify"
        )

        print(
            "  python research_agent.py research"
        )

        print(
            "  python research_agent.py full"
        )


if __name__ == "__main__":
    main()