# Composio Product Ops Research Agent

An automated research and verification pipeline for evaluating 100 apps for agent-toolkit buildability.

## Overview

This project automates the type of product research required before building an agent toolkit.

For each app, the pipeline researches:

- Category
- One-line description
- Authentication methods
- Self-serve vs gated credential access
- API surface
- MCP availability
- Buildability verdict
- Evidence from public documentation

A separate verification pass then re-checks selected fields against publicly available documentation and records whether the original research was verified, partially verified, contradicted, or not verified.

---

## Research Dataset

The project evaluates **100 apps across 10 categories**, with 10 apps per category.

The categories are:

1. CRM and Sales
2. Support and Helpdesk
3. Communications and Messaging
4. Marketing, Ads, Email and Social
5. Ecommerce
6. Data, SEO and Scraping
7. Developer, Infra and Data platforms
8. Productivity and Project Management
9. Finance and Fintech
10. AI, Research and Media-native

Project Structure
composio-product-ops/
│
├── .venv/
│
├── data/
│   ├── apps.csv
│   ├── research/
│   │   └── _results.csv
│   └── verification.csv
│
├── research_agent.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env

Technology Stack
- Python
- Composio SDK
- Composio Search
- Groq
- Pandas
- python-dotenv

The input dataset is stored in:

```text
data/apps.csv

The research pipeline follows this workflow:

100-app dataset
      ↓
Composio Search
      ↓
Find relevant documentation
      ↓
Fetch public documentation
      ↓
Groq via Composio
      ↓
Structured research result
      ↓
Research CSV
      ↓
Independent verification search
      ↓
Verification CSV

Running the Project
1. Create and activate the virtual environment
Windows PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1

2. Install dependencies
pip install -r requirements.txt

3. Configure environment variables
Create a .env file in the project root:
COMPOSIO_API_KEY=YOUR_COMPOSIO_API_KEY
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
Do not commit .env or expose API keys publicly.

4. Run research
python research_agent.py research
This researches the 100 apps and writes:
data/research/_results.csv

5. Run verification
python research_agent.py verify
This performs field-level verification and writes:
data/verification.csv

6. Run the complete pipeline
python research_agent.py full

Results
The completed research run contains:
- 100 apps researched
- 100 unique apps
- 300 field-level verification checks
- 87 apps classified as buildable
- 12 apps classified as buildable with constraints
- 1 app classified as not currently buildable