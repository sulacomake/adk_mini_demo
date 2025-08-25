from google.adk.agents import Agent
from google.genai import types
from google.adk.tools import google_search

MODEL = "gemini-2.5-flash"

agent_news = Agent(
    model=MODEL,
    name="agent_news",
    description="Get news from Peruvian news agency",
    instruction="""
      Focus on the requested topic and look for information in the news, focus on national sources. Always respond in spanish.
    """,
    generate_content_config=types.GenerateContentConfig(temperature=0.2),
    tools=[google_search],
)