from google.adk.agents import SequentialAgent
from agent_news.agent import agent_news

root_agent = SequentialAgent(
        name="agent_corp_brand",
        description="This agent acts as a friendly assistant always talks in Spanish in a friendly tone.",
        sub_agents=[agent_news],
    )