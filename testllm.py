# !pip install langchain-openai

from langchain_openai import ChatOpenAI
import os
import httpx

client = httpx.Client(verify=False)

llm = ChatOpenAI (
    base_url="https://genailab.url.in", 
    model = "azure_ai/genailab-maas-DeepSeek-V3-0324", 
    api_key="sk-798y0oiohj",
    http_client = client 
) 
resp = llm.invoke("Hi")
print(resp)
print(type(resp))

output = resp.dict()
print(output.get('content', ''))


for key, value in output.items():
    print(f"{key}: {value}")
