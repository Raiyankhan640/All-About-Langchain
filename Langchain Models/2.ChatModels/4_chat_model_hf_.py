from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Can I use Deepseek api at free of cost on my project?")

print(result.content)