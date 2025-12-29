from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt_1 = PromptTemplate(
    input_variables=["Topic"],
    template="Generate a detailed report on: {Topic}",
    output_parser=StrOutputParser()
)

prompt_2 = PromptTemplate(
    input_variables=["Report"],
    template="Summarize the following report in bullet points: {Report}",
    output_parser=StrOutputParser()
)
parser = StrOutputParser()
chain  = prompt_1 | model | prompt_2 | model
result = chain.invoke({"Topic": "The impact of AI on modern education"})

print(result.content)