from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

#first template
template_1 = PromptTemplate(
    input_variables=["topic"],
    template="Write a detailed report on {topic}",
    output_parser=StrOutputParser()
)

#second template
template_2 = PromptTemplate(
    input_variables=["report"],
    template="Summarize the following report in bullet points:\n\n{report}",
    output_parser=StrOutputParser()
)

#chain the templates
chain = template_1 | model | template_2 | model
result = chain.invoke({"topic": "The impact of AI on modern healthcare"})
print("Final Result:\n", result.content)