from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class SummaryOutput(BaseModel):
    summary: str = Field(..., description="A concise summary of the input text.")

output_parser = PydanticOutputParser(pydantic_object=SummaryOutput)

template = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in bullet points:\n\n{text} \n {format_instructions}",
    partial_variables={"format_instructions": output_parser.get_format_instructions()}
)

chain = template | model | output_parser
result = chain.invoke({"text": "The impact of AI on Medical Research."})

print(result.summary)