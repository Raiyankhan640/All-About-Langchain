from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V3.2", task="text-generation")
model = ChatHuggingFace(llm=llm)

class FeedBack(BaseModel):
    type: Literal["positive", "negative"] = Field(..., description="The type of feedback")


str_parser = StrOutputParser()
pydantic_parser = PydanticOutputParser(pydantic_object=FeedBack)

prompt_1 = PromptTemplate(
    input_variables=["response"],
    template="Given the following response, determine if the feedback is positive or negative:\n\nResponse: {response}\n\nIs the feedback positive or negative? {format_instruction}",
    partial_variables={"format_instruction": pydantic_parser.get_format_instructions()}
)

classifier_chain = prompt_1 | model | pydantic_parser

prompt_positive = PromptTemplate(
    input_variables=["feedback"],
    template="Write an appropriate response based on this positive feedback: {feedback}",
    output_parser=StrOutputParser()
)

prompt_negative = PromptTemplate(
    input_variables=["feedback"],
    template="Write an appropriate response based on this negative feedback: {feedback}",
    output_parser=StrOutputParser()
)
branch_chain = RunnableBranch(
    (lambda x: x.type == "positive", prompt_positive | model),
    (lambda x: x.type == "negative", prompt_negative | model),
    RunnableLambda(lambda x: "Unable to determine feedback type."),
)
chain = classifier_chain | branch_chain
result = chain.invoke({"response": "I absolutely love the new features in your product! It has made my life so much easier."})

print(result.content)
