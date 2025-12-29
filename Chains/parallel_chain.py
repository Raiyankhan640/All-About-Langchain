from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm_1 = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V3.2", task="text-generation")
llm_2 = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-R1", task="text-generation")

model_1 = ChatHuggingFace(llm=llm_1)
model_2 = ChatHuggingFace(llm=llm_2)

prompt_1 = PromptTemplate(
    input_variables=["text"],
    template="Give me lecture notes covering all the important concepts on: {text}",
    output_parser=StrOutputParser(),
)
prompt_2 = PromptTemplate(
    input_variables=["text"],
    template="Generate a Quiz on the following lecture: {text}",
    output_parser=StrOutputParser(),
)
prompt_3 = PromptTemplate(
    input_variables=["notes", "quiz"],
    template="Merge the following lecture notes and quiz into a single document:\n\n notes--> {notes}\n \n quiz--> {quiz}",
    output_parser=StrOutputParser(),
)
parallel_chain = RunnableParallel(
    {"notes": prompt_1 | model_1, "quiz": prompt_2 | model_2}
)
merged_chain = parallel_chain | prompt_3 | model_1

text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""
result = merged_chain.invoke({"text": text})

print(result.content)
