from pdfparser import PDFParser
from vectorstore import VectorStore
from flask import Flask, request, render_template, jsonify
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain_core.runnables import chain
from langchain_core.output_parsers import StrOutputParser


app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize embeddings + vectorstore wrapper
embeddings =  OllamaEmbeddings(    model="nomic-embed-text:latest")
vs = VectorStore(embeddings)
qa_chain = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():
    global qa_chain,vs
    file = request.files["pdf"]
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    parser = PDFParser(filepath)
    docs,filename = parser.process_file()
    # Step 3: Store in ChromaDB
    vs.add_documents(docs)




    return jsonify({"message": "PDF processed and stored in ChromaDB"})

@app.route("/chat", methods=["POST"])
def chat():
        # Step 4: Build RAG chain
    global vs
    retriever = vs.load_retriever(k=3)
    # docs = retriever.get_relevant_documents("who hosted Bajaj Finserv Limited Q1 FY '26 Earnings Conference Call?")
    # print(docs)
    llm = OllamaLLM(model='llama3.2:latest',temperature=0.0)
    
   
    prompt_hyde = ChatPromptTemplate.from_template("""Please write a passage to 
    answer the question.\n Question: {question} \n Passage:""")

    generate_doc = (
    prompt_hyde | llm | StrOutputParser() 
)
    

    retrieval_chain = generate_doc  | retriever
    
    

    prompt = ChatPromptTemplate.from_template("""Answer the following question based 
    on this context:

    {context}

    Question: {question}
    """)
    @chain
    def multi_query_qa(input):
        # fetch relevant documents 
        print(input)
        docs = retrieval_chain.invoke(input)
        # format prompt
        print(docs)
        formatted = prompt.invoke({"context": docs, "question": input})
        # generate answer
        print(formatted)
        answer = llm.invoke(formatted)
        return answer

    # run
    user_input = request.json.get("message")
    

    
    if not user_input:
        return jsonify({"error": "Empty query"}), 400

    result = multi_query_qa.invoke(user_input)
    # result = qa_chain.invoke({"input": user_input})
    print(result)
    return jsonify({
        # "answer": result["answer"]
        "answer":result
        # "sources": [doc.page_content[:200] for doc in result["source_documents"]]
    })

if __name__ == "__main__":
    app.run(debug=True)


