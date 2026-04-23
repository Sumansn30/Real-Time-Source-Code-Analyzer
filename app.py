from langchain_chroma import Chroma
from src.helper import load_embedding, repo_ingestion, load_repo, text_splitter, force_remove_readonly
from dotenv import load_dotenv
import os
import stat
import shutil
from flask import Flask, request, jsonify, render_template
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory

app = Flask(__name__, template_folder='templates', static_folder='static')

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = load_embedding()
persist_directory = "db"

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
llm = ChatOpenAI()
memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 8}),
    memory=memory
)


@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')


@app.route('/chatbot', methods=["GET", "POST"])
def gitRepo():
    global vectordb, qa, memory
    if request.method == 'POST':
        try:
            user_input = request.form.get('question')
            if not user_input:
                return jsonify({"response": "No URL provided"})

            repo_ingestion(user_input)           # ✅ Windows-safe clone

            # ✅ Rebuild vectordb inline (no os.system)
            documents = load_repo("repo/")
            text_chunks = text_splitter(documents)
            vectordb = Chroma.from_documents(
                text_chunks,
                embedding=embeddings,
                persist_directory=persist_directory
            )
            memory = ConversationSummaryMemory(
                llm=llm, memory_key="chat_history", return_messages=True
            )
            qa = ConversationalRetrievalChain.from_llm(
                llm,
                retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 8}),
                memory=memory
            )
            return jsonify({"response": "✅ Repository loaded! Ask your questions below."})

        except Exception as e:
            print(f"ERROR: {e}")
            return jsonify({"response": f"Error: {str(e)}"})

    return jsonify({"response": "Send a POST request with a GitHub URL"})


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg", "")
    print(f"User: {msg}")

    if msg.lower() == "clear":
        if os.path.exists("repo"):
            shutil.rmtree("repo", onerror=force_remove_readonly)  # ✅ Windows-safe
        return "Cleared!"

    result = qa(msg)
    print(f"Answer: {result['answer']}")
    return str(result["answer"])


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)  # ✅ Add use_reloader=False