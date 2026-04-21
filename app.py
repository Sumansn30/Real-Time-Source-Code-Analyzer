from langchain_community.vectorstores import Chroma
from src.helper import load_embedding, repo_ingestion
from dotenv import load_dotenv
import os
from flask import Flask, request, jsonify, render_template
from langchain_openai import ChatOpenAI                         # ✅
from langchain.chains import ConversationalRetrievalChain       # ✅ langchain NOT langchain_community
from langchain.memory import ConversationSummaryMemory          # ✅ langchain NOT langchain_community       # ✅ Fixed (not langchain_community)

app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')


load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


embeddings = load_embedding()
persist_directory = "db"

# Now we can load the persisted database from disk, and use it as normal.
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings)



llm = ChatOpenAI()
memory = ConversationSummaryMemory(llm=llm, memory_key = "chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(llm, retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k":8}), memory=memory)




@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')



@app.route('/chatbot', methods=["GET", "POST"])
def gitRepo():

    if request.method == 'POST':
        user_input = request.form['question']
        repo_ingestion(user_input)
        os.system("python store_index.py")

    return jsonify({"response": str(user_input) })




@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)

    if input == "clear":
        os.system("rm -rf repo")

    result = qa(input)
    print(result['answer'])
    return str(result["answer"])



if __name__ == '__main__':
   app.run(host="127.0.0.1", port=5000, debug=True)