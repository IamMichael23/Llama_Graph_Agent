import os
import time
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from dotenv import load_dotenv, find_dotenv
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from tools import retrieve_Fitting_Instructions, retrieve_Fitted_Products

load_dotenv(find_dotenv())

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "golf-agent-secret-key-change-in-production")
CORS(app)

# Initialize agent with memory
checkpointer = MemorySaver()
agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[retrieve_Fitting_Instructions, retrieve_Fitted_Products],
    checkpointer=checkpointer
)

# Store conversation history per session
conversations = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400

        # Get or create session ID
        if 'session_id' not in session:
            session['session_id'] = f"session_{int(time.time() * 1000)}"

        session_id = session['session_id']

        # Initialize conversation history for new sessions
        if session_id not in conversations:
            conversations[session_id] = []

        # Add user message to history
        conversations[session_id].append({
            'role': 'user',
            'content': user_message,
            'timestamp': time.time()
        })

        # Invoke agent
        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            {"configurable": {"thread_id": session_id}}
        )

        # Extract agent response
        agent_message = response["messages"][-1].content

        # Add agent response to history
        conversations[session_id].append({
            'role': 'agent',
            'content': agent_message,
            'timestamp': time.time()
        })

        return jsonify({
            'response': agent_message,
            'session_id': session_id
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    session_id = session.get('session_id')
    if session_id and session_id in conversations:
        return jsonify({'history': conversations[session_id]})
    return jsonify({'history': []})

@app.route('/api/clear', methods=['POST'])
def clear_history():
    session_id = session.get('session_id')
    if session_id and session_id in conversations:
        conversations[session_id] = []
    session.pop('session_id', None)
    return jsonify({'message': 'History cleared'})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5001)
