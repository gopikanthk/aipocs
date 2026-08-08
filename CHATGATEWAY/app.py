from flask import Flask, render_template, request, jsonify
from groak_adapter import GroakClient
from security_gateway import SecurityGateway, decision_metadata
import logging

app = Flask(__name__)
client = GroakClient()
security_gateway = SecurityGateway()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/message', methods=['POST'])
def message():
    data = request.json or {}
    user = data.get('message', '')
    attack = data.get('attack', '')
    conversation = data.get('conversation', [])

    prompt = user
    if attack:
        prompt = prompt + "\n\n---ATTACK PAYLOAD---\n" + attack

    security_decision = security_gateway.evaluate(prompt)
    security_meta = decision_metadata(security_decision)
    if security_decision.decision == 'BLOCK':
        return jsonify({
            'response': 'Request blocked by the prompt-injection firewall.',
            'meta': security_meta,
        }), 403

    response_text, model_meta = client.respond(prompt, conversation)
    return jsonify({'response': response_text, 'meta': {**security_meta, **model_meta}})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
