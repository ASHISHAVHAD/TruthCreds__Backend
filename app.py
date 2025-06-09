from flask import Flask, request
from flask_cors import CORS
import joblib
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
import pandas as pd
import json
from supadata import Supadata, SupadataError

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
CORS(app, resources={r"/*": {"origins": "https://truthcreds.netlify.app"}})


model = joblib.load("model.pkl")

supadata = Supadata(api_key="sd_e789b55e9041ad706b989a3f59ced98c")

def extract_text(start, end, transcript):
    start_time = start * 1000
    end_time = (end + 1)* 1000
    text = ''
    for textLine in transcript.content:
        if(textLine.offset >= start_time and textLine.offset <= end_time):
            text = text + textLine.text
        elif(textLine.offset > end_time):
            break

    return text

@app.route("/", methods = ["POST"])
def default():
    data = request.get_json()
    claim = data["claim"]
    return claim

@app.route("/getLength/", methods = ["POST"])
def getLength():
    
    data = request.get_json()
    link = data["link"]
    video_id = ''
    count = 0

    for letter in link:
        if letter == '?':
            break
        if count == 3:
            video_id = video_id + letter
        if letter == '/':
            count = count + 1
            
    video = supadata.youtube.video(id=link)

    return {"length" : video.duration}

@app.route("/youtubeLink/", methods = ["POST"])
def youtubeLink():
    data = request.get_json()
    link = data["link"]
    start = data["start"]
    end = data["end"]
    video_id = ''
    count = 0
    for letter in link:
        if letter == '?':
            break
        if count == 3:
            video_id = video_id + letter
        if letter == '/':
            count = count + 1

    transcript = supadata.youtube.transcript(video_id=video_id)

    claim_text = extract_text(start, end, transcript)
    
    genai.configure(api_key= "AIzaSyBn9UtZUXB4fMKGgs5KMNvVG0zBXvGIX2s")#os.environ["GEMINI_API_KEY"])

    generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_schema": content.Schema(
        type = content.Type.OBJECT,
        required = ["claims"],
        properties = {
        "claims": content.Schema(
            type = content.Type.ARRAY,
            items = content.Schema(
            type = content.Type.OBJECT,
            required = ["scores", "claim_text", "claim_validity", "support", "oppose"],
            properties = {
                "scores": content.Schema(
                type = content.Type.OBJECT,
                required = ["supporting_evidence_reliability", "supporting_evidence_accuracy", "supporting_evidence_sentiment_score", "opposing_evidence_reliability", "opposing_evidence_accuracy", "opposing_evidence_sentiment_score"],
                properties = {
                    "supporting_evidence_reliability": content.Schema(
                    type = content.Type.NUMBER,
                    ),
                    "supporting_evidence_accuracy": content.Schema(
                    type = content.Type.NUMBER,
                    ),
                    "supporting_evidence_sentiment_score": content.Schema(
                    type = content.Type.NUMBER,
                    ),
                    "opposing_evidence_reliability": content.Schema(
                    type = content.Type.NUMBER,
                    ),
                    "opposing_evidence_accuracy": content.Schema(
                    type = content.Type.NUMBER,
                    ),
                    "opposing_evidence_sentiment_score": content.Schema(
                    type = content.Type.NUMBER,
                    ),
                },
                ),
                "claim_text": content.Schema(
                type = content.Type.STRING,
                ),
                "claim_validity": content.Schema(
                type = content.Type.STRING,
                ),
                "support": content.Schema(
                type = content.Type.ARRAY,
                items = content.Schema(
                    type = content.Type.OBJECT,
                    required = ["source_name", "source_link", "source_cred_score", "supporting_text_from_source"],
                    properties = {
                    "source_name": content.Schema(
                        type = content.Type.STRING,
                    ),
                    "source_link": content.Schema(
                        type = content.Type.STRING,
                    ),
                    "source_cred_score": content.Schema(
                        type = content.Type.INTEGER,
                    ),
                    "supporting_text_from_source": content.Schema(
                        type = content.Type.STRING,
                    ),
                    },
                ),
                ),
                "oppose": content.Schema(
                type = content.Type.ARRAY,
                items = content.Schema(
                    type = content.Type.OBJECT,
                    required = ["source_name", "source_link", "source_cred_score", "opposing_text_from_source"],
                    properties = {
                    "source_name": content.Schema(
                        type = content.Type.STRING,
                    ),
                    "source_link": content.Schema(
                        type = content.Type.STRING,
                    ),
                    "source_cred_score": content.Schema(
                        type = content.Type.INTEGER,
                    ),
                    "opposing_text_from_source": content.Schema(
                        type = content.Type.STRING,
                    ),
                    },
                ),
                ),
            },
            ),
        ),
        },
    ),
    "response_mime_type": "application/json",
    }

    model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="Do not provide dummy data in any case",
    )

    chat_session = model.start_chat(
    history=[
        {
        "role": "user",
        "parts": [
            "Assume role of claim verifying agent. Consider this text \"\". \n\nProcess the text and extract as many claims and facts as possible from the text. Extract minimum 10 claims from text. For each claim do the following steps:\n1. Find at least 3 and upto 5 evidences that support the claim. Then find atleast 3 and upto 5 evidences that oppose the claim. Do not search for evidences in given text. You can search on internet, research papers, publications, books, etc. Do not provide dummy sources. Provide link of sources. Also provide the exact text from source that supports or opposes the given claim.\n2. For each evidence calculate following values: supporting_evidence_reliability, supporting_evidence_accuracy, supporting_evidence_sentiment_score, opposing_evidence_reliability, opposing_evidence_accuracy, opposing_evidence_sentiment_score. The values should be in range 0 to 1.\n3. Take the average of above values of all the evidences.\n\nReturn the output in specified format.\nSource cred score represents the credibility of the source of evidence. Higher the source's trustworthiness, higher the score. Its value is between 0 to 100.",
        ],
        },
    ]
    )

    response = chat_session.send_message(claim_text)

    model = joblib.load("model.pkl")

    result = json.loads(response.text)

    for claim in result['claims']:
        df = pd.DataFrame({
            'supporting_evidence_reliability' : claim['scores']['supporting_evidence_reliability'],
            'supporting_evidence_accuracy' : claim['scores']['supporting_evidence_accuracy'],
            'supporting_evidence_sentiment_score' : claim['scores']['supporting_evidence_sentiment_score'],
            'opposing_evidence_reliability' : claim['scores']['opposing_evidence_reliability'],
            'opposing_evidence_accuracy' : claim['scores']['opposing_evidence_accuracy'],
            'opposing_evidence_sentiment_score' : claim['scores']['opposing_evidence_sentiment_score'],
        }, index = [0])

        truth_value = model.predict(df)
        print("done")
        claim['truth_value'] = str(truth_value[0])

    return result

@app.route("/text/", methods = ["POST"])
def text():
    data = request.get_json()
    
    claim_text = data['claim_text']
    
    genai.configure(api_key= "AIzaSyBn9UtZUXB4fMKGgs5KMNvVG0zBXvGIX2s")

    generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_schema": content.Schema(
        type = content.Type.OBJECT,
        required = ["claims"],
        properties = {
        "claims": content.Schema(
            type = content.Type.ARRAY,
            items = content.Schema(
            type = content.Type.OBJECT,
            required = ["scores", "claim_text", "claim_validity", "support", "oppose"],
            properties = {
                "scores": content.Schema(
                type = content.Type.OBJECT,
                required = ["supporting_evidence_reliability", "supporting_evidence_accuracy", "supporting_evidence_sentiment_score", "opposing_evidence_reliability", "opposing_evidence_accuracy", "opposing_evidence_sentiment_score"],
                properties = {
                    "supporting_evidence_reliability": content.Schema(
                    type = content.Type.NUMBER,
                    ),
                    "supporting_evidence_accuracy": content.Schema(
                    type = content.Type.NUMBER,
                    ),
                    "supporting_evidence_sentiment_score": content.Schema(
                    type = content.Type.NUMBER,
                    ),
                    "opposing_evidence_reliability": content.Schema(
                    type = content.Type.NUMBER,
                    ),
                    "opposing_evidence_accuracy": content.Schema(
                    type = content.Type.NUMBER,
                    ),
                    "opposing_evidence_sentiment_score": content.Schema(
                    type = content.Type.NUMBER,
                    ),
                },
                ),
                "claim_text": content.Schema(
                type = content.Type.STRING,
                ),
                "claim_validity": content.Schema(
                type = content.Type.STRING,
                ),
                "support": content.Schema(
                type = content.Type.ARRAY,
                items = content.Schema(
                    type = content.Type.OBJECT,
                    required = ["source_name", "source_link", "source_cred_score", "supporting_text_from_source"],
                    properties = {
                    "source_name": content.Schema(
                        type = content.Type.STRING,
                    ),
                    "source_link": content.Schema(
                        type = content.Type.STRING,
                    ),
                    "source_cred_score": content.Schema(
                        type = content.Type.INTEGER,
                    ),
                    "supporting_text_from_source": content.Schema(
                        type = content.Type.STRING,
                    ),
                    },
                ),
                ),
                "oppose": content.Schema(
                type = content.Type.ARRAY,
                items = content.Schema(
                    type = content.Type.OBJECT,
                    required = ["source_name", "source_link", "source_cred_score", "opposing_text_from_source"],
                    properties = {
                    "source_name": content.Schema(
                        type = content.Type.STRING,
                    ),
                    "source_link": content.Schema(
                        type = content.Type.STRING,
                    ),
                    "source_cred_score": content.Schema(
                        type = content.Type.INTEGER,
                    ),
                    "opposing_text_from_source": content.Schema(
                        type = content.Type.STRING,
                    ),
                    },
                ),
                ),
            },
            ),
        ),
        },
    ),
    "response_mime_type": "application/json",
    }

    model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="Do not provide dummy data in any case,",
    )

    chat_session = model.start_chat(
    history=[
        {
        "role": "user",
        "parts": [
            "Assume role of claim verifying agent. Consider this text \"\". \n\nProcess the text and extract claims from the text. Extract as many claims as possible. Please search the web and verify whether [claim] is true. Use recent news or official sources if possible. For each claim do the following steps:\n1. Find at least 3 and upto 5 evidences that support the claim. Then find atleast 3 and upto 5 evidences that oppose the claim. Do not search for evidences in given text. You can search on internet, research papers, publications, books, etc. Do not provide dummy sources. Provide link of sources. Also provide the exact text from source that supports or opposes the given claim.\n2. For each evidence calculate following values: supporting_evidence_reliability, supporting_evidence_accuracy, supporting_evidence_sentiment_score, opposing_evidence_reliability, opposing_evidence_accuracy, opposing_evidence_sentiment_score. The values should be in range 0 to 1.\n3. Take the average of above values of all the evidences.\n\nReturn the output in specified format.\nSource cred score represents the credibility of the source of evidence. Higher the source's trustworthiness, higher the score. Its value is between 0 to 100.",
        ],
        },
    ]
    )

    response = chat_session.send_message(claim_text)

    model = joblib.load("model.pkl")

    result = json.loads(response.text)

    for claim in result['claims']:
        df = pd.DataFrame({
            'supporting_evidence_reliability' : claim['scores']['supporting_evidence_reliability'],
            'supporting_evidence_accuracy' : claim['scores']['supporting_evidence_accuracy'],
            'supporting_evidence_sentiment_score' : claim['scores']['supporting_evidence_sentiment_score'],
            'opposing_evidence_reliability' : claim['scores']['opposing_evidence_reliability'],
            'opposing_evidence_accuracy' : claim['scores']['opposing_evidence_accuracy'],
            'opposing_evidence_sentiment_score' : claim['scores']['opposing_evidence_sentiment_score'],
        }, index = [0])

        truth_value = model.predict(df)
        print("done")
        claim['truth_value'] = str(truth_value[0])

    return result
    

if __name__ == "__main__":
    app.run()