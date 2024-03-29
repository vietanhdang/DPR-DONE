
import os
import json
import re
from entity.question import Question, Option, AnsweringResponse
from model.pipeline import get_answer
from model.compare import Comparer
from model.preprocessing import preprocess
from model.document import Document, rawtxt_to_document
from flask import stream_with_context, make_response
from flask_restful import Resource, Api
from flask import Flask, flash, request, redirect, url_for, jsonify
from model.pipeline import get_pipeline

comparer = Comparer()

app = Flask(__name__)
api = Api(app)


@app.route('/knowledge', methods=['POST'])
def upload_knowledge():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    document = Document(file.filename)
    if os.path.isfile(document.path_txt):
        return make_response(jsonify(message=f"knowledge {document.name} exists"), 400)
    print(f"creating...")
    document = rawtxt_to_document(file.stream, file.filename)
    print(f"created {document.path_txt} success, encoding...")
    preprocess(document.path_txt, pattern="file", delete_all_document=False)
    return make_response(jsonify(message=f"Upload and encode {document.name} success"), 200)


@app.route('/knowledge', methods=['DELETE'])
def delete_knowledge():
    filename = request.form["name"]
    document = Document(filename)
    if not os.path.isfile(document.path_txt):
        return make_response(jsonify(message=f"knowledge {document.name} does not exists"), 400)
    if os.path.isfile(document.path_txt):
        os.remove(document.path_txt)
    preprocess(document.folder, pattern="folder", delete_all_document=True)
    return make_response(jsonify(message=f"delete {document.name} success"), 200)


@app.route('/qa', methods=['POST'])
def qa_res():
    def question_respond():
        json_questions = json.loads(json.dumps(request.json))
        questions = []
        answers_response = []
        for json_question in json_questions:
            options = []
            for json_option in json_question['options']:
                option = Option(json_option['key'], json_option['content'])
                options.append(option)

            question = Question(
                json_question['qn'],
                json_question['content'],
                options)
            questions.append(question)

        pipe = get_pipeline()

        for question in questions:
            query = re.sub(r"[._]{5,}", " what ", question.content)
            query = re.sub(r"\s+", " ", query.strip())
            prediction = get_answer(query, pipe=pipe)
            contexts = prediction["answers"][0]["context"].strip()
            options = []
            max = 0
            i = 0
            key = ""
            for json_option in json_question['options']:
                i += 1
                option = Option(json_option['key'], json_option['content'])
                score = comparer.compare(contexts, option.content)
                if score > max:
                    max = score
                    key = option.key
                if(i == 4):
                    i = 0
                    max = 0
            answer = AnsweringResponse(question.qn, key)
            answers_response.append(answer)
            yield json.dumps(answer.__dict__)

    return app.response_class(stream_with_context(question_respond()))


if __name__ == '__main__':

    app.run(port='5000')
