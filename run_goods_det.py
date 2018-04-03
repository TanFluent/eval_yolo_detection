import json

from flask import Flask, request

from goods_detect import *

app = Flask(__name__)



@app.after_request
def add_header(response):
    response.headers['Content-Type'] = 'application/json"'
    return response


@app.route("/detection/goods_detect", methods=['POST'])
def goods_detect():
    request_info = request.get_json()
    urls = request_info.get("urls")

    results = goods_detect_urls(urls)

    return json.dumps({
        "code": 0,
        "msg": "",
        "data": results
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0")
