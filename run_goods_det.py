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
    #            1   2   3   4   5    6  7   8   9   10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26
    each_thres=[.65,.65,.65,.65,.65,.40,.45,.50,.55,.50,.65,.65,.65,.65,.65,.60,.65,.65,.50,.65,.65,.55,.65,.65,.65,.65]
    #each_thres=[.6,.6,.6,.6,.6]
    results = goods_detect_urls(urls,conf_thres=each_thres)

    return json.dumps({
        "code": 0,
        "msg": "",
        "data": results
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0")
