from flask import Flask, jsonify, request
app = Flask(__name__)

# # This is our in memory database placeholder:
# items = [
#     {
#         'id': 1,
#         'title': "Title1",
#         'description': "Description1",
#         'bool': False
#     },
#     {
#         'id': 2,
#         'title': "Title2",
#         'description': "Description2",
#         'bool': True
#     }
# ]


@app.route('/api')
def response():
    # here we want to get the value of user (i.e. ?url=some-value)
    "for tesing example http://127.0.0.1:5000/api?url=https://stackoverflow.com/questions/26472433/return-text-html-content-type-when-using-flask-restful"
    url = request.args.get('url')
    return "Your url is: " + str(url)


if __name__ == '__main__':
    debug = True  # set to true for hot-reload
    app.run(debug=debug)
