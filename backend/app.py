# 创建flask服务器
from flask_cors import CORS
from flask import Flask, request, jsonify

app = Flask(__name__)
CORS(app)

#测试flask接口
@app.route('/api/test', methods=['get'])
def test():
    return jsonify({"success":True}), 200

# 运行Flask应用
if __name__ == '__main__':
    app.run(debug=True)