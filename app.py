#必要なモジュールをインポート
import os
import sys 


from flask import (
    Flask, 
    request, 
    redirect, 
    url_for, 
    make_response, 
    jsonify, 
    render_template, 
    send_from_directory)

import tensorflow.compat.v1 as tf
#顔認識用のvggモデル
from keras_vggface import vggface
#vggface.pyをvggという名前でインポート
import vggface as vgg
from scipy.spatial.distance import cosine
#画像の前処理に用いるモジュール

global graph

def get_similarity(face_vector1, face_vector2):
    return 1 - cosine(face_vector1, face_vector2)

# 画像のアップロード先のディレクトリ
UPLOAD_FOLDER_ENTER = './image_enter' #芸能人の写真用
UPLOAD_FOLDER_USER_FACE = './image_user' #ユーザーの写真用
FROM_PATH_TO_VECTOR = {} #画像パスとベクトルを紐つけるための空の辞書

#FlaskでAPIを書くときのおまじない
app = Flask(__name__)

#アプリのホーム画面のhtmlを返すURL
@app.route('/')
def index():
    return render_template(
        'index.html',
         enter_images=os.listdir(UPLOAD_FOLDER_ENTER)[::-1],
          user_images=os.listdir(UPLOAD_FOLDER_USER_FACE)[::-1]
          )

@app.route('/upload', methods=['GET', 'POST'])
def uploads_file():
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        if 'upload_files' not in request.files:
            print("ファイルなし")
            return redirect(request.url)

        if request.files.getlist('upload_files')[0].filename:
            #画像オブジェクトを受け取る。
            uploads_files = request.files.getlist('upload_files')
            for uploads_file in uploads_files:
                #それぞれの画像に対してimage_enterまでのパスを定義作成してsaveメソッドを用いて保存する。
                img_path = os.path.join(UPLOAD_FOLDER_ENTER, uploads_file.filename)
                uploads_file.save(img_path)

        uploads_files_path = [ os.path.join(UPLOAD_FOLDER_ENTER, uploads_file.filename) for uploads_file in uploads_files]
        #それぞれの画像のパスから画像サイズを224×224で指定して読み込む
        face_imgs = [vgg.image.load_img(image_path, target_size=(224,224)) for image_path in uploads_files_path]
        #224×224に整形した画像データを行列データに変換する。
        face_img_arrays = [vgg.image.img_to_array(face) for face in face_imgs]
        enter_face_array = vgg.preprocess_input(face_img_arrays, version=2)

        graph = tf.get_default_graph()
        model = vggface.VGGFace(model='resnet50',
            include_top=False,
            input_shape=(224, 224, 3),
            pooling='avg')
        with graph.as_default():
            user_face_vector = model.predict(enter_face_array)

            #face vectorと顔画像パスとのマッピング
            for i, vector in enumerate(user_face_vector):
                FROM_PATH_TO_VECTOR[uploads_files_path[i]] = vector
    return redirect('/')

#ユーザの画像をアップロードするURL
@app.route('/upload_user', methods=['GET', 'POST'])
def upload_user_files():
    if request.method == 'POST':
        upload_file = request.files['upload_user_file']
        img_path = os.path.join(UPLOAD_FOLDER_USER_FACE,upload_file.filename)
        upload_file.save(img_path)
        #load_imgを用いて画像を224×224サイズに整形して読み込み
        #img_to_arrayを用いて画像データを行列化 
        user_face = [vgg.image.img_to_array(vgg.image.load_img(img_path, target_size=(224,224)))]
        #行列データの型をfloat32に変換
        user_face_array = vgg.np.asarray(user_face, 'float32')
        #preprocess_inputを用いて画像行列を正規化
        user_face_array = vgg.preprocess_input(user_face_array, version=2)

        graph = tf.get_default_graph()
        model = vggface.VGGFace(model='resnet50',
            include_top=False,
            input_shape=(224, 224, 3),
            pooling='avg')

        with graph.as_default():
            user_face_vector = model.predict(user_face_array)

             #最も似ている芸能人の顔写真へのパスと類似度を保存するための変数
            most_similar_img = ''
            max_similarity = 0
            for path, vector in FROM_PATH_TO_VECTOR.items():
                if get_similarity(user_face_vector, vector) > max_similarity:
                    max_similarity = get_similarity(user_face_vector, vector)
                    most_similar_img = path

            filename = most_similar_img.split('/')[-1]
            return render_template(
            'result.html',
            filename=filename,
            score=max_similarity
            )

#ディレクトリに保存されている画像をブラウザに送る処理
@app.route('/images/<path:path>')
def send_image(path):
    return send_from_directory(UPLOAD_FOLDER_ENTER, path)

#スクリプトからAPIを叩けるようにします。
if __name__ == "__main__":
    app.run(debug=True)