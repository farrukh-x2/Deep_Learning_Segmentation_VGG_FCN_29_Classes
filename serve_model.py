import sys
import os
import flask
from flask import render_template, send_from_directory, request, redirect,url_for
from werkzeug import secure_filename
from flask import jsonify
import base64
import tensorflow as tf 
import numpy as np
import cv2
import time
# Obtain the flask app object
app = flask.Flask(__name__)
import scipy.misc
from collections import namedtuple

#Retrieved from https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
Label = namedtuple( 'Label' , ['name','color'] )
labels_Segments = [
    #       name                       color
    Label(  'unlabeled'            , (  0,  0,  0) ),
    Label(  'dynamic'              , (111, 74,  0) ),
    Label(  'ground'               , ( 81,  0, 81) ),
    Label(  'road'                 , (128, 64,128) ),
    Label(  'sidewalk'             , (244, 35,232) ),
    Label(  'parking'              , (250,170,160) ),
    Label(  'rail track'           , (230,150,140) ),
    Label(  'building'             , ( 70, 70, 70) ),
    Label(  'wall'                 , (102,102,156) ),
    Label(  'fence'                , (190,153,153) ),
    Label(  'guard rail'           , (180,165,180) ),
    Label(  'bridge'               , (150,100,100) ),
    Label(  'tunnel'               , (150,120, 90) ),
    Label(  'pole'                 , (153,153,153) ),
    Label(  'traffic light'        , (250,170, 30) ),
    Label(  'traffic sign'         , (220,220,  0) ),
    Label(  'vegetation'           , (107,142, 35) ),
    Label(  'terrain'              , (152,251,152) ),
    Label(  'sky'                  , ( 70,130,180) ),
    Label(  'person'               , (220, 20, 60) ),
    Label(  'rider'                , (255,  0,  0) ),
    Label(  'car'                  , (  0,  0,142) ),
    Label(  'truck'                , (  0,  0, 70) ),
    Label(  'bus'                  , (  0, 60,100) ),
    Label(  'caravan'              , (  0,  0, 90) ),
    Label(  'trailer'              , (  0,  0,110) ),
    Label(  'train'                , (  0, 80,100) ),
    Label(  'motorcycle'           , (  0,  0,230) ),
    Label(  'bicycle'              , (119, 11, 32) ),
]

def infer_segments(img, im_softmax, labels, image_shape):
    street_im = scipy.misc.toimage(img)
    for i in range(len(labels)):
        filterr = im_softmax[0][:,i].reshape(image_shape[0], image_shape[1])
        segmentation = (filterr > 0.50).reshape(image_shape[0], image_shape[1], 1)

        colr = np.concatenate((np.array([labels[i].color]),[[255]]),axis = 1)
        mask = np.dot(segmentation, colr)
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im.paste(mask, box=None, mask=mask)
    return scipy.misc.toimage(cv2.addWeighted(img, 1, np.array(street_im), 0.99, 0))

UPLOAD_FOLDER='static/save_images'
IMAGE_FOLDER='images'
def load_graph(trained_model):   

    with tf.gfile.GFile('SavedModels/optimized_model150/FCNVGG_graph_optimized.pb', 'rb') as f:
       graph_def_optimized = tf.GraphDef()
       graph_def_optimized.ParseFromString(f.read())


    with tf.Graph().as_default() as graph:
            restored_logits, = tf.import_graph_def(graph_def_optimized, return_elements=['softmax:0'])

    print('Operations in Optimized Graph:')

    x = graph.get_tensor_by_name('import/image_input:0')
    keep_prob = graph.get_tensor_by_name('import/keep_prob:0')
    sess = tf.Session(graph= graph)

    return sess, restored_logits, x, keep_prob
@app.route('/')
def index():
    return "Webserver is running"

@app.route('/demo',methods=['POST','GET'])
def demo():
    if request.method == 'POST':
        upload_file = request.files['file']
        filename = secure_filename(upload_file.filename)
        upload_file.save(os.path.join(UPLOAD_FOLDER, filename))
        print(filename)
        image_shape = (256, 512)

        sess ,restored_logits, x, keep_prob = app.restored_elements
        
        test_image = scipy.misc.imresize(scipy.misc.imread(upload_file), image_shape)
        toc = time.time()
        im_softmax = sess.run([(restored_logits)], feed_dict={x: [test_image], keep_prob:1.0 })
        tic = time.time()
        imgg = infer_segments(test_image, im_softmax, labels_Segments, image_shape)#75
        
        
        scipy.misc.imsave('static/save_images/temp_image_ori'+str(tic)+'.png', test_image)
        imgg.save('static/save_images/temp_image'+str(tic)+'.png')
        timee = np.around(tic-toc,3)
        print('inference times', timee)
   
        return render_template('result.html', image_path='static/save_images/temp_image_ori'+str(tic)+'.png',
                               image_path_2='static/save_images/temp_image'+str(tic)+'.png', timee = timee)
        
    return  '''
    <!doctype html>
    <html lang="en">
    <head>
      <title>Running my first AI Demo</title>
    </head>
    <body>
    <div class="site-wrapper">
        <div class="cover-container">
            <nav id="main">
                <a href="http://localhost:5000/demo" >HOME</a>
            </nav>
          <div class="inner cover">

          </div>
          <div class="mastfoot">
          <hr />
            <div class="container">
              <div style="margin-top:1%">
		            <h1 style="color:black">29 Class Image Segmentation</h1>
		            <h4 style="color:black">Upload new Image </h4>
		            <form method=post enctype=multipart/form-data>
	                 <p><input type=file name=file>
        	        <input type=submit style="color:black;" value=Upload>
		            </form>
	            </div>	
            </div>
        	</div>
     </div>
   </div>
</body>
</html>

    '''




app.restored_elements=load_graph('SavedModels/optimized_model/FCNVGG_graph_optimized.pb')  
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int("5000"), debug=True, use_reloader=False)
    

