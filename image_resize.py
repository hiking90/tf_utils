import tensorflow as tf
import argparse
import sys, glob, os

FLAGS = None

TARGET_WIDTH = 320
TARGET_HEIGHT = 240

def read_image(in_file):
    return tf.image.decode_png(tf.read_file(in_file))

def resize_image(image):
    return tf.cast(tf.image.resize_images(image, [TARGET_HEIGHT, TARGET_WIDTH]) * 255, tf.uint16)

def save_image(filename, image):
    image = tf.image.encode_png(image)
    return tf.write_file(filename, image)

parser = argparse.ArgumentParser()
parser.add_argument('--srcdir', type=str, default='./', 
    help='Directory for input images')
parser.add_argument('--dstdir', type=str, default='./',
    help='Directory for output images')
FLAGS, unparsed = parser.parse_known_args()

filenames = glob.glob(FLAGS.srcdir + '/*.png')

with tf.Graph().as_default():
    in_file = tf.placeholder(tf.string)
    out_file = tf.placeholder(tf.string)

    image = read_image(in_file)
    image = resize_image(image)
    saved = save_image(out_file, image)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    count = 0
    for in_filename in filenames:
        out_filename = FLAGS.dstdir + "/" + os.path.basename(in_filename)
        sess.run(saved, feed_dict={ in_file: in_filename, out_file: out_filename })
        count += 1
        print("\rResize Progress %f %%" % (count/len(filenames)*100), end='')

print('\n')
