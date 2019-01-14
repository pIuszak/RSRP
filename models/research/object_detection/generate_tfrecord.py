"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=images/test_labels.csv  --image_dir=images/test --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('image_dir', '', 'Path to the image directory')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(row_label):

    if row_label == 'speed15':
        return 1
    elif row_label == 'speed30':
        return 2
    elif row_label == 'speed40':
        return 3
    elif row_label == 'speed50':
        return 4
    elif row_label == 'speed60':
        return 5
    elif row_label == 'speed70':
        return 6
    elif row_label == 'speed80':
        return 7
    elif row_label == 'zakaz_skr_w_lewo_prosto':
        return 8
    elif row_label == 'zakaz_skr_w_prawo_prosto':
        return 9
    elif row_label == 'zakaz_prosto':
        return 10
    elif row_label == 'zakaz_lewo':
        return 11
    elif row_label == 'zakaz_lewo_prawo':
        return 12
    elif row_label == 'zakaz_prawo':
        return 13
    elif row_label == 'zakaz_wyprzedzania':
        return 14
    elif row_label == 'zakaz_zawracania':
        return 15
    elif row_label == 'zakaz_wiazdu':
        return 16
    elif row_label == 'zakaz_grania_na_trabce':
        return 17
    elif row_label == 'koniec_zakazu40':
        return 18
    elif row_label == 'koniec_zakazu50':
        return 19
    elif row_label == 'nakaz_prawo_prosto':
        return 20
    elif row_label == 'nakaz_prosto':
        return 21
    elif row_label == 'nakaz_lewo':
        return 22
    elif row_label == 'nakaz_lewo_prawo':
        return 23
    elif row_label == 'nakaz_prawo':
        return 24
    elif row_label == 'nakaz_jazdy_z_lewej':
        return 25
    elif row_label == 'nakaz_jazdy_z_prawej':
        return 26
    elif row_label == 'rondo':
        return 27
    elif row_label == 'droga_ekspresowa':
        return 28
    elif row_label == 'nakaz_grania_na_trabce':
        return 29
    elif row_label == 'droga_dla_rowerow':
        return 30
    elif row_label == 'zawracanie':
        return 31
    elif row_label == 'kurwa_co_to':
        return 32
    elif row_label == 'sygnalizacja':
        return 33
    elif row_label == 'inne_niebezpieczenstwo':
        return 34
    elif row_label == 'przejscie_dla_pieszych':
        return 35
    elif row_label == 'przejscie_dla_rowerow':
        return 36
    elif row_label == 'dzieci':
        return 37
    elif row_label == 'dab_left':
        return 38
    elif row_label == 'dab_right':
        return 39
    elif row_label == 'stromo_down':
        return 40
    elif row_label == 'stromo_up':
        return 41
    elif row_label == 'daj_chinski_sprzedawca_jaj':
        return 42
    elif row_label == 'skret_w_prawo':
        return 43
    elif row_label == 'skret_w_lewo':
        return 44
    elif row_label == 'osiedle':
        return 45
    elif row_label == 'zakrety':
        return 46
    elif row_label == 'piciong':
        return 47
    elif row_label == 'roboty':
        return 48
    elif row_label == 'zakrety_zakrety':
        return 49
    elif row_label == 'przejazd_kol_z_zaporami':
        return 50
    elif row_label == 'wypadki':
        return 51
    elif row_label == 'stop_chin':
        return 52
    elif row_label == 'zakaz_ruchu':
        return 53
    elif row_label == 'zakaz_zatrzymywania':
        return 54
    elif row_label == 'zakaz_wjazdu':
        return 55
    elif row_label == 'costam':
        return 56
    elif row_label == 'zakaz_wjazdu_chin':
        return 57
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(os.getcwd(), FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
