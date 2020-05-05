#!/usr/bin/python
"""
Sample usage:
  ./preprocess_imagenet_validation_data.py ILSVRC2012_img_val \
  imagenet_2012_validation_synset_labels.txt
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import os.path

import sys

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Invalid usage\n'
              'usage: preprocess_imagenet_validation_data.py '
              '<validation data dir> <validation labels file>')
        sys.exit(-1)
    data_dir = sys.argv[1]
    validation_labels_file = sys.argv[2]

    # Read in the 50000 synsets associated with the validation data set.
    labels = []
    for label in open(validation_labels_file).readlines():
        labels.append(str(int(label.strip()) + 1))
    unique_labels = set(labels)

    # Make all sub-directories in the validation data dir.
    for label in unique_labels:
        labeled_data_dir = os.path.join(data_dir, label)
        # Catch error if sub-directory exists
        try:
            os.makedirs(labeled_data_dir)
        except OSError as e:
            # Raise all errors but 'EEXIST'
            if e.errno != errno.EEXIST:
                raise

    # Move all of the image to the appropriate sub-directory.
    for i in range(len(labels)):
        basename = 'ILSVRC2012_val_000%.5d.JPEG' % (i + 1)
        original_filename = os.path.join(data_dir, basename)
        if not os.path.exists(original_filename):
            print('Failed to find: %s' % original_filename)
            sys.exit(-1)
        new_filename = os.path.join(data_dir, labels[i], basename)
        os.rename(original_filename, new_filename)
