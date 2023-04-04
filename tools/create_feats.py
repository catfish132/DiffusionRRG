import os
import base64
import numpy as np
import csv
import sys
import argparse

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']


def main(args):
    count = 0
    with open(args.infeats, "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
            if count % 1000 == 0:
                print(count)
            count += 1

            item['image_id'] = int(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
                item[field] = np.frombuffer(base64.decodebytes(item[field].encode()),
                                            dtype=np.float32).reshape((item['num_boxes'], -1))
            image_id = item['image_id']

            feats = item['features']
            np.savez_compressed(os.path.join(args.outfolder, str(image_id)), feat=feats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--infeats',
                        default="/home/jinyuda/dataset/xmodaler/mscoco/object_feature/test2014/test2014_resnet101_faster_rcnn_genome.tsv.0",
                        help='image features')
    parser.add_argument('--outfolder', default='/home/jinyuda/dataset/xmodaler/mscoco/object_feature/processed_updown/',
                        help='output folder')

    args = parser.parse_args()
    main(args)
