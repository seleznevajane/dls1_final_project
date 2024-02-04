from mmdet.apis import DetInferencer
import cv2


class process_image:
    config_file = 'rtmdet_tiny_8xb32-300e_coco.py'
    checkpoint_file = 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
    inferencer = DetInferencer(model=config_file, weights=checkpoint_file)

    labels_list = {
        14: 'bird',
        15: 'cat',
        16: 'dog',
        17: 'horse',
        18: 'sheep',
        19: 'cow',
        20: 'elephant',
        21: 'bear',
        22: 'zebra',
        23: 'giraffe'
    }

    links_list = {
        14: 'https://en.wikipedia.org/wiki/Bird',
        15: 'https://en.wikipedia.org/wiki/Cat',
        16: 'https://en.wikipedia.org/wiki/Dog',
        17: 'https://en.wikipedia.org/wiki/Horse',
        18: 'https://en.wikipedia.org/wiki/Sheep',
        19: 'https://en.wikipedia.org/wiki/Cow',
        20: 'https://en.wikipedia.org/wiki/Elephant',
        21: 'https://en.wikipedia.org/wiki/Bear',
        22: 'https://en.wikipedia.org/wiki/Zebra',
        23: 'https://en.wikipedia.org/wiki/Giraffe'
    }

    @classmethod
    def detect_animals(cls, *args, **kwargs):
        return cls.inferencer(*args, **kwargs)
    
    @classmethod
    def process_detection_result(cls, image_path, detection_result, output_image_path, threshold=0.4):
        # Load the image
        img = cv2.imread(image_path)

        # Prepare text with more information on found animals
        text_data = []
        coords_data = []
        # Flags for different animals to avoid info duplication
        is_new_animals = [True for i in range(10)] 

        # Draw bounding boxes on the image
        for i in range(len(detection_result['scores'])):
            # If a box has low confidence then we can stop
            # because boxes are sorted by confidence
            if (detection_result['scores'][i] < threshold):
                break
            # If a box is not of an animal then skip it
            if (detection_result['labels'][i] >= 24
                or detection_result['labels'][i] < 14):
                continue

            bbox = detection_result['bboxes'][i]
            label_cd = detection_result['labels'][i]

            # Convert float coordinates to integers
            bbox = [int(coord) for coord in bbox]

            # Draw bounding box
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

            # Display label
            cv2.putText(img, cls.labels_list[label_cd], (bbox[0] + 2, bbox[3] - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            bbox = ','.join([str(coord * 600 // img.shape[1]) for coord in bbox])
            coords_data.append([cls.links_list[label_cd], bbox])

            # If it is a new animal then we add information to the text and mark this animal found
            if (is_new_animals[label_cd - 14]):
                text_data.append(['You see a ' + cls.labels_list[label_cd] + '.', cls.links_list[label_cd]])
                is_new_animals[label_cd - 14] = False

        # Save the annotated image to a file
        cv2.imwrite(output_image_path, img)

        # If no animals were found fill text_data
        if len(text_data):
            pass
            # text_data = ''.join(text_data)
        else:
            text_data = [['No animals found. Please, upload a new photo.', '']]
        
        return text_data, coords_data