from detect_league_bars import DetectLeagueBars
import numpy as np
import cv2
import os
import sys

def main(argv):

    filepath = argv[0]

    color_masks = np.load('color_masks.npy')
    white_masks = np.load('white_mask.npy')

    left_char_bar_location = [[0, 137], [83, 642]]
    right_char_bar_location = [[1800, 137], [1920, 642]]
    map_location = [[1653, 815], [1920, 1080]]
    spectator_char_bar_location = [[0, 775], [305, 964]]
    incorrect_string_locations = [left_char_bar_location, map_location, spectator_char_bar_location,
                               right_char_bar_location]

    detector = DetectLeagueBars(color_masks=color_masks, white_masks=white_masks,
                                incorrect_string_locations=incorrect_string_locations,
                                output_image_size=(1280, 720))

    image = cv2.imread(filepath)

    detector.fit_image(image)
    detector.find_bars(5)
    print(detector.show_detected_bars())

if __name__ == "__main__":
    main(sys.argv[1:])