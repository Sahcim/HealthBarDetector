import numpy as np
import cv2


class DetectLeagueBars:

    def __init__(self, color_masks=None, white_masks=None, maximum_string_width=120,
                 maximum_bar_width=110, bar_maximum_distance_from_string=30,
                 minimum_bar_width=8, minimum_bar_height=5,
                 incorrect_string_locations=None, output_image_size=(1920, 1080)):
        """Algorithm detecting League of Legends bars

        Parameters:
        ---------
            color_masks: int numpy Array of shape (n_masks, number_of_channels)
                used to filter colors from input image to preprocess image

            white_masks: int numpy Array of shape (n_masks, number_of_channels)
                used to filter colors from input image to create image containing strings

            maximum_bar_width: int
                maximum width of the bar used to decide if bar is too long for it to be suitable

            minimum_bar_width: int
                minimum width of bar used while rating it

            minimum_bar_height: int
                minimum height of bar used while rating it

            maximum_string_width: int
                maximum string width used to decide if bar is below string

            bar_maximum_distance_from_string: int
                used to decide if string is in correct distance from bar

            incorrect_string_locations: int numpy array of shape (n_locations,4)
                locations where we don't want strings to appear

            output_image_size: tuple of size 2
                used for displaying preprocessed image, input image, image with detected bars

        Notes:
        ---------
            Please check readme for explanation how algorithm works
        """

        if color_masks is None:
            raise Exception("You have to load color masks!")

        if white_masks is None:
            raise Exception("You have to load white masks!")

        self.color_masks = color_masks
        self.white_masks = white_masks
        self.maximum_string_width = maximum_string_width
        self.maximum_bar_width = maximum_bar_width
        self.bar_maximum_distance_from_string = bar_maximum_distance_from_string
        self.output_image_size = output_image_size
        self.incorrect_string_locations = incorrect_string_locations
        self.minimum_bar_width = minimum_bar_width
        self.minimum_bar_height = minimum_bar_height

    def fit_image(self, image):
        """Preprocess image and create string image for a detector to detect bars

        Returns:
            self
                Detector ready to find bars
        """

        self.input_image = image
        self.__string_image()
        self.__preprocess_image()
        return self

    def __string_image(self):
        """Create string image for detector

        Filter image with white masks, then make strings connected by blurring image.

        Returns
        ----------
            self
                Detector with string image
        """

        hsv_img = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2HSV)
        # Filter image with white masks
        mask = 0
        for num_of_range in np.arange(self.white_masks.shape[0]):
            mask += cv2.inRange(hsv_img, self.white_masks[num_of_range, 0], self.white_masks[num_of_range, 1])

        result_string_img = cv2.bitwise_and(self.input_image, self.input_image, mask=mask)

        # Blur image to make strings more visable
        gray = cv2.cvtColor(result_string_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.blur(gray, (15, 15))
        thresh = cv2.threshold(blur, 5, 255, cv2.THRESH_BINARY)[1]
        # Fix strings
        dilate_kernel = np.ones((3, 11), np.uint8)
        dilate = cv2.dilate(thresh, dilate_kernel, iterations=1)

        self.string_image = dilate

    def __preprocess_image(self):
        """ Preprocess input image

        Filter image with color masks, then reduce noise and
        fix small holes in bars to make them more visible

        Returns
        ----------
        self
            Detector with preprocessed image
        """

        hsv_img = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2HSV)
        # Filter image with color masks
        mask = 0
        for num_of_range in np.arange(self.color_masks.shape[0]):
            mask += cv2.inRange(hsv_img, self.color_masks[num_of_range], self.color_masks[num_of_range])

        result_color_img = cv2.bitwise_and(self.input_image, self.input_image, mask=mask)

        # threshold image to make bars more visible
        gray = cv2.cvtColor(result_color_img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)[1]

        # Erode image to reduce noise
        erosion_kernel = np.ones((2, 3), np.uint8)
        erosion = cv2.erode(thresh, erosion_kernel, iterations=1)
        # Dilate image to fix bars that were shrunk by erode
        dilate_kernel = np.ones((2, 2), np.uint8)
        dilate = cv2.dilate(erosion, dilate_kernel, iterations=1)

        # Blur and threshold image to make bars bigger
        blur = cv2.blur(dilate, (3, 3))
        thresh = cv2.threshold(blur, 5, 255, cv2.THRESH_BINARY)[1]

        self.preprocessed_image = thresh

    def find_bars(self, number_of_iterations):
        """ Find bar on filtered and denoised image

        Compute all bars contours on pre-processed image, connect disconnected bars,
        filter the unsatisfactory ones.


        Parameters
        ----------
        number_of_iterations: int
            number of iterations for algorithm that fixes disconnected bars

        Returns
        ----------
            self
                Detector with proper bars localization and sizes
        """

        self.__find_strings()

        # Find bars contours
        contours, hierarchy = cv2.findContours(self.preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        bars = []
        for cnt in contours:

            # Find bars bounding rectangle and check if it is big enough
            approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
            rec = cv2.boundingRect(approx)

            if self.__rate_bar(rec, True, True, False, minimum_rating=1):
                bars.append(rec)

        # Fix disconnected bars
        for _ in np.arange(number_of_iterations):
            for bar_1 in bars:
                for bar_2 in bars:
                    if bar_1 != bar_2:

                        # Decide which bar is on the left
                        if bar_1[0] < bar_2[0]:
                            (x1, y1, w1, h1) = bar_1
                            (x2, y2, w2, h2) = bar_2
                        else:
                            (x1, y1, w1, h1) = bar_2
                            (x2, y2, w2, h2) = bar_1

                        upper_edge_y_value_bar_1 = y1
                        upper_edge_y_value_bar_2 = y2

                        bottom_edge_y_value_bar_1 = y1 + h1
                        bottom_edge_y_value_bar_2 = y2 + h2

                        # Check if bar_1 and bar_2 are parts of one bar by
                        # comparing their bottom and upper y_value boundaries
                        if (((upper_edge_y_value_bar_1 <= upper_edge_y_value_bar_2 + 1 and bottom_edge_y_value_bar_1 >= bottom_edge_y_value_bar_2 - 1) or
                             (upper_edge_y_value_bar_1 + 1 >= upper_edge_y_value_bar_2 and bottom_edge_y_value_bar_1 - 1 <= bottom_edge_y_value_bar_2)) and
                                (x2 - x1) + w2 < self.maximum_bar_width):
                            bars.remove(bar_1)
                            bars.remove(bar_2)
                            new_bar = (x1, np.min([y1, y2]), (x2 - x1) + w2, np.max([h1, h2]))
                            bars.append(new_bar)
                            break

        # for every bar check if it's suitable
        best_bars = [bar for bar in bars if self.__rate_bar(bar, minimum_rating=11)]

        self.bars_Locations = best_bars
        return self

    def __find_strings(self):
        """ Find strings on image filtered with white masks and outside of locations to avoid

        Returns
        ----------
            self
                Detector with middle point of suitable strings
        """

        # Find strings contours
        edges = cv2.Canny(self.string_image, 150, 250, apertureSize=3)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        strings_locations = []

        for cnt in contours:

            # Find string bounding rectangle
            is_ok = True
            approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
            rec = cv2.boundingRect(approx)

            (x, y, w, h) = rec
            middle_point = [x + (w / 2), y + (h / 2)]
            # Check if string middle point is inside unsuitable locations
            for incorrect_string_location in self.incorrect_string_locations:

                if incorrect_string_location[0][0] < middle_point[0] < incorrect_string_location[1][0] and \
                        incorrect_string_location[0][1] < middle_point[1] < incorrect_string_location[1][1]:
                    is_ok = False
                    break

            if is_ok:
                strings_locations.append(middle_point)

            self.strings_locations = strings_locations

    def __rate_bar(self, bar, test_width=True, test_height=True, test_is_string_above_bar=True, minimum_rating=0):
        """ Rate bar and decide if it has enough points do be considered as valid

        Parameters
        -----------
            bar : tuple of length (4)
                    bar to be tested

            test_width : bool
                    consider bar width in tests

            test_height : bool
                    consider bar height in tests

            test_height : bool
                    consider bar height in tests

            test_is_string_above_bar : bool
                    consider string location with respect to bar

            minimum_rating : int
                    minimum number of points for bar to be considered as valid

        Returns
        ---------
            decision : bool
                    test if bar has enough points
        """

        rating = 0
        x, y, w, h = bar

        if test_height and h >= self.minimum_bar_height:
            rating += 1
        if test_height and h >= self.minimum_bar_height+4:
            rating += 9
        if test_width and w >= self.minimum_bar_width:
            rating += 1
        if test_width and w >= self.minimum_bar_width+15:
            rating += 14
        if test_is_string_above_bar and self.__is_string_above_bar(bar):
            rating += 20

        return rating >= minimum_rating

    def __is_string_above_bar(self, bar):
        """ Determine if bar is above string

            Parameters
            -----------
                bar : tuple of length (4)
                    bar to be tested

            Returns
            -----------
                decision : bool

        """

        for string in self.strings_locations:

            # Create string boundaries
            (x_s, y_s) = string
            string_left_boundary = x_s - (self.maximum_string_width / 2)
            string_right_boundary = x_s + (self.maximum_string_width / 2)

            x_b, y_b, w_b, h_b = bar
            # Create bar boundaries
            bar_left_boundary = x_b
            bar_right_boundary = x_b + w_b

            distance_from_string = y_b - y_s
            # Check if bar is between string boundaries and is not to far from it
            if (string_left_boundary < bar_left_boundary and string_right_boundary > bar_right_boundary and
                    0 < distance_from_string < self.bar_maximum_distance_from_string):
                return True

        return False

    def show_input_image(self):
        """ Display input image """
        if self.input_image is None:
            raise Exception("Fit image first!")

        input_image_show = cv2.resize(self.input_image, self.output_image_size)
        cv2.imshow("input_image", input_image_show)
        cv2.waitKey()
        cv2.destroyWindow("input_image")

    def show_preprocessed_image(self):
        """ Display input image after pre-processing step """

        if self.input_image is None:
            raise Exception("Fit image first!")

        preprocessed_image_show = cv2.resize(self.preprocessed_image, self.output_image_size)
        cv2.imshow("preprocessed_image", preprocessed_image_show)
        cv2.waitKey()
        cv2.destroyWindow("preprocessed_image")

    def show_detected_bars(self):
        """ Display input image with green rectangles at detected bars locations """

        if self.bars_Locations is None:
            raise Exception("Find bars first!")

        detected_bars_image = self.input_image

        for bar in self.bars_Locations:
            x, y, w, h = bar
            cv2.rectangle(detected_bars_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        detected_bars_image = cv2.resize(detected_bars_image, self.output_image_size)
        cv2.imshow("detected_bars", detected_bars_image)
        cv2.waitKey()
        cv2.destroyWindow("detected_bars")

    def number_of_bars(self):
        """ Display number of detected bars

        Returns
        ---------
            bars : int
                number of detected bars
        """

        if self.bars_Locations is None:
            raise Exception("Find bars first!")

        return len(self.bars_Locations)
