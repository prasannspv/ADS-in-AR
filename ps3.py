import cv2
import numpy as np


def euclidean_distance(p0, p1):
    """Gets the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1.
        p1 (tuple): Point 2.

    Return:
        float: The distance between points
    """

    return np.sqrt((p1[0] - p0[0])**2 + (p1[1]-p0[1])**2)


def get_corners_list(image):
    """Returns a ist of image corner coordinates used in warping.

    These coordinates represent four corner points that will be projected to
    a target image.

    Args:
        image (numpy.array): image array of float64.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """

    height, width = image.shape[0], image.shape[1]
    return [(0, 0), (0, height-1), (width-1, 0), (width-1, height-1)]


def is_duplicate(mid_point, point_list):
    for point in point_list:
        if euclidean_distance(point, mid_point) <= 10:
            return True
    return False


def find_markers(image, template=None):
    """Finds four corner markers.

    Use a combination of circle finding, corner detection and convolution to
    find the four markers in the image.

    Args:
        image (numpy.array): image array of uint8 values.
        template (numpy.array): template image of the markers.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """

    image = add_noise_to_image(image)
    corners = get_harris_corners(image)
    _, _, centroids = cv2.kmeans(np.asarray(corners).astype(np.float32), 4, None,
               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)

    centroids = sorted([tuple(l) for l in np.array(centroids).astype(int)], key = lambda x:x[0])
    top_left, bottom_left = sorted(centroids[:2], key = lambda x:x[1])
    top_right, bottom_right = sorted(centroids[2:], key = lambda x:x[1])
    return top_left, bottom_left, top_right, bottom_right


def get_harris_corners(image):
    corners = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (13, 13), 25)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 7, 3, 0.1)
    dst = cv2.dilate(dst, None)
    locs = np.where(dst > 0.15 * dst.max())
    for (i, j) in zip(*locs):
        corners.append((j, i))
    return corners


def draw_box(image, markers, thickness=1):
    """Draws lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line, leave the default "lineType" and Pass the thickness
    parameter from this function.

    Args:
        image (numpy.array): image array of uint8 values.
        markers(list): the points where the markers were located.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with lines drawn.
    """

    img_out = image.copy()
    top_left, bottom_left, top_right, bottom_right = markers
    cv2.line(img_out, top_left, bottom_left, (0, 255, 0), thickness = thickness)
    cv2.line(img_out, top_left, top_right, (0, 255, 0), thickness = thickness)
    cv2.line(img_out, top_right, bottom_right, (0, 255, 0), thickness = thickness)
    cv2.line(img_out, bottom_left, bottom_right, (0, 255, 0), thickness = thickness)
    return img_out


def project_imageA_onto_imageB(imageA, imageB, homography):
    """Projects image A into the marked area in imageB.

    Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """
    height, width, _ = imageA.shape
    dst_height, dst_width, _ = imageB.shape
    inv_homography = np.linalg.inv(homography)
    for x in range(0, dst_width):
        for y in range(0, dst_height):
            projected_point = np.dot(inv_homography, np.array([x, y, 1]))
            x_p, y_p = int(projected_point[0]/projected_point[2]), int(projected_point[1]/projected_point[2])
            if 0 < x_p < width and 0 < y_p < height:
                intensity_values = imageA[y_p, x_p]
                imageB[y, x] = intensity_values
    return imageB


def find_four_point_transform(src_points, dst_points):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform

    Hint: You will probably need to use least squares to solve this.

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """
    matrix = []
    for (sx,sy), (dstx, dsty) in zip(src_points, dst_points):
        matrix.extend([[sx, sy, 1, 0, 0, 0, -dstx * sx, -dstx * sy],
                        [0, 0, 0, sx, sy, 1, -dsty * sx, -dsty * sy]])
    matrix = np.matrix(matrix).astype(np.float32)
    solution_matrix, _, _, _ = np.linalg.lstsq(matrix, np.reshape(dst_points, 8), rcond=-1)
    return np.append(solution_matrix, [1]).reshape((3, 3))


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    video = cv2.VideoCapture(filename)
    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    video.release()
    yield None
