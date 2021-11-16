"""
CS6476: Problem Set 3 Experiment file
This script consists in a series of function calls that run the ps3 
implementation and output images so you can verify your results.
"""

import os
import cv2
import numpy as np

import ps3


IMG_DIR = "input_images"
VID_DIR = "input_videos"
OUT_DIR = "output"
if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)

def video_processor(video_name, fps, frame_ids, my_video):
    video = os.path.join(VID_DIR, video_name)
    image_gen = ps3.video_frame_generator(video)

    image = image_gen.__next__()
    bh, bw, bd = image.shape
    out_path = "output/part6.mp4"
    video_out = mp4_video_writer(out_path, (bw, bh), fps)

    ad_video = os.path.join(VID_DIR, my_video)
    ad_gen = ps3.video_frame_generator(ad_video)
    ad = ad_gen.__next__()
    src_points = ps3.get_corners_list(ad)

    frame_num = 0
    output_counter = 1
    while image is not None and ad is not None:
        print("Processing fame {}".format(frame_num))

        markers = ps3.find_markers(image)

        homography = ps3.find_four_point_transform(src_points, markers)
        image = ps3.project_imageA_onto_imageB(ad, image, homography)

        if frame_num in frame_ids:
            out_str = "ps3-6-a-{}.png".format(output_counter)
            save_image(out_str, image)
            output_counter += 1

        video_out.write(image)

        image = image_gen.__next__()
        ad = ad_gen.__next__()
        frame_num += 1

    video_out.release()


def mp4_video_writer(filename, frame_size, fps=20):
    """Opens and returns a video for writing.

    Use the VideoWriter's `write` method to save images.
    Remember to 'release' when finished.

    Args:
        filename (string): Filename for saved video
        frame_size (tuple): Width, height tuple of output video
        fps (int): Frames per second
    Returns:
        VideoWriter: Instance of VideoWriter ready for writing
    """
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)


def save_image(filename, image):
    """Convenient wrapper for writing images to the output directory."""
    cv2.imwrite(os.path.join(OUT_DIR, filename), image)


def mark_location(image, pt):
    """Draws a dot on the marker center and writes the location as text nearby.

    Args:
        image (numpy.array): Image to draw on
        pt (tuple): (x, y) coordinate of marker center
    """
    color = (0, 50, 255)
    cv2.circle(image, pt, 3, color, -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "(x:{}, y:{})".format(*pt), (pt[0]+15, pt[1]), font, 0.5, color, 1)

def part_6():

    print("\nPart 6:")

    video_file = "ps3-4-a.mp4"
    my_video = "my-ad.mp4"  # Place your video in the input_video directory
    frame_ids = [355, 555, 725]
    fps = 40

    video_processor(video_file, fps, frame_ids, my_video)


if __name__ == '__main__':
    # Comment out the sections you want to skip
    part_6()
