import os
import time
import cv2
import imutils
import shutil
import img2pdf
import glob
import argparse
import re

# Define constants

OUTPUT_SLIDES_DIR = f"./output"

# FRAME_RATE = 5, FGBG_HISTORY = FRAME_RATE * 6, MIN_PERCENT = 0.2, MAX_PERCENT = 0.6.

# no.of frames per second that needs to be processed, fewer the count faster the speed
FRAME_RATE = 5
WARMUP = FRAME_RATE              # initial number of frames to be skipped
FGBG_HISTORY = FRAME_RATE * 6   # no.of frames in background object
# Threshold on the squared Mahalanobis distance between the pixel and the model to decide whether a pixel is well described by the background model.
VAR_THRESHOLD = 16
# If true, the algorithm will detect shadows and mark them.
DETECT_SHADOWS = False
# min % of diff between foreground and background to detect if motion has stopped
MIN_PERCENT = 0.2
# max % of diff between foreground and background to detect if frame is still in motion
MAX_PERCENT = 0.6


def get_frames(video_path, start_time=0):
    '''A function to return the frames from a video located at video_path
    this function skips frames as defined in FRAME_RATE'''

    # open a pointer to the video file initialize the width and height of the frame
    vs = cv2.VideoCapture(video_path)
    if not vs.isOpened():
        raise Exception(f'unable to open file {video_path}')

    total_frames = vs.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_time = start_time
    frame_count = 0
    print("total_frames: ", total_frames)
    print("FRAME_RATE", FRAME_RATE)

    # loop over the frames of the video
    while True:
        # grab a frame from the video

        # move frame to a timestamp
        vs.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)
        frame_time += 1/FRAME_RATE

        (_, frame) = vs.read()
        # if the frame is None, then we have reached the end of the video file
        if frame is None:
            break

        frame_count += 1
        yield frame_count, frame_time, frame

    vs.release()

def detect_unique_screenshots(video_path, output_folder_screenshot_path, course_request_id):
    ''''''
    # Initialize fgbg a Background object with Parameters
    # history = The number of frames history that effects the background subtractor
    # varThreshold = Threshold on the squared Mahalanobis distance between the pixel and the model to decide whether a pixel is well described by the background model. This parameter does not affect the background update.
    # detectShadows = If true, the algorithm will detect shadows and mark them. It decreases the speed a bit, so if you do not need this feature, set the parameter to false.

    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=FGBG_HISTORY, varThreshold=VAR_THRESHOLD, detectShadows=DETECT_SHADOWS)

    captured = False
    start_time = time.time()
    (W, H) = (None, None)

    screenshoots_count = 0

    # Find the latest screenshot and extract the frame time from its filename
    screenshots = sorted(glob.glob(f"{output_folder_screenshot_path}/*.png"))
    if screenshots:
        latest_screenshot = screenshots[-1]
        match = re.search(r'_(\d{3})_(\d+)\.png$', latest_screenshot)
        if match:
            screenshoots_count = int(match.group(1)) + 1
            start_frame_time = int(match.group(2))
        else:
            screenshoots_count = 0
            start_frame_time = 0
    else:
        screenshoots_count = 0
        start_frame_time = 0

    for frame_count, frame_time, frame in get_frames(video_path, start_frame_time):

        orig = frame.copy()  # clone the original frame (so we can save it later),
        frame = imutils.resize(frame, width=600)  # resize the frame
        mask = fgbg.apply(frame)  # apply the background subtractor

        # apply a series of erosions and dilations to eliminate noise
#            eroded_mask = cv2.erode(mask, None, iterations=2)
#            mask = cv2.dilate(mask, None, iterations=2)

        # if the width and height are empty, grab the spatial dimensions
        if W is None or H is None:
            (H, W) = mask.shape[:2]

        # compute the percentage of the mask that is "foreground"
        p_diff = (cv2.countNonZero(mask) / float(W * H)) * 100

        # if p_diff less than N% then motion has stopped, thus capture the frame

        if p_diff < MIN_PERCENT and not captured and frame_count > WARMUP:
            captured = True
            filename = f"{course_request_id}_{screenshoots_count:03}_{round(frame_time)}.png"

            path = os.path.join(output_folder_screenshot_path, filename)
            print("saving {}".format(path))
            cv2.imwrite(path, orig)
            screenshoots_count += 1

        # otherwise, either the scene is changing or we're still in warmup
        # mode so let's wait until the scene has settled or we're finished
        # building the background model
        elif captured and p_diff >= MAX_PERCENT:
            captured = False
    print(f'{screenshoots_count} screenshots Captured!')
    print(f'Time taken {time.time()-start_time}s')
    return


def initialize_output_folder(video_path, course_request_id):
    '''Initialize the output folder if it does not exist'''
    output_folder_screenshot_path = f"{OUTPUT_SLIDES_DIR}/{course_request_id}"

    os.makedirs(output_folder_screenshot_path, exist_ok=True)
    print('initialized output folder', output_folder_screenshot_path)
    return output_folder_screenshot_path


def convert_screenshots_to_pdf(output_folder_screenshot_path):
    output_pdf_path = f"{OUTPUT_SLIDES_DIR}/{video_path.rsplit('/')[-1].split('.')[0]}" + '.pdf'
    print('output_folder_screenshot_path', output_folder_screenshot_path)
    print('output_pdf_path', output_pdf_path)
    print('converting images to pdf..')
    with open(output_pdf_path, "wb") as f:
        f.write(img2pdf.convert(
            sorted(glob.glob(f"{output_folder_screenshot_path}/*.png"))))
    print('Pdf Created!')
    print('pdf saved at', output_pdf_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("video_path")
    parser.add_argument(
        "video_path", help="path of video to be converted to pdf slides", type=str)
    parser.add_argument("course_request_id",
                        help="ID of the course request", type=str)
    args = parser.parse_args()
    video_path = args.video_path
    course_request_id = args.course_request_id

    print('video_path', video_path)
    output_folder_screenshot_path = initialize_output_folder(video_path, course_request_id)
    detect_unique_screenshots(
        video_path, output_folder_screenshot_path, course_request_id)
