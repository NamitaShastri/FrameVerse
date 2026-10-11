"""
============================================================
                    VISIONFRAME
============================================================

Real-time Computer Vision project.

Concept:
    The user creates a director's frame using both hands.
    The region inside the frame receives a visual effect.

Gestures:
    - Director frame -> activates VisionFrame
    - Both thumbs touch and separate -> next visual mode
    - Remove hands -> return to original camera view

Technologies:
    - Python
    - OpenCV
    - NumPy
    - MediaPipe Hand Landmarker

============================================================
"""

# ============================================================
# 1. IMPORTS
# ============================================================

import cv2
import numpy as np
import time
import math

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# 2. CONFIGURATION
# ============================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

MAX_HANDS = 2

# Distance threshold for detecting thumb-touch.
# This is expressed as a fraction of the average hand size.
THUMB_TOUCH_RATIO = 0.18

# Number of seconds before another thumb-touch can trigger.
GESTURE_COOLDOWN = 0.8

# How many pixels the frame corners can move inward
# when calculating the director frame.
FRAME_PADDING = 10

# Whether to show MediaPipe landmarks.
SHOW_LANDMARKS = True

# Whether to show debugging information.
SHOW_DEBUG = True


# ============================================================
# 3. EFFECT NAMES
# ============================================================

EFFECT_NAMES = [
    "ORIGINAL",
    "CINEMATIC",
    "NEON",
    "GLITCH",
    "VHS",
    "NIGHT VISION",
    "COMIC",
    "SKETCH",
    "INFRARED",
    "PIXEL ART",
    "DREAM",
]


# ============================================================
# 4. MEDIAPIPE LANDMARK INDICES
# ============================================================

# MediaPipe Hand Landmarker provides 21 landmarks per hand.
#
# Important landmarks:
#
# 0  = Wrist
# 4  = Thumb tip
# 8  = Index finger tip
# 12 = Middle finger tip
# 16 = Ring finger tip
# 20 = Pinky tip

WRIST = 0

THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


# ============================================================
# 5. MODEL SETUP
# ============================================================

MODEL_PATH = "hand_landmarker.task"

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=MAX_HANDS,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

detector = vision.HandLandmarker.create_from_options(options)


# ============================================================
# 6. BASIC GEOMETRY FUNCTIONS
# ============================================================

def distance(p1, p2):
    """
    Calculate Euclidean distance between two 2D points.
    """

    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def midpoint(p1, p2):
    """
    Return midpoint between two points.
    """

    return (
        int((p1[0] + p2[0]) / 2),
        int((p1[1] + p2[1]) / 2)
    )


def clamp(value, minimum, maximum):
    """
    Keep a value inside a specified range.
    """

    return max(minimum, min(value, maximum))


# ============================================================
# 7. LANDMARK CONVERSION
# ============================================================

def normalized_to_pixel(landmark, width, height):
    """
    MediaPipe gives normalized coordinates.

    x and y are approximately between 0 and 1.

    Convert them into actual image coordinates.
    """

    x = int(landmark.x * width)
    y = int(landmark.y * height)

    x = clamp(x, 0, width - 1)
    y = clamp(y, 0, height - 1)

    return (x, y)


def convert_hand_landmarks(hand_landmarks, width, height):
    """
    Convert all 21 normalized landmarks
    into pixel coordinates.
    """

    points = []

    for landmark in hand_landmarks:

        point = normalized_to_pixel(
            landmark,
            width,
            height
        )

        points.append(point)

    return points


# ============================================================
# 8. HAND SIZE
# ============================================================

def calculate_hand_size(points):
    """
    Estimate hand size.

    We use wrist-to-middle-finger-tip distance
    as a simple scale reference.
    """

    wrist = points[WRIST]
    middle_tip = points[MIDDLE_TIP]

    return distance(wrist, middle_tip)


# ============================================================
# 9. THUMB TOUCH DETECTION
# ============================================================

def thumbs_are_touching(points1, points2):
    """
    Determine whether the two thumbs are close enough
    to be considered touching.

    We normalize the distance according to hand size,
    so the gesture can work at different distances
    from the camera.
    """

    thumb1 = points1[THUMB_TIP]
    thumb2 = points2[THUMB_TIP]

    thumb_distance = distance(
        thumb1,
        thumb2
    )

    size1 = calculate_hand_size(points1)
    size2 = calculate_hand_size(points2)

    average_hand_size = (
        size1 + size2
    ) / 2

    if average_hand_size <= 1:
        return False

    normalized_distance = (
        thumb_distance /
        average_hand_size
    )

    return normalized_distance < THUMB_TOUCH_RATIO


# ============================================================
# 10. FINGER EXTENSION
# ============================================================

def finger_is_extended(points, tip, pip):
    """
    Determine whether a finger is reasonably extended.

    We compare the distance of the fingertip from
    the wrist with the distance of the PIP joint
    from the wrist.

    If the fingertip is farther away,
    the finger is likely extended.
    """

    wrist = points[WRIST]

    tip_distance = distance(
        wrist,
        points[tip]
    )

    pip_distance = distance(
        wrist,
        points[pip]
    )

    return tip_distance > pip_distance * 1.15


def thumb_is_extended(points):
    """
    Basic thumb extension check.
    """

    wrist = points[WRIST]

    tip_distance = distance(
        wrist,
        points[THUMB_TIP]
    )

    ip_distance = distance(
        wrist,
        points[THUMB_IP]
    )

    return tip_distance > ip_distance * 1.1


# ============================================================
# 11. DIRECTOR FRAME GESTURE
# ============================================================

def is_director_frame_hand(points):
    """
    Check whether one hand resembles one side
    of the director-frame gesture.

    For a first prototype, we require:
        - thumb extended
        - index extended

    Other fingers should generally be folded.

    This does NOT mean the gesture is perfect.
    It simply gives us a robust prototype.
    """

    thumb_extended = thumb_is_extended(points)

    index_extended = finger_is_extended(
        points,
        INDEX_TIP,
        INDEX_PIP
    )

    middle_extended = finger_is_extended(
        points,
        MIDDLE_TIP,
        MIDDLE_PIP
    )

    ring_extended = finger_is_extended(
        points,
        RING_TIP,
        RING_PIP
    )

    pinky_extended = finger_is_extended(
        points,
        PINKY_TIP,
        PINKY_PIP
    )

    # We want thumb + index to be dominant.
    #
    # Some tolerance is intentionally allowed because
    # hand orientation can vary.

    if not thumb_extended:
        return False

    if not index_extended:
        return False

    folded_count = 0

    if not middle_extended:
        folded_count += 1

    if not ring_extended:
        folded_count += 1

    if not pinky_extended:
        folded_count += 1

    return folded_count >= 2


# ============================================================
# 12. DIRECTOR FRAME CORNER DETECTION
# ============================================================

def find_frame_corners(hand1, hand2, width, height):
    """
    Construct the rectangular region between the
    two hand gestures.

    We use:
        - index fingertip of hand 1
        - thumb tip of hand 1
        - index fingertip of hand 2
        - thumb tip of hand 2

    Then calculate the bounding rectangle.

    This creates the visual 'director's frame'.
    """

    relevant_points = [

        hand1[THUMB_TIP],
        hand1[INDEX_TIP],

        hand2[THUMB_TIP],
        hand2[INDEX_TIP],
    ]

    xs = [p[0] for p in relevant_points]
    ys = [p[1] for p in relevant_points]

    x_min = min(xs)
    x_max = max(xs)

    y_min = min(ys)
    y_max = max(ys)

    # Padding makes the region slightly larger.
    x_min -= FRAME_PADDING
    y_min -= FRAME_PADDING

    x_max += FRAME_PADDING
    y_max += FRAME_PADDING

    x_min = clamp(x_min, 0, width - 1)
    x_max = clamp(x_max, 0, width - 1)

    y_min = clamp(y_min, 0, height - 1)
    y_max = clamp(y_max, 0, height - 1)

    # Make sure region is large enough.
    if x_max - x_min < 50:
        return None

    if y_max - y_min < 50:
        return None

    return (
        x_min,
        y_min,
        x_max,
        y_max
    )


# ============================================================
# VISUAL EFFECTS
# ============================================================

def effect_cinematic(roi):
    """
    Cinematic movie-style effect.

    Increases contrast, slightly boosts color,
    and adds a subtle dark vignette around the edges.
    """

    # Increase contrast and brightness
    cinematic = cv2.convertScaleAbs(
        roi,
        alpha=1.25,
        beta=-10
    )

    # Slightly increase saturation
    hsv = cv2.cvtColor(
        cinematic,
        cv2.COLOR_BGR2HSV
    )

    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1] * 1.15,
        0,
        255
    ).astype(np.uint8)

    cinematic = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )

    # Create subtle vignette
    h, w = cinematic.shape[:2]

    kernel_x = cv2.getGaussianKernel(
        w,
        w / 2
    )

    kernel_y = cv2.getGaussianKernel(
        h,
        h / 2
    )

    mask = kernel_y @ kernel_x.T

    mask = mask / mask.max()

    mask = 0.65 + 0.35 * mask

    cinematic = (
        cinematic.astype(np.float32)
        * mask[:, :, np.newaxis]
    )

    cinematic = np.clip(
        cinematic,
        0,
        255
    ).astype(np.uint8)

    return cinematic


def effect_neon(roi):
    """
    Neon glowing-edge effect.
    """

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    # Make edges thicker
    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    edges = cv2.dilate(
        edges,
        kernel,
        iterations=1
    )

    # Create colored glow
    glow = np.zeros_like(roi)

    glow[:, :, 0] = edges
    glow[:, :, 1] = edges // 3
    glow[:, :, 2] = edges

    # Blur glow
    glow = cv2.GaussianBlur(
        glow,
        (0, 0),
        5
    )

    result = cv2.addWeighted(
        roi,
        0.35,
        glow,
        1.5,
        0
    )

    result = cv2.addWeighted(
        result,
        0.8,
        glow,
        0.7,
        0
    )

    return result


def effect_glitch(roi):
    """
    Digital RGB glitch effect.

    Separates color channels and shifts
    them horizontally.
    """

    h, w = roi.shape[:2]

    b, g, r = cv2.split(roi)

    shift = max(
        2,
        w // 40
    )

    # Shift red channel right
    red_shifted = np.roll(
        r,
        shift,
        axis=1
    )

    # Shift blue channel left
    blue_shifted = np.roll(
        b,
        -shift,
        axis=1
    )

    glitch = cv2.merge(
        [
            blue_shifted,
            g,
            red_shifted
        ]
    )

    # Add scanlines
    for y in range(
        0,
        h,
        4
    ):
        glitch[y:y + 1] = (
            glitch[y:y + 1] * 0.65
        ).astype(np.uint8)

    # Add a few horizontal displacement bands
    band_height = max(
        2,
        h // 20
    )

    for i in range(3):

        y = (
            (i + 1) * h // 4
        )

        y_end = min(
            y + band_height,
            h
        )

        band = glitch[
            y:y_end
        ].copy()

        glitch[
            y:y_end
        ] = np.roll(
            band,
            shift * (i + 1),
            axis=1
        )

    return glitch


def effect_vhs(roi):
    """
    Retro VHS effect.

    Adds:
        - noise
        - scanlines
        - color distortion
        - slight blur
    """

    h, w = roi.shape[:2]

    # Slight blur
    vhs = cv2.GaussianBlur(
        roi,
        (3, 3),
        0
    )

    # Create random noise
    noise = np.random.normal(
        0,
        18,
        (h, w, 1)
    )

    noise = np.repeat(
        noise,
        3,
        axis=2
    )

    vhs = np.clip(
        vhs.astype(np.float32)
        + noise,
        0,
        255
    ).astype(np.uint8)

    # Scanlines
    for y in range(
        0,
        h,
        4
    ):
        vhs[y] = (
            vhs[y] * 0.65
        ).astype(np.uint8)

    # Slight horizontal distortion
    shift = max(
        1,
        w // 80
    )

    vhs = np.roll(
        vhs,
        shift,
        axis=1
    )

    return vhs


def effect_night_vision(roi):
    """
    Green night-vision camera effect.
    """

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    # Improve contrast
    gray = cv2.equalizeHist(
        gray
    )

    # Green image
    night = np.zeros_like(roi)

    night[:, :, 0] = (
        gray * 0.15
    ).astype(np.uint8)

    night[:, :, 1] = gray

    night[:, :, 2] = (
        gray * 0.15
    ).astype(np.uint8)

    # Add subtle glow
    glow = cv2.GaussianBlur(
        night,
        (0, 0),
        3
    )

    night = cv2.addWeighted(
        night,
        0.8,
        glow,
        0.2,
        0
    )

    return night


def effect_comic(roi):
    """
    Comic-book effect.

    Uses smoothing + strong edges.
    """

    smooth = cv2.bilateralFilter(
        roi,
        9,
        150,
        150
    )

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.medianBlur(
        gray,
        7
    )

    edges = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9,
        9
    )

    edges = cv2.cvtColor(
        edges,
        cv2.COLOR_GRAY2BGR
    )

    comic = cv2.bitwise_and(
        smooth,
        edges
    )

    # Slightly boost saturation
    hsv = cv2.cvtColor(
        comic,
        cv2.COLOR_BGR2HSV
    )

    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1] * 1.3,
        0,
        255
    ).astype(np.uint8)

    comic = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )

    return comic


def effect_sketch(roi):
    """
    Pencil sketch effect.
    """

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    inverted = cv2.bitwise_not(
        gray
    )

    blurred = cv2.GaussianBlur(
        inverted,
        (21, 21),
        0
    )

    sketch = cv2.divide(
        gray,
        255 - blurred,
        scale=256
    )

    return cv2.cvtColor(
        sketch,
        cv2.COLOR_GRAY2BGR
    )


def effect_infrared(roi):
    """
    False-color infrared / thermal-style effect.

    This is a visual simulation and NOT actual
    infrared camera data.
    """

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.equalizeHist(
        gray
    )

    infrared = cv2.applyColorMap(
        gray,
        cv2.COLORMAP_JET
    )

    return infrared



def effect_dream(roi):
    """
    Dreamy / soft-focus effect.

    Combines blur, brightness and a soft glow.
    """

    blurred = cv2.GaussianBlur(
        roi,
        (0, 0),
        7
    )

    dream = cv2.addWeighted(
        roi,
        0.65,
        blurred,
        0.35,
        10
    )

    # Slightly boost saturation
    hsv = cv2.cvtColor(
        dream,
        cv2.COLOR_BGR2HSV
    )

    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1] * 1.15,
        0,
        255
    ).astype(np.uint8)

    dream = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )

    return dream


# ============================================================
# APPLY SELECTED EFFECT
# ============================================================

def apply_effect(roi, effect_index):

    if effect_index == 0:
        return roi.copy()

    elif effect_index == 1:
        return effect_cinematic(roi)

    elif effect_index == 2:
        return effect_neon(roi)

    elif effect_index == 3:
        return effect_glitch(roi)

    elif effect_index == 4:
        return effect_vhs(roi)

    elif effect_index == 5:
        return effect_night_vision(roi)

    elif effect_index == 6:
        return effect_comic(roi)

    elif effect_index == 7:
        return effect_sketch(roi)

    elif effect_index == 8:
        return effect_infrared(roi)

    elif effect_index == 9:
        return effect_dream(roi)

    return roi.copy()

# ============================================================
# 18. APPLY SELECTED EFFECT
# ============================================================

def apply_effect(roi, effect_index):

    if effect_index == 0:
        return roi.copy()

    elif effect_index == 1:
        return effect_cinematic(roi)

    elif effect_index == 2:
        return effect_neon(roi)

    elif effect_index == 3:
        return effect_glitch(roi)

    elif effect_index == 4:
        return effect_vhs(roi)

    elif effect_index == 5:
        return effect_night_vision(roi)

    elif effect_index == 6:
        return effect_comic(roi)

    elif effect_index == 7:
        return effect_sketch(roi)

    elif effect_index == 8:
        return effect_infrared(roi)

    elif effect_index == 9:
        return effect_dream(roi)

    return roi.copy()
    """
    Select the correct visual effect.
    """

    if effect_index == 0:
        return roi.copy()

    elif effect_index == 1:
        return effect_black_white(roi)

    elif effect_index == 2:
        return effect_neon(roi)

    elif effect_index == 3:
        return effect_sketch(roi)

    elif effect_index == 4:
        return effect_thermal(roi)

    elif effect_index == 5:
        return effect_cartoon(roi)

    return roi.copy()


# ============================================================
# 19. DRAW DIRECTOR FRAME
# ============================================================

def draw_director_frame(
    image,
    corners,
    active=True
):
    """
    Draw a visual rectangle around the selected region.
    """

    if corners is None:
        return

    x1, y1, x2, y2 = corners

    thickness = 3

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        thickness
    )

    # Draw corner markers to make it look
    # more like a director's frame.

    corner_length = 30

    # Top-left
    cv2.line(
        image,
        (x1, y1),
        (x1 + corner_length, y1),
        (0, 255, 255),
        4
    )

    cv2.line(
        image,
        (x1, y1),
        (x1, y1 + corner_length),
        (0, 255, 255),
        4
    )

    # Top-right
    cv2.line(
        image,
        (x2, y1),
        (x2 - corner_length, y1),
        (0, 255, 255),
        4
    )

    cv2.line(
        image,
        (x2, y1),
        (x2, y1 + corner_length),
        (0, 255, 255),
        4
    )

    # Bottom-left
    cv2.line(
        image,
        (x1, y2),
        (x1 + corner_length, y2),
        (0, 255, 255),
        4
    )

    cv2.line(
        image,
        (x1, y2),
        (x1, y2 - corner_length),
        (0, 255, 255),
        4
    )

    # Bottom-right
    cv2.line(
        image,
        (x2, y2),
        (x2 - corner_length, y2),
        (0, 255, 255),
        4
    )

    cv2.line(
        image,
        (x2, y2),
        (x2, y2 - corner_length),
        (0, 255, 255),
        4
    )


# ============================================================
# 20. DRAW HAND LANDMARKS
# ============================================================

HAND_CONNECTIONS = [

    (WRIST, THUMB_CMC),
    (THUMB_CMC, THUMB_MCP),
    (THUMB_MCP, THUMB_IP),
    (THUMB_IP, THUMB_TIP),

    (WRIST, INDEX_MCP),
    (INDEX_MCP, INDEX_PIP),
    (INDEX_PIP, INDEX_DIP),
    (INDEX_DIP, INDEX_TIP),

    (WRIST, MIDDLE_MCP),
    (MIDDLE_MCP, MIDDLE_PIP),
    (MIDDLE_PIP, MIDDLE_DIP),
    (MIDDLE_DIP, MIDDLE_TIP),

    (WRIST, RING_MCP),
    (RING_MCP, RING_PIP),
    (RING_PIP, RING_DIP),
    (RING_DIP, RING_TIP),

    (WRIST, PINKY_MCP),
    (PINKY_MCP, PINKY_PIP),
    (PINKY_PIP, PINKY_DIP),
    (PINKY_DIP, PINKY_TIP),

    (INDEX_MCP, MIDDLE_MCP),
    (MIDDLE_MCP, RING_MCP),
    (RING_MCP, PINKY_MCP),
]


def draw_hand_landmarks(
    image,
    points
):
    """
    Draw hand skeleton manually using
    the 21 converted landmark coordinates.
    """

    # Draw connections.
    for start, end in HAND_CONNECTIONS:

        cv2.line(
            image,
            points[start],
            points[end],
            (0, 255, 0),
            2
        )

    # Draw points.
    for index, point in enumerate(points):

        radius = 5

        if index in [
            THUMB_TIP,
            INDEX_TIP,
            MIDDLE_TIP,
            RING_TIP,
            PINKY_TIP
        ]:
            radius = 7

        cv2.circle(
            image,
            point,
            radius,
            (0, 0, 255),
            -1
        )


# ============================================================
# 21. ROI PROCESSING
# ============================================================

def process_roi(
    image,
    corners,
    effect_index
):
    """
    Apply the selected effect ONLY inside the rectangle.

    The rest of the image remains untouched.
    """

    if corners is None:
        return image

    x1, y1, x2, y2 = corners

    # Extract region of interest.
    roi = image[
        y1:y2,
        x1:x2
    ]

    if roi.size == 0:
        return image

    # Apply effect.
    processed_roi = apply_effect(
        roi,
        effect_index
    )

    # Make absolutely sure dimensions match.
    processed_roi = cv2.resize(
        processed_roi,
        (
            x2 - x1,
            y2 - y1
        )
    )

    # Put processed region back into image.
    image[
        y1:y2,
        x1:x2
    ] = processed_roi

    return image


# ============================================================
# 22. TEXT UTILITIES
# ============================================================

def draw_text(
    image,
    text,
    position,
    scale=0.7,
    thickness=2
):
    """
    Draw readable text with a dark background.
    """

    x, y = position

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Calculate text size.
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness
    )

    # Background rectangle.
    cv2.rectangle(
        image,
        (
            x - 8,
            y - text_height - baseline - 8
        ),
        (
            x + text_width + 8,
            y + 8
        ),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        image,
        text,
        (x, y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# 23. CAMERA INITIALIZATION
# ============================================================

camera = cv2.VideoCapture(
    CAMERA_INDEX
)

if not camera.isOpened():

    print(
        "ERROR: Could not open webcam."
    )

    detector.close()

    raise SystemExit


# Set preferred resolution.
camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    FRAME_WIDTH
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)


# ============================================================
# 24. APPLICATION STATE
# ============================================================

effect_index = 1

previous_thumb_touch = False

last_mode_change_time = 0

application_running = True

frame_timestamp_ms = 0


# ============================================================
# 25. FPS VARIABLES
# ============================================================

fps = 0

fps_counter = 0

fps_start_time = time.time()


# ============================================================
# 26. MAIN REAL-TIME LOOP
# ============================================================

while application_running:

    # --------------------------------------------------------
    # Capture one frame from webcam.
    # --------------------------------------------------------

    success, frame = camera.read()

    if not success:

        print(
            "WARNING: Could not read frame."
        )

        break


    # --------------------------------------------------------
    # Mirror the webcam feed.
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------------------------------
    # Get frame dimensions.
    # --------------------------------------------------------

    height, width = frame.shape[:2]


    # --------------------------------------------------------
    # Keep a clean copy.
    #
    # This allows us to preserve the original image
    # outside the director frame.
    # --------------------------------------------------------

    original_frame = frame.copy()


    # --------------------------------------------------------
    # Convert BGR -> RGB.
    #
    # OpenCV normally works with BGR.
    # MediaPipe expects an RGB image.
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Create MediaPipe Image.
    # --------------------------------------------------------

    mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb_frame
)


    # --------------------------------------------------------
    # Timestamp must increase for VIDEO mode.
    # --------------------------------------------------------

    frame_timestamp_ms += 33


    # --------------------------------------------------------
    # Run hand landmark detection.
    # --------------------------------------------------------

    result = detector.detect_for_video(
        mp_image,
        frame_timestamp_ms
    )


    # --------------------------------------------------------
    # Store converted hand points.
    # --------------------------------------------------------

    detected_hands = []


    # --------------------------------------------------------
    # Convert MediaPipe landmarks into pixel coordinates.
    # --------------------------------------------------------

    if result.hand_landmarks:

        for hand_landmarks in result.hand_landmarks:

            points = convert_hand_landmarks(
                hand_landmarks,
                width,
                height
            )

            detected_hands.append(
                points
            )


    # ========================================================
    # 27. DRAW LANDMARKS
    # ========================================================

    if SHOW_LANDMARKS:

        for points in detected_hands:

            draw_hand_landmarks(
                frame,
                points
            )


    # ========================================================
    # 28. DIRECTOR FRAME DETECTION
    # ========================================================

    director_frame_detected = False

    frame_corners = None

    if len(detected_hands) == 2:

        hand1 = detected_hands[0]
        hand2 = detected_hands[1]

        hand1_valid = is_director_frame_hand(
            hand1
        )

        hand2_valid = is_director_frame_hand(
            hand2
        )

        if hand1_valid and hand2_valid:

            frame_corners = find_frame_corners(
                hand1,
                hand2,
                width,
                height
            )

            if frame_corners is not None:

                director_frame_detected = True


    # ========================================================
    # 29. THUMB TOUCH DETECTION
    # ========================================================

    thumb_touch = False

    if director_frame_detected:

        thumb_touch = thumbs_are_touching(
            detected_hands[0],
            detected_hands[1]
        )


    # ========================================================
    # 30. THUMB TOUCH STATE CHANGE
    # ========================================================

    current_time = time.time()

    touch_started = (
        thumb_touch
        and not previous_thumb_touch
    )

    cooldown_finished = (
        current_time -
        last_mode_change_time
        > GESTURE_COOLDOWN
    )


    if touch_started and cooldown_finished:

        # Move to next effect.
        effect_index += 1

        # Loop back to first effect.
        if effect_index >= len(EFFECT_NAMES):

            effect_index = 0

        last_mode_change_time = current_time


    previous_thumb_touch = thumb_touch


    # ========================================================
    # 31. APPLY EFFECT INSIDE FRAME
    # ========================================================

    if director_frame_detected:

        frame = process_roi(
            frame,
            frame_corners,
            effect_index
        )

        # Draw frame on top of processed region.
        draw_director_frame(
            frame,
            frame_corners
        )


    # ========================================================
    # 32. USER INTERFACE
    # ========================================================

    if director_frame_detected:

        draw_text(
            frame,
            "DIRECTOR FRAME ACTIVE",
            (20, 40),
            0.75,
            2
        )

        draw_text(
            frame,
            "MODE: " + EFFECT_NAMES[effect_index],
            (20, 80),
            0.65,
            2
        )

        if thumb_touch:

            draw_text(
                frame,
                "THUMBS TOUCHING",
                (20, 120),
                0.65,
                2
            )

        else:

            draw_text(
                frame,
                "TOUCH THUMBS TO CHANGE MODE",
                (20, 120),
                0.55,
                2
            )

    else:

        draw_text(
            frame,
            "MAKE A DIRECTOR FRAME",
            (20, 40),
            0.75,
            2
        )

        draw_text(
            frame,
            "Both hands: thumb + index finger",
            (20, 80),
            0.55,
            2
        )


    # ========================================================
    # 33. FPS CALCULATION
    # ========================================================

    fps_counter += 1

    elapsed = (
        time.time()
        - fps_start_time
    )

    if elapsed >= 1.0:

        fps = fps_counter / elapsed

        fps_counter = 0

        fps_start_time = time.time()


    # ========================================================
    # 34. DEBUG INFORMATION
    # ========================================================

    if SHOW_DEBUG:

        draw_text(
            frame,
            f"FPS: {fps:.1f}",
            (width - 150, 40),
            0.6,
            2
        )

        draw_text(
            frame,
            f"Hands: {len(detected_hands)}",
            (width - 180, 80),
            0.6,
            2
        )


    # ========================================================
    # 35. DISPLAY
    # ========================================================

    cv2.imshow(
        "VisionFrame",
        frame
    )


    # ========================================================
    # 36. KEYBOARD INPUT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    # ESC -> exit
    if key == 27:

        application_running = False

    # Q -> exit
    elif key == ord("q"):

        application_running = False

    # Number keys can manually change effects.
    elif key == ord("1"):

        effect_index = 0

    elif key == ord("2"):

        effect_index = 1

    elif key == ord("3"):

        effect_index = 2

    elif key == ord("4"):

        effect_index = 3

    elif key == ord("5"):

        effect_index = 4

    elif key == ord("6"):

        effect_index = 5


# ============================================================
# 37. CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()

detector.close()


print(
    "VisionFrame closed successfully."
)