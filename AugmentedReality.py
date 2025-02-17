import cv2  # Import OpenCV for image and video processing
import numpy as np  # Import NumPy for numerical operations

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Load the target image (to be detected) and the video to overlay
imgTarget = cv2.imread("ImageTarget.jpg")
myVid = cv2.VideoCapture("video.mp4")

detection = False  # Flag to indicate if the target image is detected in the webcam frame
frameCounter = 0  # Counter to keep track of video frames

# Read the first frame from the video and resize both the video frame and target image
success, ImgVideo = myVid.read()
ImgVideo = cv2.resize(ImgVideo, (500, 400))
imgTarget = cv2.resize(imgTarget, (500, 400))

# Initialize ORB detector to find keypoints and descriptors, limiting to 1000 features
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(imgTarget, None)  # Compute keypoints and descriptors for the target image


def stackImages(imgArray, scale, labels=[]):
    """
    Stack images in a grid for easy visualization.
    Resizes images, converts grayscale images to BGR,
    and optionally adds labels.
    """
    sizeW = imgArray[0][0].shape[1]  # Get width from the first image
    sizeH = imgArray[0][0].shape[0]  # Get height from the first image
    rows = len(imgArray)  # Number of rows in the grid
    cols = len(imgArray[0])  # Number of columns in the grid
    rowsAvailable = isinstance(imgArray[0], list)  # Check if we have a 2D list (grid)

    if rowsAvailable:
        # Process each image in the grid
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (sizeW, sizeH), None, scale, scale)
                # Convert grayscale images to BGR for consistency
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        # Create a blank image for placeholders
        imageBlank = np.zeros((sizeH, sizeW, 3), np.uint8)
        hor = [imageBlank] * rows
        for x in range(0, rows):
            # Horizontally stack images in each row
            hor[x] = np.hstack(imgArray[x])
        # Vertically stack all rows to form the final image
        ver = np.vstack(hor)
    else:
        # Process a single row of images
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (sizeW, sizeH), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        ver = np.hstack(imgArray)

    # Optionally add labels to each image
    if len(labels) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                              (c * eachImgWidth + len(labels[d]) * 13 + 27, eachImgHeight * d + 30),
                              (255, 255, 255), cv2.FILLED)
                cv2.putText(ver, labels[d], (eachImgWidth * c + 10, eachImgHeight * d + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
    return ver


while True:
    # Capture frame from webcam
    success, imgWebcam = cap.read()
    imgAug = imgWebcam.copy()  # Copy frame for augmentation
    # Detect keypoints and compute descriptors for the webcam frame
    kp2, des2 = orb.detectAndCompute(imgWebcam, None)

    if detection == False:
        # Reset the video to the beginning if the target is not detected
        myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frameCounter = 0
    else:
        # Loop the video if it reaches the end
        if frameCounter == myVid.get(cv2.CAP_PROP_FRAME_COUNT):
            myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
        success, imgVideo = myVid.read()
        # Resize the video frame (note: there is a typo; it should use '=' instead of '-')
        imgVideo = cv2.resize(imgVideo, (500, 400))

    # Initialize brute-force matcher and perform kNN matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test to filter good matches
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(len(good))  # Debug: print number of good matches

    # Draw matched keypoints between the target and webcam images
    imgFeatures = cv2.drawMatches(imgTarget, kp1, imgWebcam, kp2, good, None)

    # If enough good matches are found, compute the homography matrix
    if len(good) > 7:
        detection = True
        # Get the matching keypoints' coordinates from the target image
        scrPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        # Get the matching keypoints' coordinates from the webcam frame
        dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # Compute the homography matrix using RANSAC
        matrix, mask = cv2.findHomography(scrPts, dstPts, cv2.RANSAC, 5)
        print(matrix)  # Debug: print the homography matrix

    # Define points of the target image's corners
    pts = np.float32([[0, 0], [0, 500], [400, 500], [400, 0]]).reshape(-1, 1, 2)
    # Transform these points to the perspective of the webcam frame
    dst = cv2.perspectiveTransform(pts, matrix)
    # Draw a polygon around the detected target area in the webcam frame
    img2 = cv2.polylines(imgWebcam, [np.int32(dst)], True, (255, 0, 255), 3)

    # Warp the video frame to fit the detected target area
    imgWarp = cv2.warpPerspective(ImgVideo, matrix, (imgWebcam.shape[1], imgWebcam.shape[0]))

    # Create a mask for the target area
    maskNew = np.zeros((imgWebcam.shape[0], imgWebcam.shape[1]), np.uint8)
    cv2.fillPoly(maskNew, [np.int32(dst)], (255, 255, 255))
    maskInv = cv2.bitwise_not(maskNew)
    # Black-out the target area in the augmented image
    imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInv)
    # Overlay the warped video onto the target area
    imgAug = cv2.bitwise_or(imgWarp, imgAug)

    # Stack various images for debugging and visualization
    imgstacked = stackImages(([imgWebcam, ImgVideo, imgTarget],
                              [imgFeatures, imgWarp, imgAug]), 0.5)

    # Display the stacked images
    cv2.imshow("imgstacked", imgstacked)
    cv2.waitKey(1)
    frameCounter += 1  # Increment the video frame counter
