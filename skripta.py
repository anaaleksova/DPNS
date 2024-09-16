import numpy as np
import cv2 
import argparse

def resize_image(image, target_width=500):
    (h, w) = image.shape[:2]
    scale = target_width / float(w)
    new_dim = (target_width, int(h * scale))
    resized = cv2.resize(image, new_dim)
    return resized

def process_images(image_paths, target_width=500):
    images = []
    for image in image_paths:
        img = cv2.imread(image)
        if img is not None:
            resized_img = resize_image(img, target_width)
            images.append(resized_img)
        else:
            print(f"Could not load image {image}")
    return images

def detect_match_draw(images, top_n=80):
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    for i in range(len(images) - 1):
        gray1 = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(images[i + 1], cv2.COLOR_BGR2GRAY)

        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        top_matches = sorted(good_matches, key=lambda x: x.distance)[:top_n]

        img_matches = cv2.drawMatches(
            images[i], keypoints1, images[i + 1], keypoints2,
            top_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imshow(f'Matches between Image {i + 1} and {i + 2}', img_matches)
        cv2.waitKey(0)

def post_process_stitched_image(stitched_img):
    stitched_img = cv2.copyMakeBorder(stitched_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
    thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    contours = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    areaOI = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(areaOI)

    mask = cv2.erode(cv2.rectangle(np.zeros_like(thresh_img), (x, y), (x + w, y + h), 255, -1), None)
    sub = cv2.subtract(mask, thresh_img)

    while cv2.countNonZero(sub) > 0:
        mask = cv2.erode(mask, None)
        sub = cv2.subtract(mask, thresh_img)

    areaOI=max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(areaOI)
    return stitched_img[y:y + h, x:x + w]

def main():
    parser = argparse.ArgumentParser(description="Stitch multiple images to create a panorama.")
    parser.add_argument('images', nargs='+', help='Paths to input images')
    args = parser.parse_args()

    target_width = 500 
    images = process_images(args.images, target_width)
    
    if len(images) < 2:
        print("Need at least two images to stitch")
        return

    detect_match_draw(images, top_n=80)

    imageStitcher = cv2.Stitcher_create()
    error, stitched_img = imageStitcher.stitch(images)

    if not error:
        stitched_img = post_process_stitched_image(stitched_img)

        cv2.imshow("Panorama Image Processed", stitched_img)
        cv2.waitKey(0)
    else:
        print("Images could not be stitched")

if __name__ == "__main__":
    main()
