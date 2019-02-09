import cv2
import glob
import sys

output_path = './movie/'
out_face_path = './images/'
xml_path = "./lbpcascade_animeface.xml"


def movie_to_image(imagefile):

    num_cut = 60
    capture = cv2.VideoCapture(imagefile)
    img_count = 0
    frame_count = 0

    while(capture.isOpened()):

        ret, frame = capture.read()
        if ret is False:
            break

        if frame_count % num_cut == 0:
            img_file_name = output_path + str(img_count) + ".jpg"
            cv2.imwrite(img_file_name, frame)
            img_count += 1

        frame_count += 1
    capture.releas()


def face_detect(img_list):

    classifier = cv2.CascadeClassifier(xml_path)
    img_count = 1

    for img_path in img_list:

        org_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        face_points = classifier.detectMultiScale(
                gray_img, scaleFactor=1.2, minNeighbors=2, minSize=(1, 1))

        for points in face_points:

            x, y, widht, height = points

            dst_img = org_img[y:y+height, x:x+widht]

            face_img = cv2.resize(dst_img, (64, 64))
            new_img_name = out_face_path + \
                str(img_count) + '_face.jpg'
            cv2.imwrite(new_img_name, face_img)
            img_count += 1

        img_count += 1

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage : specify file name")
        sys.exit()
    print(sys.argv)

    movie_to_image(sys.argv[1])

    images = glob.glob(output_path + '*.jpg')
    face_detect(images)
