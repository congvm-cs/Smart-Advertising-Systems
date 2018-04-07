import argparse

def main(args):

    assert args.image_dir is not None

    ## Import libraries
    # import imutils
    import dlib
    import numpy as np 
    from imutils.face_utils import rect_to_bb, FaceAligner
    from keras.models import load_model
    import cv2
    # import align_dlib

    ##====================================================================================================##    
    print('Load model...')
    model_path = '/media/vmc/12D37C49724FE954/Smart-Advertising-Systems/Models/weights-improvement-23-0.23-0.92.hdf5'
    # model_path = 'E:/Smart-Advertising-Systems-master/Models/weights-improvement-23-0.23-0.92.hdf5'
    align_predictor_path = '/media/vmc/12D37C49724FE954/Smart-Advertising-Systems/shape_predictor_68_face_landmarks.dat'

    detector = dlib.get_frontal_face_detector()
    model = load_model(model_path)
    predictor = dlib.shape_predictor(align_predictor_path)
    fa = FaceAligner(predictor, desiredFaceWidth=128)

    # Load image
    print('Load image ...')
    image = cv2.imread(args.image_dir)
    image_clone = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    # image = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    for (i, rect) in enumerate(rects):
        (x, y, h, w) = rect_to_bb(rect)             # Positions of rectangle contains face

        faceAligned = fa.align(image, gray, rect)   # Aligned face image 
        faceAligned = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)

        face_rect_resized = cv2.resize(faceAligned, (128, 128))
        
        cv2.imshow('after aligned #{}'.format(str(i)), face_rect_resized)

        face_rect_normalized = face_rect_resized * 1./255
        face_rect_reshape = np.reshape(face_rect_normalized, newshape=(1, 128, 128, 1))
        y_pred = model.predict(face_rect_reshape)

        if y_pred < 0.5:
            gender = 'Nu'
        else: 
            gender = 'Nam'

        cv2.putText(image_clone, "#{} #{} #{:2f}".format(i+1, gender, float(y_pred)), (x-10, y-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 2)       
        cv2.rectangle(image_clone, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('detect', image_clone)
    cv2.waitKey(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', help='Image directory', default=None, type=str)
    args = parser.parse_args()
    main(args)
