import cv2
import numpy as np

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('stabilized_output.avi', fourcc, 20.0, (640, 480))

feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

H = np.eye(3)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)


    H[0, 2] += (p1[0, 0, 0] - p0[0, 0, 0])
    H[1, 2] += (p1[0, 0, 1] - p0[0, 0, 1])


    stabilized_frame = cv2.warpAffine(frame, H[:2, :], (frame.shape[1], frame.shape[0]))


    out.write(stabilized_frame)

    cv2.imshow('Stabilized Frame', stabilized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    old_gray = frame_gray.copy()
    p0 = p1.reshape(-1, 1, 2)

cap.release()
out.release()
cv2.destroyAllWindows()

