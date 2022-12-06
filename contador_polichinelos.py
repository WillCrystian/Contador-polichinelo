import cv2
import mediapipe as mp
import math

video = cv2.VideoCapture('polichinelo.mp4')

pose =mp.solutions.pose
Pose = pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)
draw = mp.solutions.drawing_utils
contador = 0
check = False

while True:
    success, img = video.read()
    img = cv2.rotate(img, cv2.ROTATE_180)
    videoRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = Pose.process(videoRGB)
    points = results.pose_landmarks
    draw.draw_landmarks(img, points, pose.POSE_CONNECTIONS)
    h, w, _ = img.shape
    
    if points:
        peDX = int(points.landmark[pose.PoseLandmark.RIGHT_FOOT_INDEX].x*w)
        peDY = int(points.landmark[pose.PoseLandmark.RIGHT_FOOT_INDEX].y*h)
        peEX = int(points.landmark[pose.PoseLandmark.LEFT_FOOT_INDEX].x*w)
        peEY = int(points.landmark[pose.PoseLandmark.LEFT_FOOT_INDEX].y*h)
        
        moDX = int(points.landmark[pose.PoseLandmark.RIGHT_INDEX].x*w)
        moDY = int(points.landmark[pose.PoseLandmark.RIGHT_INDEX].y*h)
        moEX = int(points.landmark[pose.PoseLandmark.LEFT_INDEX].x*w)
        moEY = int(points.landmark[pose.PoseLandmark.LEFT_INDEX].y*h)

        distMO = math.hypot(moDX - moEX, moDY - moEY)
        distPE = math.hypot(peDX - peEX, peDY - peEY)
    
        if distMO < 80 and distPE > 115 and not check:
            contador +=1
            check = True
            print(contador)
            print(f'maos= {distMO}/ pes= {distPE}')
        if distMO > 80 and distPE < 115 and check:
            check = False
            
        texto = f'QTD {contador}'
        cv2.rectangle(img, (80, 540), (300, 620), (255, 0, 0), -1)
        cv2.putText(img, texto, (100, 600), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
        
    cv2.imshow('Resultado', img)
    cv2.waitKey(10)