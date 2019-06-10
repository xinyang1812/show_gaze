import numpy as np
import cv2

     
#2D image points. If you change the image, you need to change vector
image_points = np.zeros((15,2), dtype="double")
 

def angle(x,y):
    Lx = np.sqrt(x.dot(x))
    Ly = np.sqrt(y.dot(y))
    cos_angle = x.dot(y) / (Lx * Ly)
    # print(cos_angle)
    angle_r = np.arccos(cos_angle)
    angle_a = angle_r * 360 / 2 / np.pi
    return angle_a

def indexofMin(arr):
    minindex = 0
    currentindex = 1
    while currentindex < len(arr):
        if arr[currentindex] < arr[minindex]:
            minindex = currentindex
        currentindex += 1
    return minindex


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def show_gaze(img, normals):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((3 * 4, 3), np.float32)
    objp[:, :2] = np.mgrid[0:3, 0:4].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, 25.0, 0.0),
        (0.0, 50.0, 0.0),
        (25.0, 0.0, 0.0),
        (25.0, 25.0, 0.0),
        (25.0, 50.0, 0.0),
        (50.0, 0.0, 0.0),
        (50.0, 25.0, 0.0),
        (50.0, 50.0, 0.0),
        (75.0, 0.0, 0.0),
        (75.0, 25.0, 0.0),
        (75.0, 50.0, 0.0)
    ])

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (3,4),None)
    print('ret', ret)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        print('imgpoints', imgpoints)
        # Draw and display the corners
        # img = cv2.drawChessboardCorners(img, (5,3), corners2,ret)
        # cv2.imshow('img',img)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        print('mtx', mtx)
        # img = cv2.imread('left12.jpg')
        # h,  w = img.shape[:2]
        # mtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        
        image_points = np.asarray(imgpoints, dtype=np.float32)[0,:,0,:]
        # image_points = imgpoints
        print('img points', image_points)
        for p in image_points:
            # print('p',p)
            p1 = ( int(p[0]), int(p[1]))
            cv2.circle(img, p1, 3, (255,0,0), -1)

        (ret, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        print('ret', ret)
        # (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, mtx, dist)
        (gaze_point2D, jacobian) = cv2.projectPoints(model_points, rotation_vector, translation_vector, mtx, dist)
        print(image_points,'===',gaze_point2D)
        # exit(1)
         
         
        # (start_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 0.0)]), rotation_vector, translation_vector, mtx, dist)
        # p1 = ( int(start_point2D[0][0][0]), int(start_point2D[0][0][1]))
        # print('start_point2D', p1)


         
        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose

        (compare1, jacobian) = cv2.projectPoints(np.array([(0.0, 25.0, 0.0)]), rotation_vector, translation_vector, mtx,   dist)
        (compare2, jacobian) = cv2.projectPoints(np.array([(0.0, 25.0, 100.0)]), rotation_vector, translation_vector, mtx, dist)

        com1 = (int(compare1[0][0][0]), int(compare1[0][0][1]))
        com2 = (int(compare2[0][0][0]), int(compare2[0][0][1]))
        cv2.line(img, com1, com2, (0, 255, 0), 2)

        Angle = []
        nn = np.array([0.0, 0.0, 100.0])
        for ii in normals:
            ii = np.array(ii)
            Angle.append(angle(ii, nn))
        print Angle
        inmin = indexofMin(Angle[1:])
        inmin += 1

        normals_big = normals*100
        (normals_point2d, jacobian) = cv2.projectPoints(normals_big, rotation_vector, translation_vector, mtx, dist)
        p0 = ( int(normals_point2d[0][0][0]), int(normals_point2d[0][0][1]))

        # minangle
        print('start point', p0)
        color = (100,0,0)
        q1 = ( int(normals_point2d[inmin][0][0]), int(normals_point2d[inmin][0][1]))
        print('gaze_points2d', q1)
        cv2.circle(img, q1, 3, (0,0,255), -1)
        cv2.line(img, p0, q1, color, 2)

        # idx = 25
        # for p in normals_point2d:
        #     color = (idx,0,0)
        #     idx = idx+25
        #     q1 = ( int(p[0][0]), int(p[0][1]))
        #     print('gaze_points2d', q1)
        #     cv2.circle(img, q1, 3, (0,0,255), -1)
        #     cv2.line(img, p0, q1, color, 2)


        text1 = "Minimum angle: " + str(Angle[inmin])
        aa = map(int, Angle[1:])
        text2 = "all angles: " + str(aa[0:4])
        text3 = str(aa[4:])
        AddText = img.copy()
        cv2.putText(AddText, text1, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(AddText, text2, (1, 80), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(AddText, text3, (1, 100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('img',AddText)
        cv2.waitKey(100)
