import imutils
import numpy as np
import math
import cv2

class OrbDetection:
    def __init__(self):
        # initiate ORB detector
        self.orb = cv2.ORB_create(nfeatures = 1000)

        self.view = True



    def trainingImage(self, imgstrain):
        features = self.orb.detectAndCompute(imgstrain, None)
        return features

    # prepare the image by getting the ROI and making the ROI black and weight/gray scaled
    def prepareImages(self, image, boundingBox):
        # be mindful that of what ratio tu use, a big one will hit the image borders, or may hold additional wheels..
        (x, y, w, h) = self.enlarge_bounding(image_shape=image.shape, boundingBox=boundingBox,ratio=1.6)
        newImage = image[y:y + h, x:x + w]
        newImage = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
        return newImage

    # methode to create an orb detection on a larger patch of image,
    # this should increase ORB effectivness
    def enlarge_bounding(self, image_shape, boundingBox,ratio=3):
        # establish what (x2,y2,w2,h2)
        (x1, y1, w1, h1) = boundingBox
        w2 = w1*ratio
        h2 = h1*ratio
        #print("w2: "+str(w2)+" h2: "+str(h2))

        x2 = x1 - (w2-w1)/2
        y2 = y1 - (h2-h1)/2
        #print("x2: " + str(x2) + " y2: " + str(y2))

        # establish the upper limit of image (X,Y). Nobrainer, the lower limits of (X,Y) is (0,0)
        img_height  = image_shape[0]
        img_width   = image_shape[1]
        #print("img_height: " + str(img_height) + " img_width: " + str(img_width))

        # make sure the new bounding box is within the image
        if x2 < 0:
            x2 = 0
        if y2 < 0:
            y2 = 0
        if y2+h2 > img_height:
            h2 = img_height
        if x2+w2 > img_width:
            w2 = img_width

        # math.floor is used to round down a number, this is necessary to make sure ints are obtained
        boundingBox = (math.floor(x2),math.floor(y2),math.floor(w2),math.floor(h2))
        return boundingBox

    # this is the main control block for feature detection, 3 last argument. matches_threshold is used to set how
    # sensitive the orb detection is. This is important to keep quit sensitive/low when checking dictionary for already
    # known bounding boxes. Specifically, because blurry images are hard to match. querry_ROI and train_ROI exist
    # to help debug the code. Look line 89, if querry_ROI is not None and train_ROI is not None: to use them correctly
    def computeQuery(self, queryfeatures, trainfeatures, matches_threshold = 52, querry_ROI = None, train_ROI = None):
        if trainfeatures is None:
            print("[INFO] missing trainfeatures. Please "
                  "supply trainfeatures to the computeQuery function")

        elif queryfeatures is None:
            print("[INFO] missing an image query. Please "
                  "supply an image to query")
        else:
            # print("detecting matching descriptors")
            (kp1, des1) = queryfeatures
            (kp2, des2) = trainfeatures


            # compute homography of our points. If the match is good then there should be a GOOD homography transformation
            # between our query and train images. Furthermore the homography can be used to display the result
            M = self.matchKeypoints(kp1, kp2, des1, des2, ratio=0.75, reprojThresh=5.0)
            matchesH, homography, status = None, None, None
            totalSuccess = 0
            if M is not None:
                (matchesH, H, status) = M
                for success in status:
                    if success == 1:
                        totalSuccess += 1


                # if querry_ROI is not None and train_ROI is not None:
                #     print(totalSuccess)  # a count of all the matches found with ransac
                #     self.drawMatches(querry_ROI, train_ROI, cv2.KeyPoint_convert(kp1), cv2.KeyPoint_convert(kp2),
                #                      matchesH, status)
                    # LEFT IS TRAIN IMAGE AND RIGHT IS QUERY IMAGE

            # if the match is None, then there aren't enough matched
            # keypoints to create a panorama
            if totalSuccess < matches_threshold:
                # print("\n[INFO] Failure to match image. RANSAC has found very few matches that compute a homography ")
                return False,totalSuccess

            else:
                # print("\n[INFO] Success, a match was made. RANSAC found a homography that fits "
                #       + str(totalSuccess) + " feature points.")
                return True,totalSuccess

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                       ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        if featuresA is None or featuresB is None:  # this is a check to avoid a semi likely error
            return None
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))


        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            kpsA=cv2.KeyPoint_convert(kpsA)
            kpsB=cv2.KeyPoint_convert(kpsB)
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,reprojThresh)
            #(H, status) = cv2.findHomography(ptsA, ptsB, 0)
            #(H, status) = cv2.findHomography(ptsA, ptsB, cv2.LMEDS)
                # cv2.RANSAC, reprojThresh  with value 0-10

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

        # otherwise, no homograpy could be computed
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        cv2.imshow("Match using LMEDS", vis)
        cv2.waitKey(0)
        cv2.destroyWindow("Match using LMEDS")

