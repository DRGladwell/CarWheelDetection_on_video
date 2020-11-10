from videoBuffer import catchup_buffer
from cnn_thread import cnnThread
from quick_check_BB import quick_check_BB
import concurrent.futures
import cv2
import time



class drawBoundingBox:

    def __init__(self):

        # these colors are actually BGR
        self.line_color_wheel = (0, 255, 0)  # green
        self.line_color_notwheel = (0, 0, 255)  # red
        self.line_type = cv2.LINE_4

        # Object Instances

        # code to inexpensively query a bounding box
        self.quick_check = quick_check_BB()

        # video buffer to allow CNN_ML code which runs on a separate thread to catch up to main thread.
        self.catchup_buffer = catchup_buffer()

        # module that runs all the CNN, Machine Learning and catch up code. This will exist on a thread
        self.cnnThread = cnnThread(self.catchup_buffer)

        # threading CNN adn ML
        self.cnn_ml_thread = None


    def draw(self, originalFrame, rectanglesOriginal, displayFrame, rectanglesDisplay):
        # timer to limit how long a frame can be processed
        start = time.perf_counter()

        # update the scores of currently known car wheel objects to delete those with a bad score.
        # this check is to be performed every frame
        self.quick_check.scoreBB()

        # variable used to collect undefined bounding boxes. This will be used in the video buffer
        # which in turn in used in the CNN_ML thread to create new bounding box object. It makes sens to avoid confusing
        # with already known bounding boxes.
        undefined_bounding_boxes = []

        # Iterate through all the bounding boxes. The zip is used to iterate through both sets of BBs at the same time
        # It's worth nothing that the two sets of bounding boxes (BBs) are the same just at different img scales.
        for rectangleOri, rectangleDis in zip(rectanglesOriginal, rectanglesDisplay):
            # end time to stop the loop. This means some bounding boxes may not be checked.
            finish = time.perf_counter()
            if finish-start >= 0.1:
                #print(f' Frame took {round(finish - start, 2)}second(s)')
                break

            # set up the current bounding box read to be drawn onto the display image
            (x, y, w, h) = rectangleDis
            top_left = (x, y)
            bottom_right = (x + w, y + h)

            # check if this is a bounding box already known. This avoids wasting CPU cycles on expensive CNN_ML thread
            result = self.quick_check.run(originalFrame, rectangleOri)

            # the current frame ID (an integer value) is acquired here to be fed into the CNN_ML thread if needed
            current_frame_ID = self.catchup_buffer.frameID

            # if the bounding box is unknown a CNN_ML "might" be launched. It depends if previous thread has ended.
            if result == None:
                # as the bounding box is unknown it gets added to this list. Remember that the
                # launch_cnn_and_ml_thread will not work instantly (appox 0.7 on my machine). So this BB is undefined.
                undefined_bounding_boxes.append(rectangleOri)

                # launch thread to add car wheel to search
                self.launch_cnn_and_ml_thread(originalFrame=originalFrame, boundingBox=rectangleOri
                                              , current_frame_ID=current_frame_ID)
                # draw bounding box
                cv2.rectangle(displayFrame, top_left, bottom_right, self.line_color_notwheel, lineType=self.line_type)
                # draw text
                cv2.putText(displayFrame, "failure to find wheel", (int(round(x + w / 2)), int(round(y + h / 2))),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.line_color_notwheel, 2)
            else:
                # unpack result variable
                (name, probability) = result
                # draw bounding box
                cv2.rectangle(displayFrame, top_left, bottom_right, self.line_color_wheel, lineType=self.line_type)
                # draw text
                cv2.putText(displayFrame, probability, (int(round(x + w / 2)), int(round(y + h / 2))),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.line_color_wheel, 2)
                cv2.putText(displayFrame, name, (int(round(x + w / 2)), int(round(y + h / 2))+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.line_color_wheel, 2)

        self.quick_check.check_if_carwheel_already_exist(originalFrame, undefined_bounding_boxes)
        # add this frame and found bounding boxes to the video buffer.
        # ideally only add undefined bounding box list.
        self.catchup_buffer.update(frame=originalFrame, BBleftovers=undefined_bounding_boxes)
        return displayFrame

    def launch_cnn_and_ml_thread(self, originalFrame, boundingBox, current_frame_ID):
        # retrieve thread variable that "potentially" contains a thread
        thread = self.cnn_ml_thread

        # if no running thread object exists, create one.
        if thread is None:
            # create a pool object to handle multi threading
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=6)
            # "thread" is a future object. It has a couple of methods used to control the thread.
            thread = executor.submit(self.cnnThread.single_BB_checker, originalFrame, boundingBox,
                                             current_frame_ID)
            # very important to stop the code from blocking in this methode (blocking = wait for thread to complete)
            executor.shutdown(wait=False)

            # call self.merge_cnn_thread_with_main() when thread completes, the thread is used as a required argument
            # Note: in merge_cnn_thread_with_main the self.cnn_ml_thread is made into a None.
            # this means a new thread can be launched.
            thread.add_done_callback(self.merge_cnn_thread_with_main)

            # set up thread variable to no longer be a None
            self.cnn_ml_thread = thread
            return
        else:
            return

    def merge_cnn_thread_with_main(self, t):
        # generate a car wheel name
        if t.result() is not None:

            # this return a (frame, BB, frameID) = buffer_item. The argument (0,0) means the last item is returned.
            last_buffer_item = self.catchup_buffer.retrieve(0, 0)

            # add the new up to date BB object to quick_check object. quick_check holds the dictionary of car wheels.
            # Note: there is a last check in add_carWheel to make sure the car wheel can be seen in the current frame.
            self.quick_check.add_potential_carwheel(buffer_item=last_buffer_item, bb_obj=t.result())

        # the self.cnn_ml_thread is set to None to make it possible to launch a new CNN_ML thread
        self.cnn_ml_thread = None

    # code used for debugging. In the videoControler/main code, call this instance method to watch the video buffer.
    # It makes sens to do this after the main loop has completed.
    def showVideoBuffer(self):
        bufferQueue = self.catchup_buffer.getFullBuffer()
        for frame in bufferQueue:
            (img, BBleftovers, frameID) = frame
            cv2.imshow("The last frames saved", img)
            key = cv2.waitKey(1) & 0xFF

            # if the 'q' key is pressed, stop the loop
            if key == ord("q") or key == ord("Q"):
                break

            # this is a time delay to keep the display at roughly 60 fps. Might need tweaking is playback speed is off.
            time.sleep(0.016)
