# import the necessary packages
from collections import deque


class catchup_buffer:
    def __init__(self, bufSize=125, timeout=1.0):
        # store the maximum buffer size of frames to be kept
        # in memory along with the sleep timeout during threading
        self.bufSize = bufSize
        # initialize the buffer of frames, queue of frames that
        # need to be written to file, video writer, writer thread,
        # and boolean indicating whether recording has started or not
        self.buffer = deque(maxlen=bufSize)
        self.frameID = 0

    # this adds a new buffer item to the video buffer
    def update(self, frame, BBleftovers):
        # update the buffer
        if self.frameID == self.bufSize:
            self.frameID = 0  # this makes sure there are no naming issues, as the buffer just loops back over
        else:
            self.frameID += 1
        buffer_item = [frame, BBleftovers, self.frameID]
        # Note: we append the queue to the right, in doing so the oldest/leftest item is removed
        self.buffer.append(buffer_item)


    def retrieve(self,currentIDframe , IDframe):
        queue_back_log = currentIDframe - IDframe
        # MIGHT ADD A CHECK HERE TO SEE IF WE ARE AT CURRENT FRAME
        # the currentIDframe is the last item in the queue, so to get IDframe you must go backwards in the queue.
        buffer_item = self.buffer[-queue_back_log]
        return buffer_item

    def getFullBuffer(self):
        # the first item is the last one in with just self.buffer if you use appendLeft to create buffer
        return self.buffer

    def get_buffer_length(self):
        return self.bufSize

