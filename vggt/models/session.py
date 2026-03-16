import torch
import torch.nn.functional as F
from queue import Queue, Empty
from threading import Thread
import time

class ImageIteratorStreamer:
    def __init__(self):
        self.queue = Queue()

    def put(self, value):
        self.queue.put(value)

    def close(self):
        self.queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        x = self.queue.get()
        if x is None:
            raise StopIteration
        return x

class MoReSession:
    def __init__(self, model):
        self.model = model

    def generate(self, input_streamer, output_streamer):
        while True and input_streamer.running:
            try:
                item = input_streamer.take()
            except Empty:
                time.sleep(5)
                continue
            predictions = self.model(item)
            output_streamer.put(predictions)
