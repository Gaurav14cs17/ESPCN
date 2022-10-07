import glob
import cv2
import numpy as np


class VideoDemo:

    def __init__(self, input_left_dir, input_right_dir, output_path, start_frame, pause_frame, repeat_when_pause,
                 slide_step, line_width, frame_rate):
        self.paths_left = sorted(glob.glob(f'{input_left_dir}/*'))
        self.paths_right = sorted(glob.glob(f'{input_right_dir}/*'))
        self.output_path = output_path
        self.start_frame = start_frame
        self.pause_frame = pause_frame
        self.repeat_when_pause = repeat_when_pause
        self.slide_step = slide_step
        self.line_width = line_width
        self.frame_rate = frame_rate
        # initialize video writer
        self.video_writer = None

    def merge_images(self, img_left, img_right, x_coord):
        img_out = np.copy(img_left)
        img_out[:, x_coord:, :] = img_right[:, x_coord:, :]
        # add white line
        img_out[:, x_coord:x_coord + self.line_width, :] *= 0
        img_out[:, x_coord:x_coord + self.line_width, :] += 255
        return img_out

    def __call__(self):
        for i, (path_left, path_right) in enumerate(
                zip(self.paths_left, self.paths_right)):

            # start sliding
            if i >= self.start_frame:
                img_left = cv2.imread(path_left)
                img_right = cv2.imread(path_right)
                current_idx = self.slide_step * (i - self.start_frame)
                img_out = self.merge_images(img_left, img_right, current_idx)

            else:
                img_out = cv2.imread(path_left)

            if self.video_writer is None:
                h, w = img_out.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(self.output_path, fourcc,self.frame_rate, (w, h))
            self.video_writer.write(img_out.astype(np.uint8))

            # pause at somewhere
            if i == self.pause_frame:
                for _ in range(0, self.repeat_when_pause):
                    self.video_writer.write(img_out.astype(np.uint8))

        # pause before sliding over the last frame
        for _ in range(0, self.repeat_when_pause):
            self.video_writer.write(img_out.astype(np.uint8))

        # slide over the last frame
        w = img_out.shape[1]
        current_idx = min(current_idx, w - self.line_width)
        while current_idx + self.line_width >= 0:
            img_out = self.merge_images(img_left, img_right, current_idx)
            self.video_writer.write(img_out.astype(np.uint8))
            current_idx -= self.slide_step

        # pause before ending the demo
        self.video_writer.write(img_right.astype(np.uint8))
        for _ in range(0, self.repeat_when_pause):
            self.video_writer.write(img_right.astype(np.uint8))

        cv2.destroyAllWindows()
        self.video_writer.release()


if __name__ == '__main__':
    video_demo = VideoDemo(
        input_left_dir='data/demo_000',
        input_right_dir='data/demo_000',
        output_path='demo_video.mp4',
        start_frame=5,
        pause_frame=5,
        repeat_when_pause=25,
        slide_step=100,
        line_width=10,
        frame_rate=5,
    )
    video_demo()
