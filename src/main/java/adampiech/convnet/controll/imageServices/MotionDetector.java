package adampiech.convnet.controll.imageServices;

import adampiech.convnet.utils.Callback;
import org.opencv.core.*;
import org.opencv.videoio.VideoCapture;

import java.util.ArrayList;
import java.util.List;

import static org.opencv.core.Core.absdiff;
import static org.opencv.imgproc.Imgproc.*;
import static org.opencv.imgproc.Imgproc.boundingRect;

/**
 * Created by Adam Piech on 2016-10-28.
 */
public class MotionDetector {

    public static  void detectMovement(Callback callback) {
        VideoCapture camera = new VideoCapture();
        camera.open(0);

        Mat originalFrame = new Mat();
        Mat recentFrame = new Mat();
        Mat previousFrame = new Mat();

        while (true) {
            sleep();
            camera.read(originalFrame);

            originalFrame.copyTo(recentFrame);
            cvtColor(recentFrame, recentFrame, COLOR_BGR2GRAY, 1);
            blur(recentFrame, recentFrame, new Size(25, 25));

            if (previousFrame.empty()) {
                recentFrame.copyTo(previousFrame);
            }

            Mat deltaFrame = new Mat();
            absdiff(previousFrame, recentFrame, deltaFrame);
//            threshold(deltaFrame, deltaFrame, 25, 255, THRESH_BINARY);
            blur(deltaFrame, deltaFrame, new Size(50, 50));
            threshold(deltaFrame, deltaFrame, 25, 255, THRESH_BINARY);
            findMotion(callback, originalFrame, deltaFrame);
            recentFrame.copyTo(previousFrame);
        }
    }

    private static void findMotion(Callback callback, Mat originalFrame, Mat deltaFrame) {
        dilate(deltaFrame, deltaFrame, getStructuringElement(MORPH_CROSS, new Size(10, 10)));
        List<MatOfPoint> contours = new ArrayList<>();
        findContours(deltaFrame, contours, new Mat(), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        for (MatOfPoint points : contours) {
            createMotionImg(callback, originalFrame, points);
        }
    }

    private static void createMotionImg(Callback callback, Mat originalFrame, MatOfPoint points) {
        if (points.size().height > 50 || points.size().width > 50) {
            Rect rect = boundingRect(points);
            try {
                Mat subMatFrame = originalFrame.submat(rect.x, rect.x + rect.width, rect.y, rect.y + rect.height);
                callback.setAction(subMatFrame);
            } catch (CvException e) {
            }
        }
    }

    private static void sleep() {
        try {
            Thread.sleep(250);
        } catch (InterruptedException e) {
        }
    }
}
