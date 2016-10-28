package adampiech.convnet.controll.imageServices;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import static org.opencv.core.Core.cartToPolar;
import static org.opencv.core.Core.convertScaleAbs;
import static org.opencv.core.CvType.*;
import static org.opencv.imgcodecs.Imgcodecs.imwrite;
import static org.opencv.imgproc.Imgproc.*;

/**
 * Created by Adam Piech on 2016-10-28.
 */
public class ImageServices {

    public static Mat prepareImageToProcessing(Mat image) {
        Imgproc.resize(image, image, new Size(224, 224));
        blur(image, image, new Size(5, 5));
//        imwrite("results\\scaled.jpg", image);
        return image;
    }

    public static  Mat[] recognizeShadows(Mat image) {
        Mat grad_x = new Mat();
        Mat grad_y = new Mat();

        Scharr(image, grad_x, CV_32FC1, 1, 0);
        Scharr(image, grad_y, CV_32FC1, 0, 1);

        Mat magnitude = new Mat();
        Mat direction = new Mat();
        boolean useDegree = true;
        cartToPolar(grad_x, grad_y, magnitude, direction, useDegree);

//        Mat directionPrint = new Mat();
//        convertScaleAbs(direction, directionPrint);
//        imwrite("results\\direction.jpg", directionPrint);

//        Mat magnitudePrint = new Mat();
//        convertScaleAbs(magnitude, magnitudePrint);
//        imwrite("results\\magnitude.jpg", magnitudePrint);

        Mat[] imageArray = {direction, magnitude};
        return imageArray;
    }
}
