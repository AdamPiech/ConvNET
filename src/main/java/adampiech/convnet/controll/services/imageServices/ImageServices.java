package adampiech.convnet.controll.services.imageServices;

import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;

import java.io.File;

import static org.opencv.core.Core.*;
import static org.opencv.core.CvType.*;
import static org.opencv.imgcodecs.Imgcodecs.imwrite;
import static org.opencv.imgproc.Imgproc.*;

/**
 * Created by Adam Piech on 2016-10-28.
 */
public class ImageServices {

    private static int counter = 0;

    public static Mat prepareImageToProcessing(Mat image) {
        resize(image, image, new Size(56, 56));
        blur(image, image, new Size(3, 3));
        imwrite("results\\scaled.jpg", image);
        return image;
    }

    public static  Mat[] getHistogram(Mat image) {
        Mat grad_x = new Mat();
        Mat grad_y = new Mat();
        Scharr(image, grad_x, CV_32FC1, 1, 0);
        Scharr(image, grad_y, CV_32FC1, 0, 1);

        normalize(grad_x);
        normalize(grad_y);

        Mat magnitude = new Mat();
        Mat direction = new Mat();
        boolean useDegree = true;
        cartToPolar(grad_x, grad_y, magnitude, direction, useDegree);

        multiply(direction, new Scalar(2.8, 2.8, 2.8), direction);//?
//        resize(direction, direction, new Size(32, 32));//?
//        resize(magnitude, magnitude, new Size(32, 32));//?

//        convertScaleAbs(direction, direction);
//        convertScaleAbs(magnitude, magnitude);
//        imwrite("results\\direction.jpg", direction);
//        imwrite("results\\magnitude.jpg", magnitude);

        Mat[] imageArray = {direction, magnitude};
        return imageArray;
    }

    private static void normalize(Mat image) {
        for (int indexX = 0; indexX < image.width(); indexX++) {
            for (int indexY = 0; indexY < image.width(); indexY++) {
                double[] value = image.get(indexX, indexY);
                image.put(indexX, indexY,
                        55.0 > value[0] ? 0.0 : value[0],
                        55.0 > value[1] ? 0.0 : value[1],
                        55.0 > value[2] ? 0.0 : value[2]);
            }
        }
    }

    public static void saveImage(Mat mat, String directory) {
        String path = "results" + File.separator;
        Mat image = new Mat();
        convertScaleAbs(mat, image);

        new File(path + directory).mkdir();
        imwrite(path + directory + File.separator + directory + "_" + ++counter + ".png", image);
//        System.out.println(directory + "_" + counter);
    }

}
