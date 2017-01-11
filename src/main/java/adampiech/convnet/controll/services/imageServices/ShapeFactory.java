package adampiech.convnet.controll.services.imageServices;

import adampiech.convnet.controll.utils.FiguresShapeGenerator;
import org.opencv.core.*;

import static org.opencv.core.CvType.*;
import static org.opencv.core.Mat.*;
import static org.opencv.imgproc.Imgproc.*;

/**
 * Created by Adam Piech on 2017-01-03.
 */

public class ShapeFactory {

    private final static int MAT_WIDTH = 112;
    private final static int MAT_HEIGHT = 112;
    private final static int MAT_TYPE = CV_8UC3;

    public Mat drawCross(int x1, int x2, int x3, int y1, int y2, int y3) {
        Mat mat = zeros(MAT_WIDTH, MAT_HEIGHT, MAT_TYPE);
        FiguresShapeGenerator shapeGenerator = new FiguresShapeGenerator();

        Point point1 = new Point(x1 - shapeGenerator.getScale() + shapeGenerator.getWidthOffset(), y2 + shapeGenerator.getHeightOffset());
        Point point2 = new Point(x3 + shapeGenerator.getScale() + shapeGenerator.getWidthOffset(), y2 + shapeGenerator.getHeightOffset());
        Point point3 = new Point(x2 + shapeGenerator.getWidthOffset(), y1 - shapeGenerator.getScale() + shapeGenerator.getHeightOffset());
        Point point4 = new Point(x2 + shapeGenerator.getWidthOffset(), y3 + shapeGenerator.getScale() + shapeGenerator.getHeightOffset());

        line(mat, point1, point2, shapeGenerator.getColor(), shapeGenerator.getThickness());
        line(mat, point3, point4, shapeGenerator.getColor(), shapeGenerator.getThickness());

        warpAffine(mat, mat, getRotationMatrix2D(new Point(MAT_WIDTH / 2, MAT_HEIGHT / 2), shapeGenerator.getAngle(), 1), new Size());
        blur(mat, mat, new Size(shapeGenerator.getBlur(), shapeGenerator.getBlur()));
//        resize(mat, mat, new Size(64, 64));

        return mat;
    }

    public Mat drawRectangle(int x1, int x2, int y1, int y2) {
        Mat mat = zeros(MAT_WIDTH, MAT_HEIGHT, MAT_TYPE);
        FiguresShapeGenerator shapeGenerator = new FiguresShapeGenerator();

        Point point1 = new Point(x1 - shapeGenerator.getScale() + shapeGenerator.getWidthOffset(), y1 - shapeGenerator.getScale() + shapeGenerator.getHeightOffset());
        Point point2 = new Point(x2 + shapeGenerator.getScale() + shapeGenerator.getWidthOffset(), y2 + shapeGenerator.getScale() + shapeGenerator.getHeightOffset());

        rectangle(mat, point1, point2, shapeGenerator.getColor(), shapeGenerator.getThickness());

        warpAffine(mat, mat, getRotationMatrix2D(new Point(MAT_WIDTH / 2, MAT_HEIGHT / 2), shapeGenerator.getAngle(), 1), new Size());
        blur(mat, mat, new Size(shapeGenerator.getBlur(), shapeGenerator.getBlur()));
//        resize(mat, mat, new Size(64, 64));

        return mat;
    }

    public Mat drawCircle(int x, int y, int radius) {
        Mat mat = zeros(MAT_WIDTH, MAT_HEIGHT, MAT_TYPE);
        FiguresShapeGenerator shapeGenerator = new FiguresShapeGenerator();

        Point point = new Point(x + shapeGenerator.getWidthOffset(), y + shapeGenerator.getHeightOffset());
        radius += Math.abs((shapeGenerator.getHeightOffset() + shapeGenerator.getWidthOffset() / 2));

        circle(mat, point, radius + shapeGenerator.getScale(), shapeGenerator.getColor(), shapeGenerator.getThickness());

        blur(mat, mat, new Size(shapeGenerator.getBlur(), shapeGenerator.getBlur()));
//        resize(mat, mat, new Size(64, 64));

        return mat;
    }

    public Mat drawTriangle(int x1, int x2, int x3, int y1, int y2, int y3) {
        Mat mat = zeros(MAT_WIDTH, MAT_HEIGHT, MAT_TYPE);
        FiguresShapeGenerator shapeGenerator = new FiguresShapeGenerator();

        Point point1 = new Point(x1 - shapeGenerator.getScale() + shapeGenerator.getWidthOffset(), y1 - shapeGenerator.getScale() + shapeGenerator.getHeightOffset());
        Point point2 = new Point(x2 + shapeGenerator.getWidthOffset(), y2 + shapeGenerator.getScale() + shapeGenerator.getHeightOffset());
        Point point3 = new Point(x3 + shapeGenerator.getScale() + shapeGenerator.getWidthOffset(), y3 - shapeGenerator.getScale() + shapeGenerator.getHeightOffset());

        line(mat, point1, point2, shapeGenerator.getColor(), shapeGenerator.getThickness());
        line(mat, point2, point3, shapeGenerator.getColor(), shapeGenerator.getThickness());
        line(mat, point3, point1, shapeGenerator.getColor(), shapeGenerator.getThickness());

        warpAffine(mat, mat, getRotationMatrix2D(new Point(MAT_WIDTH / 2, MAT_HEIGHT / 2), shapeGenerator.getAngle(), 1), new Size());
        blur(mat, mat, new Size(shapeGenerator.getBlur(), shapeGenerator.getBlur()));
//        resize(mat, mat, new Size(64, 64));

        return mat;
    }



}
