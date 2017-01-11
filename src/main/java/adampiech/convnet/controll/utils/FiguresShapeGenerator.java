package adampiech.convnet.controll.utils;

import org.opencv.core.Range;
import org.opencv.core.Scalar;

import java.util.Random;

/**
 * Created by Adam Piech on 2017-01-03.
 */

public class FiguresShapeGenerator {

    private Random random = new Random();

    private Scalar color;
    private int heightOffset;
    private int widthOffset;
    private int thickness;
    private int blur;
    private int scale;
    private int angle;

    public FiguresShapeGenerator() {
        color = generateRandomColor();
        heightOffset = generateOffset();
        widthOffset = generateOffset();
        thickness = generateThickness();
        blur = generateBlur();
        scale = generateScale();
        angle = generateAnge();
    }

    private Scalar generateRandomColor() {
//        return new Scalar(random.nextInt(255), random.nextInt(255), random.nextInt(255));
        return new Scalar(255, 255, 255);
    }

    private int generateOffset() {
        return (int) (random.nextGaussian() * 5.0);
//        return 0;
    }

    private int generateThickness() {
//        return (int) (Math.abs(random.nextGaussian() * 5.0) + 3.0);
        return 3;
    }

    private int generateBlur() {
//        return (int) (Math.abs(random.nextGaussian() * 6.0) + 3.0);
        return 2;
    }

    private int generateScale() {
        return (int) (random.nextGaussian() * 5.0 + 1.0);
//        return 1;
    }

    private int generateAnge() {
//        return random.nextInt(360);
//        return (int) (random.nextInt(180) * random.nextGaussian() / 20.0);
        return 0;
    }

    public Scalar getColor() {
        return color;
    }

    public int getHeightOffset() {
        return heightOffset;
    }

    public int getWidthOffset() {
        return widthOffset;
    }

    public int getThickness() {
        return thickness;
    }

    public int getBlur() {
        return blur;
    }

    public int getScale() {
        return scale;
    }

    public int getAngle() {
        return angle;
    }
}
