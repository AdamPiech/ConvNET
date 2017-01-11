package adampiech.convnet.controll.utils;

import org.la4j.Matrix;

/**
 * Created by Adam Piech on 2016-12-28.
 */

public class TrainingData {

    private Matrix[] image;
    private double[] target;

    public TrainingData(Matrix[] image, double[] target) {
        this.image = image;
        this.target = target;
    }

    public Matrix[] getImage() {
        return image;
    }

    public void setImage(Matrix[] image) {
        this.image = image;
    }

    public double[] getTarget() {
        return target;
    }

    public void setTarget(double[] target) {
        this.target = target;
    }

}
