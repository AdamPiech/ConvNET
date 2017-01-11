package adampiech.convnet.view;

/**
 * Created by Adam Piech on 2016-10-01.
 */

import adampiech.convnet.controll.engine.Training;
import adampiech.convnet.controll.engine.convNETLayers.ConvNETLayer;
import adampiech.convnet.controll.engine.convNETLayers.PoolingLayer;
import adampiech.convnet.controll.engine.convNETLayers.artificialNeuralNetwork.ANN;
import adampiech.convnet.controll.services.imageServices.ImageServices;
import adampiech.convnet.controll.services.imageServices.ShapeFactory;
import adampiech.convnet.controll.utils.TrainingData;
import org.la4j.Matrix;
import org.opencv.core.*;

import java.util.ArrayList;
import java.util.List;

import static adampiech.convnet.controll.services.imageServices.ImageServices.*;
import static adampiech.convnet.controll.services.matrixServices.MatrixServices.matToMatrix;
import static org.opencv.core.Core.split;

public class Main {

    private final static double[] CROSS_PATTERN = {0.0, 1.0};
    private final static double[] RECTANGLE_PATTERN = {1.0, 0.0};
    private final static double[] CIRCLE_PATTERN = {0.0, 0.0};
    private final static double[] TRIANGLE_PATTERN = {1.0, 1.0};

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {

        ShapeFactory factory = new ShapeFactory();
        int[] architecture = {256, 32, 2};

        List<TrainingData> trainingData = new ArrayList<>();
        for (int index = 0; index < 1500; index++) {
            trainingData.add(new TrainingData(prepareImage(factory.drawCross(20, 56, 92, 20, 56, 92)), CROSS_PATTERN));
            trainingData.add(new TrainingData(prepareImage(factory.drawRectangle(25, 88, 25, 88)), RECTANGLE_PATTERN));
            trainingData.add(new TrainingData(prepareImage(factory.drawCircle(56, 56, 30)), CIRCLE_PATTERN));
            trainingData.add(new TrainingData(prepareImage(factory.drawTriangle(26, 56, 86, 76, 26, 76)), TRIANGLE_PATTERN));
        }

        List<ConvNETLayer> convNET = new ArrayList<>();
        convNET.add(new ConvNETLayer(112, 3, 3, 4, 1, 0));
        convNET.add(new ConvNETLayer(55, 4, 4, 8, 1, 0));
        convNET.add(new ConvNETLayer(26, 8, 3, 16, 1, 0));
        convNET.add(new ConvNETLayer(12, 16, 3, 36, 1, 0));
        convNET.add(new ConvNETLayer(5, 36, 2, 64, 1, 0));
        PoolingLayer poolingLayer = new PoolingLayer(2, 2);
        ANN ann = new ANN(architecture);

        Training training = new Training();
        training.train(trainingData, convNET, poolingLayer, ann);

        System.out.println(Main.class.getName() + "--->" + "TEST CROSS");
        training.test(prepareImage(factory.drawCross(20, 56, 92, 20, 56, 92)), convNET, poolingLayer, ann);
        System.out.println(Main.class.getName() + "--->" + "TEST RECTANGLE");
        training.test(prepareImage(factory.drawRectangle(25, 88, 25, 88)), convNET, poolingLayer, ann);
        System.out.println(Main.class.getName() + "--->" + "TEST CIRCLE");
        training.test(prepareImage(factory.drawCircle(56, 56, 30)), convNET, poolingLayer, ann);
        System.out.println(Main.class.getName() + "--->" + "TEST TRIANGLE");
        training.test(prepareImage(factory.drawTriangle(26, 56, 86, 76, 26, 76)), convNET, poolingLayer, ann);
    }

    private static Matrix[] prepareImage(Mat mat) {
        List<Mat> list = new ArrayList<>();
        split(mat, list);
        return matToMatrix(list);
    }

}