package adampiech.convnet.view;

/**
 * Created by Adam Piech on 2016-10-01.
 */

import adampiech.convnet.controll.imageServices.ImageServices;
import adampiech.convnet.controll.imageServices.MotionDetector;
import adampiech.convnet.controll.engine.convNETLayers.ConvNETLayer;
import adampiech.convnet.controll.engine.convNETLayers.FullConnectedLayer;
import adampiech.convnet.controll.engine.convNETLayers.PoolingLayer;
import adampiech.convnet.controll.engine.convNETLayers.interfaces.CCNLayer;
import org.opencv.core.*;

import static adampiech.convnet.controll.imageServices.ImageServices.*;
import static org.opencv.imgcodecs.Imgcodecs.*;

public class Main {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) throws InterruptedException {
//        Mat image = imread(Main.class.getClassLoader().getResource("people.jpg").getPath());
        Mat image = imread("C:\\Users\\Adam Piech\\IdeaProjects\\ConvNET\\src\\main\\resources\\anne.jpeg", IMREAD_COLOR);


//        MotionDetector.detectMovement(m -> imwrite("motion" + ".png", m));
        image = prepareImageToProcessing(image);
        Mat[] imgArr = recognizeShadows(image);


        CCNLayer convNETLayer64 = new ConvNETLayer(image, 4, 64, 2, 1, "CONV_64");
        Mat[] result = convNETLayer64.processLayer();

        CCNLayer poolingLayer = new PoolingLayer(result, 2, 2);
        result = poolingLayer.processLayer();

        CCNLayer convNETLayer128 = new ConvNETLayer(result, 3, 128, 1, 1, "CONV_128");
        result = convNETLayer128.processLayer();

        CCNLayer poolingLayer2 = new PoolingLayer(result, 2, 2);
        result = poolingLayer2.processLayer();

        CCNLayer convNETLayer256 = new ConvNETLayer(result, 3, 256, 1, 1, "CONV_256");
        result = convNETLayer256.processLayer();

        CCNLayer poolingLayer3 = new PoolingLayer(result, 2, 2);
        result = poolingLayer3.processLayer();

        FullConnectedLayer fullConnectedLayer = new FullConnectedLayer(result, 1024);
        int[] finishResult = fullConnectedLayer.processLayerMatToInt();

        FullConnectedLayer fullConnectedLayer2 = new FullConnectedLayer(finishResult, 200);
        finishResult = fullConnectedLayer2.processLayerIntToInt();

    }
}