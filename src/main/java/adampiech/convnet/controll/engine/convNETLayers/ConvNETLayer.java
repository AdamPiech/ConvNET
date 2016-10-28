package adampiech.convnet.controll.engine.convNETLayers;

import adampiech.convnet.controll.engine.convNETLayers.interfaces.CCNLayer;
import org.opencv.core.*;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import static java.lang.Math.*;
import static org.opencv.core.Core.*;
import static org.opencv.core.CvType.*;
import static org.opencv.imgcodecs.Imgcodecs.*;

/**
 * Created by Adam Piech on 2016-10-10.
 */

public class ConvNETLayer implements CCNLayer {

    private static final int TYPE_OF_MATRIX = CV_16S;
//    private static final int TYPE_OF_MATRIX = CV_64FC1;

    private List<Mat> processImage;
    private int receptiveField;
    private int depth;
    private int stride;
    private int zeroPadding;

    private int counter = 0;
    private String directory;

    public ConvNETLayer(Mat image, int receptiveField, int depth, int stride, int zeroPadding, String directory) {
        this.receptiveField = receptiveField;
        this.depth = depth;
        this.stride = stride;
        this.zeroPadding = zeroPadding;
        this.directory = directory;
        this.processImage = adjustImageData(image);
    }

    public ConvNETLayer(Mat[] image, int receptiveField, int depth, int stride, int zeroPadding, String directory) {
        this.receptiveField = receptiveField;
        this.depth = depth;
        this.stride = stride;
        this.zeroPadding = zeroPadding;
        this.directory = directory;
        this.processImage = adjustImageData(image);
    }

    @Override
    public Mat[] processLayer() {
        List<Mat[]> weights = createWeightsList();
        Mat[] biases = createBiasMatrix();
        Mat[] results = createResultMatrix();

        for (int depthIndex = 0; depthIndex < depth; depthIndex++) {
            processImage(processImage, weights.get(depthIndex), biases[depthIndex], results[depthIndex]);
            createTestImage(results[depthIndex]);
        }
        return results;
    }

    private void processImage(List<Mat> procesImg, Mat[] weights, Mat bias, Mat results) {
        for (int imgIndexX = 0; imgIndexX + receptiveField <= processImage.get(0).width(); imgIndexX += stride) {
            for (int imgIndexY = 0; imgIndexY + receptiveField <= processImage.get(0).height(); imgIndexY += stride) {
                int result = processChannel(procesImg, weights, imgIndexX, imgIndexY);
                results.put(imgIndexX / stride, imgIndexY / stride, max(result + (int) (bias.get(0, 0)[0]), 0));
            }
        }
    }

    private int processChannel(List<Mat> procesImg, Mat[] weights, int imgIndexX, int imgIndexY) {
        int result = 0;
        for (int channelIndex = 0; channelIndex < procesImg.size(); channelIndex++) {
            Mat mat = new Mat();
            multiply(procesImg.get(channelIndex).submat(imgIndexX, imgIndexX + receptiveField, imgIndexY, imgIndexY + receptiveField), weights[channelIndex], mat);
            result += (int) (sumElems(mat).val[0]);
        }
        return result;
    }

    private List<Mat> adjustImageData(Mat image) {
        List<Mat> procaessImg = new ArrayList<>();
        split(image, procaessImg);
        for (Mat channel : procaessImg) {
            channel.convertTo(channel, TYPE_OF_MATRIX);
            addZeroPadding(channel).copyTo(channel);
        }
        return procaessImg;
    }

    private List<Mat> adjustImageData(Mat[] image) {
        List<Mat> processImg = new ArrayList<>();
        for (Mat layer : image) {
            layer.convertTo(layer, TYPE_OF_MATRIX);
            addZeroPadding(layer).copyTo(layer);
            processImg.add(layer);
        }
        return processImg;
    }

    private Mat addZeroPadding(Mat imageLayer) {
        Mat imageLayerWithZeros = Mat.zeros(imageLayer.width() + zeroPadding * 2, imageLayer.height() + zeroPadding * 2, TYPE_OF_MATRIX);
        for (int indexX = 0; indexX < imageLayer.width(); indexX++) {
            for (int indexY = 0; indexY < imageLayer.height(); indexY++) {
//                double[] buff = new double[imageLayer.channels()];
                short[] buff = new short[imageLayer.channels()];
                imageLayer.get(indexX, indexY, buff);
                imageLayerWithZeros.put(indexX + zeroPadding, indexY + zeroPadding, buff);
            }
        }
        return imageLayerWithZeros;
    }

    private Mat[] createResultMatrix() {
        Mat[] weights = new Mat[depth];
        for (int index = 0; index < weights.length; index++) {
            weights[index] = new Mat(countNewLayerSize(), countNewLayerSize(), TYPE_OF_MATRIX);
        }
        return weights;
    }

    private List<Mat[]> createWeightsList() {
        List<Mat[]> weightsList = new ArrayList<>();
        for (int index = 0; index < depth; index++) {
            weightsList.add(index, createWeightMatrix());
        }
        return weightsList;
    }

    private Mat[] createWeightMatrix() {
        Mat[] weight = new Mat[processImage.size()];
        for (int index = 0; index < weight.length; index++) {
            weight[index] = new Mat(receptiveField, receptiveField, TYPE_OF_MATRIX);
            randu(weight[index], -1.0, 2.0);

//            Random random = new Random();
//            Scalar alpha = new Scalar(random.nextDouble());
//            Core.multiply(weight[index], alpha, weight[index]);

//            System.out.println(weight[index].dump());
//            System.out.println();
        }
        return weight;
    }

    private Mat[] createBiasMatrix() {
        Mat[] biases = new Mat[depth];
        for (int index = 0; index < biases.length; index++) {
            biases[index] = new Mat(1, 1, TYPE_OF_MATRIX);
            randu(biases[index], 0, 2);
        }
        return biases;
    }

    public int countNewLayerSize() {
        // return (processImage.get(0).width() - receptiveField + zeroPadding * 2) / stride + 1;
        return (processImage.get(0).width() - receptiveField) / stride + 1;
    }

    private void createTestImage(Mat result) {
        String path = "results" + File.separator;
        Mat image = new Mat();
        convertScaleAbs(result, image);

        System.out.println(directory + "_" + ++counter);
        new File(path + directory).mkdir();
        imwrite(path + directory + File.separator + directory + "_" + counter + ".png", image);
    }
}
