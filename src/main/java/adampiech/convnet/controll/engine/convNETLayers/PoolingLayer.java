package adampiech.convnet.controll.engine.convNETLayers;

import adampiech.convnet.controll.engine.convNETLayers.interfaces.CCNLayer;
import org.opencv.core.Mat;

import static org.opencv.core.Core.*;

/**
 * Created by Adam Piech on 2016-10-17.
 */
public class PoolingLayer implements CCNLayer {

    private Mat[] layer;
    private int receptiveField;
    private int stride;

    public PoolingLayer(Mat[] layer, int receptiveField, int stride) {
        this.layer = layer;
        this.receptiveField = receptiveField;
        this.stride = stride;
    }

    @Override
    public Mat[] processLayer() {
        Mat[] results = new Mat[layer.length];
        for (int depthIndex = 0; depthIndex < layer.length; depthIndex++) {
            Mat result = new Mat(layer[0].width() / 2, layer[0].height() / 2, layer[0].type());
            processSlice(layer[depthIndex], result);
            results[depthIndex] = result;
        }
        return results;
    }

    private void processSlice(Mat mat, Mat result) {
        for (int indexX = 0; indexX + receptiveField <= layer[0].width(); indexX += stride) {
            for (int indexY = 0; indexY + receptiveField <= layer[0].height(); indexY += stride) {
                double value = minMaxLoc(mat.submat(indexX, indexX + receptiveField, indexY, indexY + receptiveField)).maxVal;
                result.put(indexX / 2, indexY / 2, value);
            }
        }
    }

    public int countNewLayerSize() {
        return (layer[0].width() - receptiveField) / stride + 1;
    }
}
