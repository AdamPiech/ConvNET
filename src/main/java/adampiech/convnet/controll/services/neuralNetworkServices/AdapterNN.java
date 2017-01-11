package adampiech.convnet.controll.services.neuralNetworkServices;

import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;

/**
 * Created by Adam Piech on 2016-12-21.
 */

public class AdapterNN {

    private int cnnDepth = 0;
    private int cnnRows = 0;
    private int cnnColumns = 0;

    public double[] cnnToANN(Matrix[] cnn) {

        cnnDepth = cnn.length;
        cnnRows = cnn[0].rows();
        cnnColumns = cnn[0].columns();

        double[] ann = new double[cnnDepth * cnnRows * cnnColumns];

        for (int depthIndex = 0; depthIndex < cnnDepth; depthIndex++) {
            for (int colIndex = 0; colIndex < cnnColumns; colIndex++) {
                for (int rowIndex = 0; rowIndex < cnnRows; rowIndex++) {
                    ann[rowIndex + colIndex + depthIndex] = cnn[depthIndex].get(rowIndex, colIndex);
                }
            }
        }

        return ann;
    }

    public Matrix[] annToCNN(double[] ann) {

        Matrix[] cnn = new Matrix[cnnDepth];
        for (int depthIndex = 0; depthIndex < cnnDepth; depthIndex++) {
            cnn[depthIndex] = new Basic2DMatrix(cnnRows, cnnColumns);
            for (int colIndex = 0; colIndex < cnnColumns; colIndex++) {
                for (int rowIndex = 0; rowIndex < cnnRows; rowIndex++) {
                    cnn[depthIndex].set(rowIndex, colIndex, ann[rowIndex + colIndex + depthIndex]);
//                    System.out.println(getClass().getName() + " --> " + "WEIGHT    ANN --> CNN:   " + ann[rowIndex + colIndex + depthIndex]);
                }
            }
        }

        return cnn;
    }
}
