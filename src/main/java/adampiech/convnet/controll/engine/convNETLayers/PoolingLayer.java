package adampiech.convnet.controll.engine.convNETLayers;

import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;

/**
 * Created by Adam Piech on 2016-11-15.
 */
public class PoolingLayer {

    private Matrix[] matrix;
    private int receptiveField;
    private int stride;

    public PoolingLayer(int receptiveField, int stride) {
        this.receptiveField = receptiveField;
        this.stride = stride;
    }

    public Matrix[] processLayer(Matrix[] matrix) {
        Matrix[] results = new Matrix[matrix.length];
        for (int depth = 0; depth < matrix.length; depth++) {
            Matrix result = new Basic2DMatrix(matrix[0].rows() / 2, matrix[0].columns() / 2);
            processSlice(matrix[depth], result);
            results[depth] = result;
        }
        return results;
    }

    private void processSlice(Matrix matrix, Matrix result) {
        for (int row = 0; row + receptiveField <= matrix.rows(); row += stride) {
            for (int col = 0; col + receptiveField <= matrix.columns(); col += stride) {
                double value = matrix.slice(row, col, row + receptiveField, col + receptiveField).max();
                result.set(row / 2, col / 2, value);
            }
        }
    }

    public int countNewLayerSize() {
        return (matrix[0].rows() - receptiveField) / stride + 1;
    }

}
