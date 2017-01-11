package adampiech.convnet.controll.engine.convNETLayers;

import org.la4j.Matrix;

import java.util.List;

import static adampiech.convnet.controll.services.matrixServices.MatrixServices.*;

/**
 * Created by Adam Piech on 2016-12-16.
 */

public class BackPropagation {

    private final static double ETA = -1.0;

    public Matrix[] computeBackPropagation(Matrix[] input, Matrix[] output, Matrix[] delta, List<Matrix[]> filters, List<Double> bias, int stride) {
        Matrix[] newDelta = copyMatrixArchitecture(input);
        for (int depthIndex = 0; depthIndex < delta.length; depthIndex++) {
            computeSingleLayerError(input, output[depthIndex], delta[depthIndex], newDelta, filters.get(depthIndex), bias.get(depthIndex), stride);
        }
        return newDelta;
    }

    private void computeSingleLayerError(Matrix[] input, Matrix outputSlice, Matrix delta, Matrix[] inputDelta, Matrix[] filter, double bias, int stride) {

        double deltaWeights = 0.0;
        double deltaBias = 0.0;

        for (int rowIndex = 0; rowIndex < delta.rows(); rowIndex++) {
            for (int colIndex = 0; colIndex < delta.columns(); colIndex++) {

                countNewDeltas(inputDelta, filter, delta.get(rowIndex, colIndex));

                deltaWeights += countNeuronDeltasWeight(subMatrix(input,
                        rowIndex * stride, colIndex * stride,
                        rowIndex * stride + filter[0].rows(), colIndex * stride + filter[0].columns()),
                        delta.get(rowIndex, colIndex));

                deltaBias += delta.get(rowIndex, colIndex);
            }
        }

        updateNeuronWeights(filter, deltaWeights);
        updateNeuronBias(bias, deltaBias);
    }

    private void countNewDeltas(Matrix[] inputDelta, Matrix[] filter, double delta) {

        double inputDepth = inputDelta.length;
        double inputRows = inputDelta[0].rows();
        double inputColumns = inputDelta[0].columns();

        double filterDepth = filter.length;
        double filterRows = filter[0].rows();
        double filterColumns = filter[0].columns();

        for (int depthInputIndex = 0; depthInputIndex < inputDepth - filterDepth; depthInputIndex++) {
            for (int rowInputIndex = 0; rowInputIndex < inputRows - filterRows; rowInputIndex++) {
                for (int colInputIndex = 0; colInputIndex < inputColumns - filterColumns; colInputIndex++) {

                    for (int depthFilterIndex = 0; depthFilterIndex < filterDepth; depthFilterIndex++) {
                        for (int rowFilterIndex = 0; rowFilterIndex < filterRows; rowFilterIndex++) {
                            for (int colFilterIndex = 0; colFilterIndex < filterColumns; colFilterIndex++) {

                                inputDelta[depthInputIndex + depthFilterIndex].set(rowInputIndex + rowFilterIndex, colInputIndex + colFilterIndex,
                                        inputDelta[depthInputIndex + depthFilterIndex].get(rowInputIndex + rowFilterIndex, colInputIndex + colFilterIndex)
                                        + filter[depthFilterIndex].get(rowFilterIndex, colFilterIndex) * delta);

                            }
                        }
                    }

                }
            }
        }
    }

    private Matrix[] updateNeuronWeights(Matrix[] filter, double weightsDelta) {

        double depth = filter.length;
        double rows = filter[0].rows();
        double columns = filter[0].columns();

        for (int depthIndex = 0; depthIndex < depth; depthIndex++) {
            for (int rowIndex = 0; rowIndex < rows; rowIndex++) {
                for (int colIndex = 0; colIndex < columns; colIndex++) {
                    double temp = filter[depthIndex].get(rowIndex, colIndex);
                    filter[depthIndex].set(rowIndex, colIndex, filter[depthIndex].get(rowIndex, colIndex) + weightsDelta);
                }
            }
        }

        return filter;
    }

    private double countNeuronDeltasWeight(Matrix[] inputFilterValues, double delta) {

        double depth = inputFilterValues.length;
        double rows = inputFilterValues[0].rows();
        double columns = inputFilterValues[0].columns();

        double sumSingleDeltaWeight = 0.0;

        for (int depthIndex = 0; depthIndex < depth; depthIndex++) {
            for (int rowIndex = 0; rowIndex < rows; rowIndex++) {
                for (int colIndex = 0; colIndex < columns; colIndex++) {
                    sumSingleDeltaWeight += countSingleDeltaWeight(inputFilterValues[depthIndex].get(rowIndex, colIndex), delta);
                }
            }
        }

        return sumSingleDeltaWeight / (depth * rows * columns);
    }

    private double countSingleDeltaWeight(double inputFilterSingleValue, double delta) {
        return ETA * delta * inputFilterSingleValue;
    }

    private double updateNeuronBias(double bias, double delta) {
        return ETA * bias * delta;
    }

}
