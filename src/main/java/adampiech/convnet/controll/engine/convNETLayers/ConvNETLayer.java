package adampiech.convnet.controll.engine.convNETLayers;

import adampiech.convnet.controll.utils.Callback;
import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static adampiech.convnet.controll.services.matrixServices.MatrixServices.*;
import static java.lang.Math.*;

/**
 * Created by Adam Piech on 2016-11-14.
 */

public class ConvNETLayer {

    private Random rand = new Random();

    private Matrix[] processInput;
    private Matrix[] output;
    private List<Matrix[]> weights;
    private List<Double> biases;

    private int size;
    private int length;
    private int receptiveField;
    private int depth;
    private int stride;
    private int zeroPadding;
    private int newLayerSize;

    public ConvNETLayer(int size, int length, int receptiveField, int depth, int stride, int zeroPadding) {
        this.size = size;
        this.length = length;
        this.receptiveField = receptiveField;
        this.depth = depth;
        this.stride = stride;
        this.zeroPadding = zeroPadding;
        this.newLayerSize = countNewLayerSize();
        this.weights = createWeightsMatrix();
        this.biases = createBiasMatrix();
    }

    public Matrix[] processLayer(Matrix[] input, Callback callback) {
        processInput = adjustImageData(input);
        Matrix[] results = createResultMatrix();

        for (int depthIndex = 0; depthIndex < depth; depthIndex++) {
            for (int row = 0; row + receptiveField <= size; row += stride) {
                for (int col = 0; col + receptiveField <= size; col += stride) {
                    double result = 0.0;
                    for (int subLayer = 0; subLayer < processInput.length; subLayer++) {
                        result += arrayMultiplicationAndSum(weights.get(depthIndex)[subLayer],
                                processInput[subLayer].slice(row, col, row + receptiveField, col + receptiveField));
                    }
                    results[depthIndex].set(row / stride, col / stride, activationFunction(result + biases.get(depthIndex)));
                }
            }
            if (callback != null) {
                callback.setAction(results[depthIndex]);
            }
        }
        return output = results;
    }

    public Matrix[] backPropagation(Matrix[] delta) {
        BackPropagation backpropagation = new BackPropagation();
        return backpropagation.computeBackPropagation(processInput, output, delta, weights, biases, stride);
    }

    private Matrix[] adjustImageData(Matrix[] input) {
        Matrix[] processMatrix = new Matrix[input.length];
        for (int index = 0; index < input.length; index++) {
            processMatrix[index] = addZeroPadding(input[index]);
        }
        return processMatrix;
    }

    private Matrix addZeroPadding(Matrix matrixDepthLayer) {
        Matrix imageLayerWithZeros = Matrix.zero(matrixDepthLayer.rows() + zeroPadding * 2, matrixDepthLayer.columns() + zeroPadding * 2);
        for (int row = 0; row < matrixDepthLayer.rows(); row++) {
            for (int col = 0; col < matrixDepthLayer.columns(); col++) {
                imageLayerWithZeros.set(row + zeroPadding, col + zeroPadding, matrixDepthLayer.get(row, col));
            }
        }
        return imageLayerWithZeros;
    }

    private Matrix[] createResultMatrix() {
        Matrix[] matrixForResults = new Matrix[depth];
        for (int index = 0; index < depth; index++) {
            matrixForResults[index] = new Basic2DMatrix(newLayerSize, newLayerSize);
        }
        return matrixForResults;
    }

    private List<Matrix[]> createWeightsMatrix() {
        List<Matrix[]> weightsList = new ArrayList<>();
        for (int index = 0; index < depth; index++) {
            weightsList.add(index, createSingleWeightMatrix());
        }
        return weightsList;
    }

    private Matrix[] createSingleWeightMatrix() {
        Matrix[] weight = new Matrix[length];
        for (int index = 0; index < weight.length; index++) {
            weight[index] = new Basic2DMatrix(receptiveField, receptiveField);
            for (int row = 0; row < receptiveField; row++) {
                for (int col = 0; col < receptiveField; col++) {
                    weight[index].set(row, col, rand.nextDouble() * 2 - 1);
                }
            }
        }
        return weight;
    }

    private List<Double> createBiasMatrix() {
        List<Double> biases = new ArrayList<>(depth);
        for (int index = 0; index < depth; index++) {
            biases.add(rand.nextDouble());
        }
        return biases;
    }

    public int countNewLayerSize() {
        return (size - receptiveField + zeroPadding * 2) / stride + 1;
    }

    private double activationFunction(double value) {
        return max(value, 0.0);
    }

    public List<Matrix[]> getWeights() {
        return weights;
    }

    public void setWeights(List<Matrix[]> weights) {
        this.weights = weights;
    }

    public List<Double> getBiases() {
        return biases;
    }

    public void setBiases(List<Double> biases) {
        this.biases = biases;
    }

    public int getSize() {
        return size;
    }

    public void setSize(int size) {
        this.size = size;
    }

    public int getLength() {
        return length;
    }

    public void setLength(int length) {
        this.length = length;
    }

    public int getReceptiveField() {
        return receptiveField;
    }

    public void setReceptiveField(int receptiveField) {
        this.receptiveField = receptiveField;
    }

    public int getDepth() {
        return depth;
    }

    public void setDepth(int depth) {
        this.depth = depth;
    }

    public int getStride() {
        return stride;
    }

    public void setStride(int stride) {
        this.stride = stride;
    }

    public int getZeroPadding() {
        return zeroPadding;
    }

    public void setZeroPadding(int zeroPadding) {
        this.zeroPadding = zeroPadding;
    }

}
